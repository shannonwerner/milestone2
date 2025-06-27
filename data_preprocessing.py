"""
This script preprocesses raw NIfTI medical images by reorienting, resampling,
resizing, and normalizing them. It then saves the processed images and
creates a CSV file with file paths and corresponding labels.
"""
import argparse
import pandas as pd
import numpy as np
import os
import random
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def get_lists_of_files(normal_files, low_files, low_and_high_files, low_and_high_meta_df):
    """
    Filters and categorizes MRI file paths into normal, low-grade, and high-grade glioma groups based on file naming
    and metadata, removes irrelevant entries, and randomly samples the normal and high-grade files to match the 
    number of low-grade files for class balance.

    Parameters:
        normal_files (list of str): File path for normal brain MRIs.
        low_files (list of str): File paths for brains with low-grade gliomas.
        low_and_high_files (list of str): File paths for brains with low and high-grade gliomas.
        low_and_high_meta_df (pandas.DataFrame): DataFrame containing metadata for low-grade and high-grade gliomas.

    Returns:
        randomly_sampled_normal_files (list of str): Randomly sampled normal files, equal in count to low-grade files.
        low_files (list of str): All low-grade files including newly identified ones from the mixed list.
        randomly_sampled_high_files (list of str): Randomly sampled high-grade files, equal in count to low-grade files.
    """

    # Filter out any duplicate files and .DS_Store file
    normal_files = [f for f in normal_files if '.nii' in f]
    low_files = [f for f in low_files if '.nii' in f]
    low_and_high_files = [f for f in low_and_high_files if '_FU' not in f and '.nii' in f]

    # Categorize mixed files from low_and_high_files into correct categories
    high_files = []
    for f in low_and_high_files:
        id = str(f[:10] + f[11:14])
        grade = low_and_high_meta_df.loc[id]
        if grade == 2:
            low_files.append(f)
        else:
            high_files.append(f)
        
    # Randomly sample normal and high-grade files to match number of low-grade files
    n = len(low_files)
    randomly_sampled_normal_files = random.sample(normal_files, n)
    randomly_sampled_high_files = random.sample(high_files, n)

    return randomly_sampled_normal_files, low_files, randomly_sampled_high_files


def reorient_resample_resize_and_normalize(data, img, target_spacing = (1.0, 1.0, 1.0), target_size = (64, 64)):
    """
    Converts a 3D medical image to a normalized 2D axial slice by reorienting it to RAS, resampling to uniform voxel 
    spacing, resizing to a fixed shape, and applying min-max normalization to scale pixel values to the range [-1, 1].

    Parameters:
        data (numpy.ndarray): The raw image data array.
        img (nibabel.Nifti1Image): The loaded NIfTI image object containing affine and header metadata.
        target_spacing (tuple of float): Desired voxel spacing in mm along (x, y, z). Default is (1.0, 1.0, 1.0).
        target_size (tuple of int): Desired output size of the 2D axial slice as (height, width). Default is (64, 64).

    Returns:
        normalized_2d (numpy.ndarray): A 2D numpy array representing the normalized axial slice, with pixel values in the range [-1, 1].
    """

    # Reorient to RAS (right, anterior (front), superior (top))
    # Left to right along x-axis, back to front along y-axis, bottom to top along z-axis
    current_orientation = io_orientation(img.affine)
    target_orientation = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(current_orientation, target_orientation)
    reoriented_data = apply_orientation(data, transform)

    # Resample to target spacing
    current_spacing = img.header.get_zooms()[:3]
    zoom_factors = [c/t for c, t in zip(current_spacing, target_spacing)]
    resampled_data = zoom(reoriented_data, zoom_factors, order = 1)

    # Extract axial slice
    axial_slice = resampled_data[:, :, resampled_data.shape[2] // 2]

    # Resize to 
    scale_0 = target_size[0] / axial_slice.shape[0]
    scale_1 = target_size[1] / axial_slice.shape[1]
    resized_data = zoom(axial_slice, (scale_0, scale_1), order = 1)

    # Min/max normalization to [-1, 1]
    min, max = float(resized_data.min()), float(resized_data.max())
    if max - min < 1e-8:
        normalized_2d = np.zeros_like(resized_data, dtype = np.float32)
    else:
        normalized_0_1 = (resized_data - min) / (max - min)
        normalized_2d = (normalized_0_1 * 2.0 - 1.0).astype(np.float32)

    return normalized_2d


# Process images and put into a dataframe with grade label
def process_images_to_df(folder_path, files_list, preprocessed_folder, final_normal_files, final_low_files):
    """
    Loads and processes a list of MRI files by extracting a normalized 2D axial slice from each image, assigning a 
    glioma grade label (normal, low-grade, or high-grade), saving the preprocessed images as '.npy' files, and 
    returning a dataframe containing the file paths and corresponding labels.

    Parameters:
        folder_path (str): Path to the folder containing the raw '.nii' MRI files.
        files_list (list of str): List of file names to be processed.
        preprocessed_folder (str): Root directory where the preprocessed '.npy' files will be saved.
        final_normal_files (list of str): List of normal files.
        final_low_files (list of str): List of low-grade files.

    Returns:
        df (pandas.DataFrame): A dataframe with two columns— 'image_file_path' (str) and 'label' (int)—representing 
                                the path to the saved 2D slice and its corresponding class label (0 = normal, 
                                1 = low-grade, 2 = high-grade).
    """
    
    images = []
    counter = {'normal': 1, 'low_grade': 1, 'high_grade': 1}

    for i, f in enumerate(files_list):
        if '.nii' in f:
            file_path = os.path.join(folder_path, f)
            img = nib.load(file_path)
            data = img.get_fdata()
            axial_view = reorient_resample_resize_and_normalize(data, img)

            # Create grade label for images (0: normal, 1: low-grade, 2: high-grade)
            if f in final_normal_files:
                subfolder = 'preprocessed_normal'
                grade = 'normal'
                label = 0
            elif f in final_low_files:
                subfolder = 'preprocessed_low_grade'
                grade = 'low_grade'
                label = 1
            else:
                subfolder = 'preprocessed_high_grade'
                grade = 'high_grade'
                label = 2

            new_folder = os.path.join(preprocessed_folder, subfolder)
            os.makedirs(new_folder, exist_ok = True)

            if f in final_low_files and 'UCSF' in f:
                file_name = f'UCSF_{grade}{counter[grade]}.npy'
            else:
                file_name = f'{grade}{counter[grade]}.npy'
            new_file_path = os.path.join(new_folder, file_name)

            np.save(new_file_path, axial_view)
            images.append({'image_file_path': new_file_path, 'label':label})
            counter[grade] += 1

    df = pd.DataFrame(images)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocess medical imaging data.')
    parser.add_argument('input_dir', type = str,
                        help = 'Directory containing the raw image subfolders.')
    parser.add_argument('output_dir', type = str,
                        help = 'Directory to save the preprocessed data.')
    parser.add_argument('image_data_file', type = str,
                        help = 'Path to the image data CSV file.')
    args = parser.parse_args()

    input_dir = args.input_dir
    preprocessed_folder = args.output_dir
    image_data_file = args.image_data_file

    os.makedirs(preprocessed_folder, exist_ok=True)

    # Load dataframe and clean data
    low_and_high_meta_df = pd.read_csv(image_data_file, index_col = 'ID')
    low_and_high_meta_df.rename(columns = lambda x: x.strip(), inplace = True)

    # Delete entry for UCSF-PDGM-369 due to the file being corrupted
    low_and_high_meta_df = low_and_high_meta_df.drop('UCSF-PDGM-369')

    # Narrow to just ID and grade
    low_and_high_meta_df = low_and_high_meta_df['WHO CNS Grade']

    # Get folder paths for images
    low_and_high_folder_path = os.path.join(input_dir, 'low_and_high_grade')
    low_folder_path = os.path.join(input_dir, 'low_grade')
    normal_folder_path = os.path.join(input_dir, 'normal')
    all_images_path = os.path.join(input_dir, 'all_images')

    # Get sorted lists of files
    low_and_high_files = os.listdir(low_and_high_folder_path)
    low_files = os.listdir(low_folder_path)
    normal_files = os.listdir(normal_folder_path)

    final_normal_files, final_low_files, final_high_files = get_lists_of_files(
        normal_files, low_files, low_and_high_files, low_and_high_meta_df
    )

    # Combine all files and save df as csv
    all_files = final_normal_files + final_low_files + final_high_files

    all_grades_df = process_images_to_df(all_images_path, all_files, preprocessed_folder, final_normal_files, final_low_files)

    file_path = os.path.join(preprocessed_folder, 'all_grades_df.csv')
    all_grades_df.to_csv(file_path, index = False)
    
    print(f'Preprocessed data saved to {file_path}')

