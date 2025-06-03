import pandas as pd
import numpy as np
import os
import random
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


# make this into function
# load dataframe and clean data
low_and_high_meta_df = pd.read_csv('../data/UCSF-PDGM-metadata_v2.csv', index_col = 'ID')
low_and_high_meta_df.rename(columns = lambda x: x.strip(), inplace=True)

# delete entry for UCSF-PDGM-369 due to the file being corrupted
low_and_high_meta_df = low_and_high_meta_df.drop('UCSF-PDGM-369')

# narrow to just ID and grade
low_and_high_meta_df = low_and_high_meta_df['WHO CNS Grade']

# get folder paths for images
low_and_high_folder_path = '../data/low_and_high_grade'
low_folder_path = '../data/low_grade'
normal_folder_path = '../data/normal'
all_images_path = '../data/all_images'

# get new folder path for images
new_folder_path = '../data/processed_data/'

# get list of files
low_and_high_files = os.listdir(low_and_high_folder_path)
low_files = os.listdir(low_folder_path)
normal_files = os.listdir(normal_folder_path)


def get_lists_of_files(normal_files, low_files, low_and_high_files):
    """ Process lists of files with different grades of gliomas
    into normal, low-grade, and high-grade and randomly sample
    to keep balance of classifications
    
    Parameters:

    Returns:
    """

    # filter out any duplicate files and .DS_Store file
    normal_files = [f for f in normal_files if '.nii' in f]
    low_files = [f for f in low_files if '.nii' in f]
    low_and_high_files = [f for f in low_and_high_files if '_FU' not in f and '.nii' in f]

    # categorize mixes files from low_and_high_files into correct categories
    all_low_files = low_files
    high_files = []
    for f in low_and_high_files:
        id = str(f[:10] + f[11:14])
        grade = low_and_high_meta_df.loc[id]
        if grade == 2:
            low_files.append(f)
        else:
            high_files.append(f)
        
    # randomly sample normal and high-grade files to match number of low-grade files
    n = len(low_files)
    randomly_sampled_normal_files = random.sample(normal_files, n)
    randomly_sampled_high_files = random.sample(high_files, n)

    return randomly_sampled_normal_files, low_files, randomly_sampled_high_files


def reorient_resample_resize_and_normalize(data, img, target_spacing = (1.0, 1.0, 1.0), target_size = (64, 64)):
    """ 

    Parameter:

    Return:
    """

    # reorient to RAS (right, anterior (front), superior (top))
    # left to right along x-axis, back to front along y-axis, top to bottom along z-axis
    current_orientation = io_orientation(img.affine)
    target_orientation = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(current_orientation, target_orientation)
    reoriented_data = apply_orientation(data, transform)

    # resample to target spacing
    current_spacing = img.header.get_zooms()[:3]
    zoom_factors = [c/t for c, t in zip(current_spacing, target_spacing)]
    resampled_data = zoom(reoriented_data, zoom_factors, order=1)

    # extract axial slice
    axial_slice = resampled_data[:, :, resampled_data.shape[2] // 2]

    # resize slice to target size
    scale_factors = (target_size[0] / axial_slice.shape[0], target_size[1] / axial_slice.shape[1])
    resized_slice = zoom(axial_slice, scale_factors, order=1)

    # z-score normalization
    mean = np.mean(resized_slice)
    std = np.std(resized_slice) + 1e-8
    normalized_slice = (resized_slice - mean) / std

    return normalized_slice.astype(np.float32)


# process images and put into a dataframe with grade label
def process_images_to_df(folder_path, files_list, new_folder_path):
    # put in input and output
    """Takes a folder path, a list of files, and a column title and returns
    a dataframe with one column that contains all the images
    
    Parameters:

    Returns:
    """
    
    images = []
    counter = {'normal': 1, 'low_grade': 1, 'high_grade': 1}

    for i, f in enumerate(files_list):
        if '.nii' in f:
            file_path = os.path.join(folder_path, f)
            img = nib.load(file_path)
            data = img.get_fdata()
            axial_view = reorient_resample_resize_and_normalize(data, img)

            # create grade label for images (0: normal, 1: low-grade, 2: high-grade)
            if f in final_normal_files:
                subfolder = 'processed_normal'
                grade = 'normal'
                label = 0
            elif f in final_low_files:
                subfolder = 'processed_low_grade'
                grade = 'low_grade'
                label = 1
            else:
                subfolder = 'processed_high_grade'
                grade = 'high_grade'
                label = 2

            new_folder = os.path.join(new_folder_path, subfolder)
            os.makedirs(new_folder, exist_ok = True)
            file_name = f'{grade}{counter[grade]}.npy'
            new_file_path = os.path.join(new_folder, file_name)

            # new_img = Image.fromarray(axial_view, mode = 'F')
            np.save(new_file_path, axial_view)
            images.append({'image_file_path': new_file_path, 'label':label})
            counter[grade] += 1

    df = pd.DataFrame(images)

    return df


# make this into pipeline
final_normal_files, final_low_files, final_high_files = get_lists_of_files(normal_files, low_files, low_and_high_files)
all_files = final_normal_files + final_low_files + final_high_files

all_grades_df = process_images_to_df(all_images_path, all_files, new_folder_path)

all_grades_df.to_csv('../data/other_files/all_grades_df.csv')

print('Done')


