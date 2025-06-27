"""
This script trains a Generative Adversarial Network (GAN) to generate
synthetic medical images for different glioma grades.
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l
from tqdm import tqdm
import lpips


def convert_images_to_tensors(df, grade, UCSF = False, batch_size = 256):
    '''
    Filters a dataframe by glioma grade and UCSF identifier (if specified), loads the corresponding preprocessed 
    images, converts them to PyTorch tensors, and returns a DataLoader.

    Parameters:
        df (pandas.DataFrame): Dataframe containing image file paths and labels.
        grade (str): Grade of images to filter ('normal', 'low', or 'high').
        UCSF (bool): If True and grade is 'low', filters only UCSF-labeled images. Default is False.
        batch_size (int): Batch size to use for the DataLoader. Default is 256.

    Returns:
        dataloader (torch.utils.data.DataLoader): DataLoader containing tensors of the selected images.
        grade (str): The grade string used for filtering.
    '''

    grade_dict = {'normal':0, 'low':1, 'high':2}

    # Narrow df based on grade
    if grade == 'low' and UCSF:
        grade_df = df[df['image_file_path'].str.contains('UCSF')]
    elif grade == 'low' and not UCSF:
        grade_df = df[(~df['image_file_path'].str.contains('UCSF')) & (df['label'] == grade_dict[grade])]
    else:
        grade_df = df[df['label'] == grade_dict[grade]]

    # Load images
    grade_images = []
    for img_path in grade_df['image_file_path']:
        img = np.load(img_path)
        img_tensor = torch.tensor(img).unsqueeze(0)
        grade_images.append(img_tensor)
    image_tensor = torch.stack(grade_images)

    dataset = TensorDataset(image_tensor)
    dataloader = DataLoader(dataset, batch_size = batch_size)

    return dataloader, grade


def visualize_real_data(data_iter, filename_png, preprocessed_folder):
    """
    Visualizes a 3x3 grid of grayscale images from a PyTorch DataLoader and saves the figure as a PNG file.

    Parameters:
        data_iter (torch.utils.data.DataLoader): DataLoader yielding batches of image tensors.
        filename_png (str): Name of the PNG file to save the visualization.
        preprocessed_folder (str): Path to the folder containing preprocessed images.

    Returns:
        None
    """

    images = next(iter(data_iter))[0][:9]

    fig, axes = plt.subplots(3, 3, figsize = (8, 8))
    axes = axes.flatten()

    for i in range(9):
        img = images[i]
        img_2d = img[0].cpu().numpy()
        axes[i].imshow(img_2d, cmap = 'gray', interpolation = 'bilinear')
        axes[i].axis('off')

    plt.tight_layout()

    file_path = os.path.join(preprocessed_folder, filename_png)
    plt.savefig(file_path, dpi = 300)
    plt.show()


def LPIPS_metric(net_G, real_data_iter, lpips_model, latent_dim, device = d2l.try_gpu()):
    """
    Computes the average LPIPS (Learned Perceptual Image Patch Similarity) score between real and generated 
    images to evaluate the perceptual similarity of a GAN generator's outputs.

    Parameters:
        net_G (torch.nn.Module): The generator model used to produce fake images.
        real_data_iter (torch.utils.data.DataLoader): DataLoader containing batches of real image tensors.
        lpips_model (lpips.LPIPS): Pretrained LPIPS model for perceptual distance measurement.
        latent_dim (int): Dimensionality of the generator's input noise vector.
        device (torch.device): Device to perform computation on. Defaults to GPU if available.

    Returns:
        average_lpips (float): The average LPIPS score across all batches.
    """

    total_lpips = 0.0
    n_batches   = 0
    with torch.no_grad():
        for real_batch in real_data_iter:
            if isinstance(real_batch, (list,tuple)):
                real_imgs = real_batch[0].to(device)
            else:
                real_batch.to(device)
            B = real_imgs.size(0)

            z = torch.randn(B, latent_dim, 1, 1, device = device)
            fake_imgs = net_G(z)

            d = lpips_model(real_imgs, fake_imgs)
            d = d.view(B, -1).mean(dim = 1)

            total_lpips += d.mean().item()
            n_batches   += 1

    average_lpips = total_lpips/n_batches
    print(average_lpips)
    
    return average_lpips


def visualize_fake_data(net_G, latent_dim, filename_png, gan_results_folder, device = d2l.try_gpu()):
    """
    Generates a 3x3 grid of fake images from a GAN generator using random latent vectors, and saves the visualization as a PNG file.

    Parameters:
        net_G (torch.nn.Module): Trained GAN generator model.
        latent_dim (int): Dimensionality of the latent space used to generate images.
        filename_png (str): Name of the PNG file to save the visualization.
        gan_results_folder (str): Path to the folder for saving GAN results.
        device (torch.device): Device to perform computation on. Defaults to GPU if available.

    Returns:
        None
    """

    Z = torch.randn(9, latent_dim, 1, 1, device = device)
    with torch.no_grad():
        fake = net_G(Z)
    fake_img = (fake + 1.0) / 2.0

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(9):
        img_i = fake_img[i, 0].cpu().numpy()
        axes[i].imshow(img_i, cmap = 'gray', interpolation = 'bilinear')
        axes[i].axis('off')
    plt.tight_layout()
    
    file_path = os.path.join(gan_results_folder, filename_png)
    plt.savefig(file_path, dpi = 300)
    plt.show()


def fake_filepaths_to_df(folder_path, grade, UCSF = False):
    """
    Creates a dataframe of file paths and labels for fake images generated by a GAN, filtered by grade and optionally by UCSF tag.

    Parameters:
        folder_path (str): Path to the folder containing .npy files of generated images.
        grade (str): Grade category to filter by ('normal', 'low', or 'high').
        UCSF (bool): If True, includes only files with 'UCSF' in the filename. Default is False.

    Returns:
        fake_df (pandas.DataFrame): A dataframe with columns 'image_file_path' and 'label', 
        where 'label' is an integer corresponding to the image grade (0 = normal, 1 = low-grade, 2 = high-grade).
    """

    best_results_files = os.listdir(folder_path)
    
    if UCSF:
        best_results_files = [f for f in best_results_files if 'UCSF' in f]

    grade_dict = {'normal':0, 'low':1, 'high':2}

    images = []
    for file_name in best_results_files:
        if file_name.endswith('.npy') and grade in file_name:
            label = grade_dict[grade]
            file_path = os.path.join(folder_path, file_name)
            images.append({'image_file_path': file_path, 'label':label})
    
    fake_df = pd.DataFrame(images)
    
    return fake_df


# DCGAN setup and training adapated from code provided in SIADS 642: Deep Learning
# Week 4- GAN Assignment, University of Michigan MADS 
class G_block(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size = 4, stride = 2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.batch = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU()

    def forward(self, X):

        X = self.conv(X)
        X = self.batch(X)
        X = self.activate(X)
        
        return X

net_G = nn.Sequential(        
    G_block(100, 1024, stride = 1, padding = 0),
    
    G_block(1024, 512),
    
    G_block(512, 256),
    
    G_block(256, 128),
    
    nn.ConvTranspose2d(128, 1, kernel_size = 4, stride = 2, padding = 1, bias = False),
    
    nn.Tanh())


class D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2,
                 padding=1, alpha = 0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.batch = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU(alpha)

    def forward(self, X):

        X = self.conv(X)
        X = self.batch(X)
        X = self.activate(X)
        
        return X

net_D = nn.Sequential(
    # No batchnorm for first layer    
    nn.Conv2d(1, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.LeakyReLU(0.2, inplace = True),
    
    D_block(64, 128),
    
    D_block(128, 256),
    
    D_block(256, 512),
    
    nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0)

    )


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, grade, gan_results_folder, best_results_folder, UCSF = False, device = d2l.try_gpu()):
    """
    Trains a DCGAN using the provided discriminator and generator networks, tracks generator quality via LPIPS score,
    saves intermediate and best-generated images, and returns a dataframe of the best outputs.

    Parameters:
        net_D (torch.nn.Module): The discriminator model.
        net_G (torch.nn.Module): The generator model.
        data_iter (torch.utils.data.DataLoader): DataLoader providing real image batches.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate for both generator and discriminator.
        latent_dim (int): Dimensionality of the latent vector input to the generator.
        grade (str): Glioma grade being modeled ('normal', 'low', or 'high').
        gan_results_folder (str): Path to the folder containing GAN results.
        best_results_folder (str): Path to the folder containing best results.
        UCSF (bool): Whether to restrict to UCSF subset of data. Default is False.
        device (torch.device): Device on which to run training. Defaults to GPU if available.

    Returns:
        lpips_dict (dict): Dictionary mapping epoch number to LPIPS score.
        fake_df (pandas.DataFrame): Dataframe containing file paths and labels of saved best fake images.
    """
    
    loss = nn.BCEWithLogitsLoss(reduction = 'sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)

    trainer_hp = {'lr': lr, 'betas': [0.5, 0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)

    animator = d2l.Animator(xlabel = 'epoch', ylabel = 'loss',
                            xlim = [1, num_epochs], nrows = 2, figsize = (5, 5),
                            legend = ['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace = 0.3)


    lpips_model = lpips.LPIPS(net = 'alex').to(device)
    lpips_model.eval()
    lpips_dict = {}
    for epoch in tqdm(range(1, num_epochs + 1)):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        for X_batch in data_iter:
            X = X_batch[0]
            batch_size = X.shape[0]
            X = X.to(device)

            # Update D once
            Z_D = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            loss_D = d2l.update_D(X, Z_D, net_D, net_G, loss, trainer_D)

            # Update G twice
            Z_G1 = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            loss_G1 = d2l.update_G(Z_G1, net_D, net_G, loss, trainer_G)

            Z_G2 = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            loss_G2 = d2l.update_G(Z_G2, net_D, net_G, loss, trainer_G)

            # Average G-loss over 2 updates
            loss_G = (loss_G1 + loss_G2)/2

            metric.add(loss_D, loss_G, batch_size)

        # Show generated examples
        Z = torch.normal(0, 1, size = (21, latent_dim, 1, 1), device = device)

        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat([torch.cat([fake_x[i * 7 + j].cpu().detach() for j in range(7)], dim = 1)
            for i in range(len(fake_x) // 7)], dim = 0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.squeeze(-1), cmap='gray')

        # Save images and LPIPS scores every 10 epochs
        best_score = 1
        if epoch % 10 == 0:  
            if UCSF:
                filename = f"GAN_{grade}_UCSF_snapshot_epoch_{epoch:03d}.png"
            else:
                filename = f"GAN_{grade}_snapshot_epoch_{epoch:03d}.png"
            filepath = os.path.join(gan_results_folder, filename)
            animator.fig.savefig(filepath, dpi = 300)

            # Caclulate LPIPS score and only save fake images if new score is better than best
            score = LPIPS_metric(net_G, data_iter, lpips_model, latent_dim)
            lpips_dict[epoch] = score
            if score < best_score:
                best_score = score
                filepath = os.path.join(best_results_folder, filename)
                animator.fig.savefig(filepath, dpi = 300)

                net_G.eval()
                with torch.no_grad():
                    if grade == 'low' and UCSF:
                        Z = torch.randn(56, latent_dim, 1, 1, device = device)
                    elif grade == 'low' and not UCSF:
                        Z = torch.randn(210, latent_dim, 1, 1, device = device)
                    else:
                        Z = torch.randn(266, latent_dim, 1, 1, device = device)
                    fakes = (net_G(Z) + 1) / 2
                    fakes = fakes.cpu().numpy()

                    for i, img in enumerate(fakes, start = 1):
                        if UCSF:
                            fname = f'fake_{grade}_UCSF_{i:03d}.npy'
                        else:
                            fname = f'fake_{grade}_{i:03d}.npy'
                        path  = os.path.join(best_results_folder, fname)
                        np.save(path, img)

                if UCSF:
                    grid_file_name = f'fake_{grade}_UCSF_9_samples.png'
                else:
                    grid_file_name = f'fake_{grade}_9_samples.png'
                visualize_fake_data(net_G, latent_dim, grid_file_name, gan_results_folder)
                
                net_G.train()

        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, 'f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')

    if loss_G1:
        num_Gs = '2G'
    else:
        num_Gs = '1G'
    file_path = os.path.join(gan_results_folder, f'{grade}_training_output_{lr}_{num_epochs}_{num_Gs}.png')
    plt.savefig(file_path, dpi = 300)  

    # Create df for 266 new images
    if UCSF:
        fake_df = fake_filepaths_to_df(best_results_folder, grade, UCSF = True)
    else:
        fake_df = fake_filepaths_to_df(best_results_folder, grade)

    d2l.plt.show()
    print('Done')

    return lpips_dict, fake_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train GANs to generate synthetic medical images.')
    parser.add_argument('input_file', type = str,
                        help = 'Path to the input CSV file from data preprocessing.')
    parser.add_argument('output_dir', type = str,
                        help = 'Directory to save the GAN results.')
    args = parser.parse_args()
    input_file = args.input_file
    gan_results_folder = args.output_dir

    all_grades_df = pd.read_csv(input_file)
    os.makedirs(gan_results_folder, exist_ok = True)
    best_results_folder = os.path.join(gan_results_folder, 'best_results')
    os.makedirs(best_results_folder, exist_ok = True)
    
    preprocessed_folder = os.path.dirname(input_file)

    # Train GANs for each grade
    grades_to_train = ['normal', 'low', 'high']
    all_fake_dfs = []
    
    for grade in grades_to_train:
        print(f"Training GAN for {grade} grade...")
        
        # Set parameters based on grade
        latent_dim = 100
        num_epochs = 300
        if grade == 'normal':
            lr = 0.0002
        else:
            lr = 0.0005
        
        # Train regular version
        data_iter, _ = convert_images_to_tensors(all_grades_df, grade)
        visualize_real_data(data_iter, f'{grade}_9_samples.png', preprocessed_folder)
        
        lpips_dict, fake_df = train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, grade, gan_results_folder, best_results_folder)
        
        # Save results
        lpips_df = pd.DataFrame(list(lpips_dict.items()), columns = ['Epoch', 'LPIPS'])
        lpips_df.to_csv(os.path.join(best_results_folder, f'lpips_{grade}_{lr}_{num_epochs}.csv'), index = False)
        fake_df.to_csv(os.path.join(best_results_folder, f'fake_{grade}_df.csv'), index = False)
        all_fake_dfs.append(fake_df)
        
        # Special case: also train UCSF version for low grade
        if grade == 'low':
            print("Training GAN for low grade UCSF...")
            
            data_iter_ucsf, _ = convert_images_to_tensors(all_grades_df, 'low', UCSF = True)
            visualize_real_data(data_iter_ucsf, 'low_grade_UCSF_9_samples.png', preprocessed_folder)
            
            lpips_dict_ucsf, fake_df_ucsf = train(net_D, net_G, data_iter_ucsf, num_epochs, lr, latent_dim, grade, gan_results_folder, best_results_folder, UCSF = True)
            
            # Save UCSF results
            lpips_df_ucsf = pd.DataFrame(list(lpips_dict_ucsf.items()), columns = ['Epoch', 'LPIPS'])
            lpips_df_ucsf.to_csv(os.path.join(best_results_folder, f'lpips_low_UCSF_{lr}_{num_epochs}.csv'), index = False)
            fake_df_ucsf.to_csv(os.path.join(best_results_folder, 'fake_low_grade_UCSF_df.csv'), index = False)
            all_fake_dfs.append(fake_df_ucsf)

    # Combine original and generated data
    all_fake_df = pd.concat(all_fake_dfs, ignore_index=True)
    original_and_gan_df = pd.concat([all_grades_df, all_fake_df], ignore_index = True)
    
    output_path = os.path.join(gan_results_folder, 'original_and_gan_df.csv')
    original_and_gan_df.to_csv(output_path, index = False)
    print(f'Combined dataframe saved to {output_path}')