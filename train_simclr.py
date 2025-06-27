"""
This script trains a SimCLR model for self-supervised learning on medical images.
The trained encoder is saved for downstream tasks.
"""
import argparse
import pandas as pd
import numpy as np
import os

from simclr import SimCLR
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from simclr.modules import NT_Xent, get_resnet
from simclr.modules.identity import Identity

from umap import UMAP
import matplotlib.pyplot as plt


# Create custom dataset for SimCLR data
class SimCLRTensorDataset(Dataset):
    def __init__(self, df, transform):
        self.image_paths = df['image_file_path'].tolist()  # Only store paths
        self.labels = df['label'].tolist()
        self.transform = transform
    
    def __getitem__(self, index):
        # Load image on-demand
        img = np.load(self.image_paths[index])  # Load only when needed
        img_2D = np.squeeze(img)
        img = torch.tensor(img_2D).unsqueeze(0).repeat(3, 1, 1)
        
        x_i = self.transform(img)
        x_j = self.transform(img)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        
        return ((x_i, x_j), y)
    
    def __len__(self):
        return len(self.image_paths)


class SimCLR(nn.Module):
    # This class definition has been taken from: https://github.com/Spijkervet/SimCLR/blob/master/simclr/simclr.py
    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        self.encoder.fc = Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias = False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias = False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


def convert_images_to_SimCLR_tensors(df, transform, batch_size = 28):
    '''
    Converts a dataframe of image file paths into a PyTorch DataLoader of transformed image tensors for SimCLR training.

    Parameter(s):
        df (pandas.DataFrame): Dataframe containing a column 'image_file_path' with paths to .npy image files.
        transform (torchvision.transforms.Compose): Composed set of data augmentation transforms to apply to each image.
        batch_size (int): Number of samples per batch in the DataLoader. Default is 28.

    Return(s):
        dataloader (torch.utils.data.DataLoader): DataLoader yielding batches of transformed image tensors for SimCLR.
    '''

    dataset = SimCLRTensorDataset(df, transform)
    dataloader = DataLoader(dataset, batch_size = batch_size)

    return dataloader


def get_resnet(name):
    """
    Returns a pretrained ResNet-50 model with ImageNet weights.

    Returns:
        model (torchvision.models.ResNet): A ResNet-50 model preloaded with ImageNet weights.
    """

    return torchvision.models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)


def train_SimCLR(dataloader, model, criterion, optimizer, simclr_folder):
    """
    Trains a SimCLR model using contrastive loss and saves the encoder weights.

    Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader yielding batches of positive image pairs (x_i, x_j).
        model (torch.nn.Module): SimCLR model that produces projection vectors from image pairs.
        criterion (torch.nn.Module): Contrastive loss function to minimize (e.g., NT-Xent).
        optimizer (torch.optim.Optimizer): Optimizer used to update the model's parameters.
        simclr_folder (str): Path to save the encoder weights.
    """

    # This function has been modified from: https://github.com/Spijkervet/SimCLR/blob/master/main.py
    total_loss = 0
    for step, ((x_i, x_j), _) in enumerate(dataloader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking = True)
        x_j = x_j.cuda(non_blocking = True)

        # positive pair, with encoding
        _, _, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()

        if step % 50 == 0:
            print(f'Step {step}/{len(dataloader)} — loss: {loss.item():.4f}')

    # save encoder weights
    file_name = os.path.join(simclr_folder, 'simclr_encoder.pth')
    torch.save(model.encoder.state_dict(), file_name)
    print(f'Finished, encoder saved to {file_name}. Average loss: {total_loss / len(dataloader):.4f}')


def plot_SimCLR_UMAP(model, dataloader, simclr_folder, n_components = 2, random_state = 42, figsize = (8, 8)):
    """
    Extracts feature embeddings from a SimCLR-trained encoder, reduces them with UMAP, 
    and visualizes the result as a 2D scatter plot colored by class labels.

    Parameters:
        model (torch.nn.Module): A SimCLR model with an `.encoder` attribute.
        dataloader (torch.utils.data.DataLoader): DataLoader yielding ((x_i, x_j), y) batches.
        simclr_folder (str): Path to save the UMAP visualization.
        n_components (int): Number of UMAP output dimensions. Default is 2.
        random_state (int): Seed for UMAP dimensionality reduction. Default is 42.
        figsize (tuple): Size of the matplotlib figure. Default is (8, 8).

    Returns:
        emb_2d (np.ndarray): 2D UMAP projection of the feature embeddings.
        labels (np.ndarray): Corresponding class labels for each embedding.
    """

    device = next(model.parameters()).device

    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for ((x_i, x_j), y) in dataloader:
            x = x_i.to(device)
            h = model.encoder(x)
            embeddings.append(h.cpu().numpy())
            labels.append(y.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # Reduce features with UMAP
    reducer = UMAP(n_components = n_components, random_state = random_state)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize = figsize)
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c = labels, cmap = 'tab10', s = 5, alpha = 0.7)
    plt.title('UMAP of SimCLR Features')
    plt.legend(*scatter.legend_elements(), title = 'Label')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()

    file_path = os.path.join(simclr_folder, 'simclr_results.png')
    plt.savefig(file_path, dpi = 300)
    plt.show()

    return emb_2d, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train a SimCLR model on medical images.')
    parser.add_argument('input_file', type = str,
                        help = 'Path to the input CSV file with image paths and labels.')
    parser.add_argument('output_dir', type = str,
                        help = 'Directory to save SimCLR results.')
    parser.add_argument('--batch_size', type = int, default = 28,
                        help = 'Batch size for training.')
    parser.add_argument('--lr', type = float, default = 3e-5,
                        help = 'Learning rate.')
    parser.add_argument('--projection_dim', type = int, default = 128,
                        help = 'Projection dimension for SimCLR.')
    args = parser.parse_args()
    simclr_folder = args.output_dir
    os.makedirs(simclr_folder, exist_ok = True)

    batch_size = args.batch_size
    lr = args.lr
    projection_dim = args.projection_dim
    input_file = args.input_file

    df = pd.read_csv(input_file)

    SimCLR_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale = (0.9, 1.0)),
        transforms.GaussianBlur(kernel_size = 5, sigma = (0.1, 0.5)),
        transforms.RandomAffine(degrees = 5, translate = (0.02, 0.02)),
    ])

    dataloader = convert_images_to_SimCLR_tensors(df, SimCLR_transform, batch_size = batch_size)

    encoder = get_resnet('resnet50')
    n_features = encoder.fc.in_features

    model = SimCLR(encoder, projection_dim = projection_dim, n_features = n_features).cuda()

    criterion = NT_Xent(batch_size = batch_size, temperature = 0.5, world_size = 1)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    train_SimCLR(dataloader, model, criterion, optimizer, simclr_folder)

    plot_SimCLR_UMAP(model, dataloader, simclr_folder)