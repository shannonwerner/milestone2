import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l
from tqdm import tqdm

import matplotlib.pyplot as plt


all_grades_df = pd.read_csv('../data/other_files/all_grades_df.csv')
# all_images = torchvision.datasets.ImageFolder('../data/processed_data')


# def convert_images_to_tensors(images, batch_size = 256):
#     ''' Takes a dataframe with different grades of images
#     and converts them to Pytorch tensors
    
#     Parameter:

#     Return:
#     '''

#     transformer = transforms.Compose([
#         transforms.Grayscale(num_output_channels = 1),
#         transforms.ToTensor()
#     ])
#     images.transform = transformer    

#     data_iter = torch.utils.data.DataLoader(images, batch_size = batch_size, shuffle = True)

#     return data_iter


def convert_images_to_tensors(df, grade, batch_size = 256):
    ''' Takes a dataframe with different grades of images
    and converts them to Pytorch tensors
    
    Parameter:

    Return:
    '''

    grade_dict = {'normal':0, 'low':1, 'high':2}

    grade_df = df[df['label'] == grade_dict[grade]]

    # load images
    grade_images = []
    for img_path in grade_df['image_file_path']:
        img = np.load(img_path)
        img_tensor = torch.tensor(img).unsqueeze(0)
        grade_images.append(img_tensor)
    image_tensor = torch.stack(grade_images)

    dataset = TensorDataset(image_tensor)
    dataloader = DataLoader(dataset, batch_size = batch_size)

    return dataloader


def visualize_data(data_iter, filename_png):
    """

    Parameters:

    Return:
    """
    images = next(iter(data_iter))[0][:9]

    fig, ax = plt.subplots(3, 3, figsize = (8, 8))
    ax = ax.flatten()

    for i in range(9):
        ax[i].imshow(images[i].squeeze().numpy(), cmap='gray', interpolation = 'bilinear')
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename_png, dpi = 300)


# build generator
class G_block(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size = 4, stride = 2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
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
    
    nn.ConvTranspose2d(128, 1, kernel_size = 4, stride = 2, padding = 1),
    
    nn.Tanh())


# build discriminator
class D_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2,
                 padding=1, alpha = 0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU(alpha)

    def forward(self, X):

        X = self.conv(X)
        X = self.batch(X)
        X = self.activate(X)
        
        return X
    
net_D = nn.Sequential(    
    D_block(1, 64),
    
    D_block(64, 128),
    
    D_block(128, 256),
    
    D_block(256, 512),
    
    nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0)

    )


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, device = d2l.try_gpu()):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)

    trainer_hp = {'lr': lr, 'betas': [0.5, 0.999]}
    trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
    trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)

    animator = d2l.Animator(xlabel = 'epoch', ylabel = 'loss',
                            xlim=[1, num_epochs], nrows = 2, figsize = (5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace = 0.3)

    for epoch in tqdm(range(1, num_epochs + 1)):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X_batch in data_iter:
            X = X_batch[0]
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       d2l.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
            
        # Show generated examples
        Z = torch.normal(0, 1, size=(21, latent_dim, 1, 1), device=device)

        # Normalize the synthetic data to N(0, 1)
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat([
            torch.cat([fake_x[i * 7 + j].cpu().detach()
                       for j in range(7)], dim=1)
            for i in range(len(fake_x) // 7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.squeeze(-1), cmap='gray')

        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec on {str(device)}')
    
    d2l.plt.show()
    

# make into pipeline
normal_data_iter = convert_images_to_tensors(all_grades_df, 'normal')
# low_grade_data_iter = convert_images_to_tensors(all_grades_df, 'low')
# high_grade_data_iter = convert_images_to_tensors(all_grades_df, 'high')

visualize_data(normal_data_iter, 'normal_9_samples.png')
# visualize_data(low_grade_data_iter, 'low_grade_9_samples.png')
# visualize_data(high_grade_data_iter, 'high_grade_9_samples.png')

latent_dim, lr, num_epochs = 100, 0.0002, 200
train(net_D, net_G, normal_data_iter, num_epochs, lr, latent_dim)



# ngpu = 1
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# real_label = 1
# fake_label = 0
# criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
# fixed_noise = torch.randn(64, 100, 1, 1, device = device)
# nz = 100


# # train the network
# def train_network(net_D, net_G, data_iter, num_epochs, lr, device = device):
#     img_list = []
#     G_losses = []
#     D_losses = []
#     iters = 0

#     print('Staring Training Loop...')

#     optimizerD = optim.Adam(net_D.parameters(), lr=lr, betas = (0.5, 0.999))
#     optimizerG = optim.Adam(net_G.parameters(), lr=lr, betas = (0.5, 0.999))

#     for epoch in range(num_epochs):
#         # For each batch in the dataloader
#         for i, data in enumerate(data_iter, 0):

#             ############################
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             ###########################
#             ## Train with all-real batch
#             net_D.zero_grad()
#             # Format batch
#             real_cpu = data[0].to(device)
#             b_size = real_cpu.size(0)
#             label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
#             # Forward pass real batch through D
#             output = net_D(real_cpu).view(-1)
#             # Calculate loss on all-real batch
#             errD_real = criterion(output, label)
#             # Calculate gradients for D in backward pass
#             errD_real.backward()
#             D_x = output.mean().item()

#             ## Train with all-fake batch
#             # Generate batch of latent vectors
#             noise = torch.randn(b_size, 100, 1, 1, device=device)
#             # Generate fake image batch with G
#             fake = net_G(noise)
#             label.fill_(fake_label)
#             # Classify all fake batch with D
#             output = net_D(fake.detach()).view(-1)
#             # Calculate D's loss on the all-fake batch
#             errD_fake = criterion(output, label)
#             # Calculate the gradients for this batch, accumulated (summed) with previous gradients
#             errD_fake.backward()
#             D_G_z1 = output.mean().item()
#             # Compute error of D as sum over the fake and the real batches
#             errD = errD_real + errD_fake
#             # Update D
#             optimizerD.step()

#             ############################
#             # (2) Update G network: maximize log(D(G(z)))
#             ###########################
#             net_G.zero_grad()
#             label.fill_(real_label)  # fake labels are real for generator cost
#             # Since we just updated D, perform another forward pass of all-fake batch through D
#             output = net_D(fake).view(-1)
#             # Calculate G's loss based on this output
#             errG = criterion(output, label)
#             # Calculate gradients for G
#             errG.backward()
#             D_G_z2 = output.mean().item()
#             # Update G
#             optimizerG.step()

#             # Output training stats
#             if i % 50 == 0:
#                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                     % (epoch, num_epochs, i, len(data_iter),
#                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

#             # Save Losses for plotting later
#             G_losses.append(errG.item())
#             D_losses.append(errD.item())

#             # Check how the generator is doing by saving G's output on fixed_noise
#             if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(data_iter)-1)):
#                 with torch.no_grad():
#                     fake = net_G(fixed_noise).detach().cpu()
#                 img_list.extend([img.cpu() for img in fake])

#             iters += 1

#             stacked_tensor = torch.stack(img_list)  # shape (N, 1, H, W)

#             # Wrap in a dataset and dataloader
#             new_dataset = TensorDataset(stacked_tensor)
#             new_dataloader = DataLoader(new_dataset, batch_size=16, shuffle=False)

#     return new_dataloader, G_losses, D_losses


# lr, num_epochs = 0.005, 20
# new_normal_data_iter, normal_G_losses, normal_D_losses = train_network(net_D, net_G, normal_data_iter, num_epochs, lr, device = device)
# visualize_data(new_normal_data_iter, 'new_normal_9_samples.png')
# total_images = len(new_normal_data_iter.dataset)
# print(total_images)

# plt.figure(figsize=(10,5))
# plt.title("Generator and Discriminator Loss During Training")
# plt.plot(normal_G_losses,label="G")
# plt.plot(normal_D_losses,label="D")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# print('Done')
