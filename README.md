# Image Synthesis and Classification Project

This project focuses on generating and classifying medical images. It includes scripts for data preprocessing, training a Generative Adversarial Network (GAN) to synthesize new images, training a SimCLR model for representation learning, and training a CNN for image classification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shannonwerner/milestone2.git
   cd milestone2
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Managing Dependencies

The `requirements.txt` file is generated from `requirements.in` using `pip-compile`. If you need to add or update dependencies, modify `requirements.in` and then run the following command to regenerate `requirements.txt`:

```bash
make requirements
```

It is not necessary to run this on every build, only when `requirements.in` has been changed.

`pip-compile` was necessary because when the `d2l` dependency was added, its dependendies were incompatible with those of some other packages. Also, it is the reason why this project has only been tested with `python 3.11`.

## Data

### Data Sources, Formats and Preprocessing
- **Normal Images**: 
  - Website: https://brain-development.org/ixi-dataset/
  - Download: https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
- **UCSF Low-grade and High-grade Glioma Images**:
  - Website: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- **Other Low-grade Glioma Images**:
  - Website: https://figshare.com/articles/dataset/Diffuse_Low_grade_Glioma_Database/1550871

This project processes medical images in NIfTI format (`.nii`). The `data_preprocessing.py` script expects the raw data to be organized in a specific directory structure and uses a metadata file to assign labels.

The preprocessing pipeline involves several steps:
- **Reorientation**: Images are reoriented to the RAS (Right, Anterior, Superior) coordinate system.
- **Resampling**: Voxel spacing is standardized.
- **Resizing**: Images are resized to a consistent dimension.
- **Slice Extraction**: An axial slice is extracted from the 3D volume.
- **Normalization**: Pixel values are normalized to the range `[-1, 1]`.

The script categorizes images into three classes: `normal`, `low-grade` glioma, and `high-grade` glioma. It also performs class balancing by random sampling. The processed images are saved as NumPy arrays (`.npy` files), and a CSV file is generated to map file paths to their corresponding labels.

## Usage

The project uses a `Makefile` to simplify running the different components.

### Data Preprocessing

To preprocess the data, run:
```bash
make data_preprocessing
```
This will process the original data and save it in a format suitable for the models.

### Training the GAN

To train the GAN for image synthesis, run:
```bash
make gan
```
This command trains separate GAN models for each of the three classes: `normal`, `low-grade`, and `high-grade`. The script uses the Learned Perceptual Image Patch Similarity (LPIPS) metric to evaluate the quality of the generated images and saves the best-performing models. The synthetic images are saved as `.npy` files.

### Training the SimCLR Model

To train the SimCLR model for representation learning, run:
```bash
make train_simclr
```
This script trains a SimCLR model using a ResNet-50 encoder pre-trained on ImageNet. It employs a contrastive loss (NT-Xent) to learn meaningful representations from the images without relying on labels. The script applies various data augmentations to create pairs of views for the contrastive learning task.

After training, the script saves the trained encoder to `simclr_encoder.pth`. This encoder can then be used as a feature extractor. It also generates a UMAP plot to visualize the learned feature space, saving it as `simclr_results.png`.

### Training the Classifier

To train the image classification models, you can use one of the following commands:
```bash
make train_classifier_cnn
make train_classifier_vit
make train_classifier_mlp_mixer
```
This script takes the encoder trained by the SimCLR model (`simclr_encoder.pth`), freezes its weights, and attaches a new linear classifier head. It then trains this new classifier on the labeled image data. You can choose to train a CNN, a Vision Transformer (ViT), or an MLP-Mixer model.

After training and evaluation, it generates LIME (Local Interpretable Model-agnostic Explanations) plots for some of the test images. These plots help to visualize which parts of an image the model is using to make its predictions, providing insights into the model's decision-making process.

### Running All Steps

To run the main pipeline (preprocessing, GAN, SimCLR, and all classifiers), you can use:
```bash
make all
```

### Cleaning Up

To remove all generated files and directories, run:
```bash
make clean
```

## Project Structure

- `data_preprocessing.py`: Script for preprocessing the input data.
- `gan.py`: Script for training the Generative Adversarial Network.
- `train_simclr.py`: Script for training the SimCLR model.
- `train_classifier.py`: Script for training a classifier on the learned representations.
- `Makefile`: Contains commands to run the project pipeline.
- `requirements.in`: A list of Python dependencies for the project.
- `requirements.txt`: A list of Python dependencies for the project, generated from `requirements.in`.
