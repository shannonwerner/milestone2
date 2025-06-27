"""
This script trains a classifier on medical image data, evaluates its performance,
and provides LIME explanations for its predictions.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from lime import lime_image

import timm

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models import ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

from simclr.modules.identity import Identity

from skimage.segmentation import mark_boundaries, slic

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def train_test_val_split(file_path):
    """
    Splits a labeled dataset of image file paths into stratified training, validation, and test sets.

    Parameter(s):
        file_path (str): Path to a CSV file containing columns 'image_file_path' and 'label'.

    Return(s):
        train_df (pandas.DataFrame): DataFrame containing 80% of the data for training.
        val_df (pandas.DataFrame): DataFrame containing 10% of the data for validation.
        test_df (pandas.DataFrame): DataFrame containing 10% of the data for testing.
    """

    df = pd.read_csv(file_path)
    df = df[~df['image_file_path'].str.contains('UCSF')]

    X = df['image_file_path']
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, train_size = 0.8, stratify = y, random_state = 42)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, train_size = 0.5, stratify = y_temp, random_state = 42)

    train_df = pd.DataFrame({'image_file_path': X_train, 'label': y_train})
    val_df   = pd.DataFrame({'image_file_path': X_val, 'label': y_val})
    test_df  = pd.DataFrame({'image_file_path': X_test, 'label': y_test})

    return train_df, val_df, test_df


class ClassifierTensorDataset(Dataset):
    def __init__(self, df):
        # Store only file paths
        self.image_paths = df['image_file_path'].tolist()
        self.labels = df['label'].tolist()

    def __getitem__(self, index):
        # Lazy load image
        img_path = self.image_paths[index]
        img = np.load(self.image_paths[index]) 
        img = torch.tensor(np.squeeze(img)).unsqueeze(0)
        img = img.repeat(3, 1, 1)

        img = TF.resize(img, size = [224, 224])

        label = torch.tensor(self.labels[index], dtype = torch.long)
        return img, label, img_path

    def __len__(self):
        return len(self.image_paths)
    

def convert_images_to_tensors(df, batch_size = 256):
    """
    Converts a dataframe of image file paths into a PyTorch DataLoader for use in classification or evaluation tasks.

    Parameter(s):
        df (pandas.DataFrame): DataFrame containing at least a column 'image_file_path' with paths to .npy image files, and optionally a 'label' column.
        batch_size (int): Number of samples per batch in the DataLoader. Default is 256.

    Return(s):
        dataloader (torch.utils.data.DataLoader): DataLoader yielding batches of image tensors (and labels if available).
    """

    dataset = ClassifierTensorDataset(df)
    dataloader = DataLoader(dataset, batch_size = batch_size)

    return dataloader


def get_encoder(model_type = 'cnn'):
    """
    Loads a pretrained encoder model (ResNet-50, ViT, or MLP-Mixer) with its classification head removed, 
    and returns the model along with its feature dimensionality.

    Parameters:
        model_type (str): Type of encoder to load. Options are:
            - 'cnn' for ResNet-50
            - 'vit' for Vision Transformer (ViT-B/16)
            - 'mlp_mixer' for MLP-Mixer (mixer_b16_224 from timm)
            Default is 'cnn'.

    Returns:
        model (torch.nn.Module): Pretrained model with its classification head replaced by an identity layer.
        feat_dim (int): Dimensionality of the output feature embeddings from the encoder.
    """

    if model_type == 'cnn':
        model = torchvision.models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = model.fc.in_features
        model.fc = Identity()
    elif model_type == 'vit':
        model = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = model.heads.head.in_features
        model.heads.head = Identity()
    elif model_type == 'mlp_mixer':
        model = timm.create_model('mixer_b16_224', pretrained = True)
        feat_dim = model.head.in_features
        model.head = Identity()
    
    return model, feat_dim
    

class SimpleClassifier(nn.Module):
    def __init__(self, encoder, feat_dim, num_classes, freeze_encoder = True):
        super(SimpleClassifier, self).__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, X):
        h = self.encoder(X)

        h = h.view(h.size(0), -1)

        return self.classifier(h)
    

def eval_model(model, data_loader):
    """
    Evaluates a trained classification model on a given dataset and returns accuracy, detailed metrics, 
    a confusion matrix, and a dataframe of misclassified examples.

    Parameters:
        model (torch.nn.Module): Trained PyTorch classification model.
        data_loader (torch.utils.data.DataLoader): DataLoader yielding batches of (x, y, file_path).

    Returns:
        acc (float): Overall classification accuracy.
        metrics_df (pandas.DataFrame): DataFrame containing precision, recall, f1-score, and support for each class.
        conf_matrix (numpy.ndarray): Confusion matrix comparing true and predicted labels.
        prediction_df (pandas.DataFrame): DataFrame of misclassified samples with columns: 'image_file_path', 'label', and 'predicted_label'.
    """

    model.eval()
    y_true_list = []
    y_pred_list = []
    path_list = []

    model.eval()
    for x,y, file_path in data_loader:
        x, y = x.cuda(), y.cuda()
        outputs = model(x)
        _, y_pred = torch.max(outputs, 1)
        y_pred_list.extend(y_pred.cpu().clone().detach().tolist())
        y_true_list.extend(y.cpu().clone().detach().tolist())
        path_list.extend(file_path)

    acc = classification_report(y_true_list, y_pred_list, output_dict = True)['accuracy']
    metrics_report = classification_report(y_true_list, y_pred_list, output_dict = True)
    metrics_df = pd.DataFrame(metrics_report).transpose()
    conf_matrix = confusion_matrix(y_true_list, y_pred_list)

    prediction_df = pd.DataFrame({'image_file_path':path_list, 'label':y_true_list, 'predicted_label':y_pred_list})
    prediction_df = prediction_df[prediction_df['label'] != prediction_df['predicted_label']]

    return acc, metrics_df, conf_matrix, prediction_df


def plot_confusion_matrix(conf_matrix, file_name, classifier_dir, class_names = None):
    """
    Plots and saves a confusion matrix as a heatmap.

    Parameters:
        conf_matrix (numpy.ndarray): Confusion matrix array (square, shape = [n_classes, n_classes]).
        file_name (str): Name of the output PNG file (e.g., 'confusion_matrix.png').
        class_names (list of str, optional): List of class labels to display on axes. Default is None.

    Returns:
        None
    """

    disp = ConfusionMatrixDisplay(conf_matrix, display_labels = class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax = ax, cmap = 'Blues', colorbar = True)
    ax.set_xlabel('Predicted label', labelpad = 15)
    ax.set_ylabel('True label', labelpad = 15)

    file_path = os.path.join(classifier_dir, file_name)
    plt.savefig(file_path, dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close() 


def plot_val_accuracy(val_acc_list, lr, weight_decay, model_type, classifier_dir):
    """
    Plots and saves a line chart of validation accuracy over epochs for a given model and hyperparameter setting.

    Parameters:
        val_acc_list (list of tuples): List of (epoch, validation_accuracy) pairs.
        lr (float): Learning rate used in training.
        weight_decay (float): Weight decay used in training.
        model_type (str): Model identifier used in the output filename (e.g., 'cnn', 'vit').

    Returns:
        None
    """

    epochs = [e for e, _ in val_acc_list]
    accs = [a for _, a in val_acc_list]

    plt.plot(epochs, accs, marker = 'o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy vs Epochs (Best: lr={lr}, wd={weight_decay})')
    plt.grid(True)
    plt.tight_layout()

    file_path = os.path.join(classifier_dir, f'{model_type}_val_acc_vs_epochs.png')
    plt.savefig(file_path, dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()   


def train_classifier(train_loader, val_loader, encoder_checkpoint, model_type, lrs, weight_decay_options,
                     classifier_dir, loss_function = nn.CrossEntropyLoss(), num_epochs = 20, momentum = 0.9):
    """
    Performs a grid search over learning rates and weight decay values to train a classifier using a frozen pretrained encoder.
    Evaluates the model every 5 epochs on the validation set, tracks the best performing configuration, and saves a plot of
    validation accuracy over time.

    Parameters:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        encoder_checkpoint (str): Path to the saved encoder weights (.pth file).
        model_type (str): Encoder architecture type ('cnn', 'vit', or 'mlp_mixer').
        lrs (list of float): List of learning rates to try.
        weight_decay_options (list of float): List of weight decay values to try.
        loss_function (torch.nn.Module): Loss function to use. Default is CrossEntropyLoss.
        num_epochs (int): Number of training epochs. Default is 20.
        momentum (float): Momentum for SGD optimizer. Default is 0.9.

    Returns:
        best_model (torch.nn.Module): Best-performing classifier model.
        best_val_acc (float): Highest validation accuracy achieved.
        val_metrics_df (pandas.DataFrame): Validation metrics (precision, recall, f1-score).
        val_conf_matrix (np.ndarray): Confusion matrix for the best model.
        val_prediction_df (pandas.DataFrame): DataFrame of misclassified validation samples.
        model_type (str): The model type used, passed through for reference.
    """

    best_val_acc = 0.0
    best_model = None
    best_lr = None
    best_weight_decay = None

    for lr in lrs:
        for weight_decay in weight_decay_options:
            # Build & load encoder
            encoder, feat_dim = get_encoder(model_type)
            encoder.load_state_dict(torch.load(encoder_checkpoint), strict = False)
            encoder = encoder.cuda()

            # Wrap in classifier
            model = SimpleClassifier(encoder, feat_dim = feat_dim, num_classes = 3, freeze_encoder = True).cuda()

            # Optimizer for only the trainable params
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = lr, momentum = momentum, weight_decay = weight_decay
            )

            # Training loop
            val_acc_list = []
            for epoch in range(1, num_epochs + 1):
                model.train()

                total_loss = 0
                for X, y, file_path in train_loader:
                    X, y = X.cuda(), y.cuda()
                    optimizer.zero_grad()
                    loss = loss_function(model(X), y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                print(f"Epoch {epoch}, LR {lr}, WD {weight_decay}, Train Loss: {total_loss:.4f}")

                # Evaluate every 5 epochs
                if epoch % 5 == 0:
                    val_acc, val_metrics_df, val_conf_matrix, val_prediction_df = eval_model(model, val_loader)
                    val_acc_list.append((epoch, val_acc))

                    print(f"Epoch {epoch}, Val Acc: {val_acc:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc   = val_acc
                        best_model = copy.deepcopy(model)
                        best_lr = lr
                        best_weight_decay = weight_decay
                        best_val_acc_list = val_acc_list.copy()

    plot_val_accuracy(best_val_acc_list, best_lr, best_weight_decay, model_type, classifier_dir)

    print(f'Best Learning Rate: {best_lr}')
    print(f'Best Weight Decay: {best_weight_decay}')

    return best_model, best_val_acc, val_metrics_df, val_conf_matrix, val_prediction_df, model_type


def save_val_results(best_val_acc, val_metrics_df, val_conf_matrix, val_prediction_df, model_type, classifier_dir):
    """
    Saves evaluation outputs from the validation set, including the confusion matrix image, 
    validation metrics as a CSV, and misclassified predictions as a CSV. Prints best accuracy.

    Parameters:
        best_val_acc (float): Best validation accuracy achieved.
        val_metrics_df (pandas.DataFrame): DataFrame with precision, recall, and F1-score for each class.
        val_conf_matrix (np.ndarray): Confusion matrix array.
        val_prediction_df (pandas.DataFrame): DataFrame containing misclassified validation examples.
        model_type (str): Model architecture name used to generate output filenames.

    Returns:
        None
    """

    class_names = ['Normal', 'Low-grade', 'High-grade']

    # Save results from validation set
    plot_confusion_matrix(val_conf_matrix, f'{model_type}_val_confusion_matix.png', classifier_dir, class_names)

    val_metrics_file_path = os.path.join(classifier_dir, f'{model_type}_val_metrics_df.csv')
    val_metrics_df.to_csv(val_metrics_file_path)

    val_prediction_file_path = os.path.join(classifier_dir, f'{model_type}_val_prediction_df.csv')
    val_prediction_df.to_csv(val_prediction_file_path)

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')


def predict_fn(images, model):
    """
    Runs inference on a batch of images using a trained model and returns softmax probabilities.

    Parameters:
        images (np.ndarray): Array of images with shape (N, H, W, C) and pixel values in [0, 255].
        model (torch.nn.Module): Trained PyTorch model for classification.

    Returns:
        np.ndarray: Array of softmax probabilities with shape (N, num_classes).
    """

    images = torch.tensor(images.transpose(0, 3, 1, 2)).float()/255.0
    images = F.interpolate(images, size=(224, 224), mode = 'bilinear', align_corners = False)
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim = 1)
    return probs.detach().numpy()


def LIME_explanation(model, prediction_df, model_type, classifier_dir):
    """
    Generates and saves LIME visual explanations for misclassified images in the given prediction DataFrame.

    Parameters:
        model (torch.nn.Module): Trained PyTorch model used for prediction.
        prediction_df (pandas.DataFrame): DataFrame containing misclassified samples, with columns 
                                          'image_file_path', 'label', and 'predicted_label'.
        model_type (str): Identifier for the model type (used in output filenames).

    Returns:
        None
    """

    model.cpu()
    model.eval()

    explainer = lime_image.LimeImageExplainer()

    for idx, row in prediction_df.iterrows():
        img = np.load(row['image_file_path'])
        img = img.squeeze()

        # Get rid of extra space
        coords = np.argwhere(img)
        y0, x0 = coords.min(axis = 0)
        y1, x1 = coords.max(axis = 0) + 1
        img = img[y0:y1, x0:x1]

        img_rgb = np.stack([img] * 3, axis = -1)

        img_rgb = (img_rgb - img_rgb.min())/(img_rgb.max() - img_rgb.min() + 1e-8)
        img_rgb = (img_rgb * 255).astype(np.uint8)

        explanation = explainer.explain_instance(
            img_rgb,
            lambda x: predict_fn(x, model),
            top_labels = 1,
            hide_color = 0,
            num_samples = 1000,
            segmentation_fn=lambda x: slic(x, n_segments = 100, compactness = 1, sigma = 1)
        )

        temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only = False,
        num_features = 5,
        hide_rest = False
        )

        fig, ax = plt.subplots()
        ax.imshow(mark_boundaries(temp, mask))
        ax.axis('off')

        file_name = row['image_file_path'].split('/')[-1]
        ax.set_title(f"{file_name}- True: {row['label']}, Pred: {row['predicted_label']}")

        file_path = os.path.join(classifier_dir, f'{model_type}_lime_{idx}.png')
        plt.savefig(file_path, bbox_inches = 'tight', dpi = 300)
        plt.show()
        plt.clf()
        plt.close()


def save_test_results(best_model, test_loader, model_type, classifier_dir):
    """
    Evaluates the best model on the test set, saves the confusion matrix and performance metrics,
    stores misclassified predictions, and generates LIME explanations.

    Parameters:
        best_model (torch.nn.Module): Trained classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader containing test set samples.
        model_type (str): Identifier for the model architecture (used for output filenames).

    Returns:
        None
    """
    
    class_names = ['Normal', 'Low-grade', 'High-grade']

    # Get and save results from test set
    test_acc, test_metrics_df, test_conf_matrix, test_prediction_df = eval_model(best_model, test_loader)
    plot_confusion_matrix(test_conf_matrix, f'{model_type}_test_confusion_matrix.png', classifier_dir, class_names)

    test_metrics_file_path = os.path.join(classifier_dir, f'{model_type}_test_metrics_df.csv')
    test_metrics_df.to_csv(test_metrics_file_path)

    test_prediction_file_path = os.path.join(classifier_dir, f'{model_type}_test_prediction_df.csv')
    test_prediction_df.to_csv(test_prediction_file_path)

    LIME_explanation(best_model, test_prediction_df, model_type, classifier_dir)
        

    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Metrics Report: {test_metrics_df}')


def run_classifier(args, train_loader, val_loader, test_loader, classifier_dir):
    """
    Trains and evaluates a classifier model based on the specified architecture.

    Depending on the model type provided in `args.model_type` ('cnn', 'mlp_mixer', or 'vit'),
    this function sets appropriate learning rates and weight decay options for grid search.
    It then trains the model using `train_classifier`, evaluates performance on the validation set,
    and saves the validation and test results.

    Parameters:
        args (Namespace): Contains user-defined settings such as model_type and num_epochs.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        classifier_dir (str): Directory path to save model outputs and result files.

    Raises:
        Exception: If an unsupported model type is specified.

    Returns:
        None
    """
    
    if args.model_type == 'cnn':
        lrs = [0.001, 0.003, 0.01]
        weight_decay_options = [0.0, 0.0001, 0.001]
    elif args.model_type == 'mlp_mixer':
        lrs = [0.001, 0.003]
        weight_decay_options = [0.01, 0.05, 0.1]
    elif args.model_type == 'vit':
        lrs = [0.001, 0.003]
        weight_decay_options = [0.01, 0.05, 0.1]
    else:
        raise Exception(f'Unsupported model type: {args.model_type}')

    best_model, best_val_acc, val_metrics_df, val_conf_matrix, val_prediction_df, model_type = train_classifier(
        train_loader = train_loader,
        val_loader = val_loader,
        encoder_checkpoint = args.encoder_checkpoint_file,
        model_type = args.model_type,
        lrs = lrs,
        weight_decay_options = weight_decay_options,
        num_epochs = args.num_epochs,
        classifier_dir = classifier_dir
    )

    save_val_results(best_val_acc, val_metrics_df, val_conf_matrix, val_prediction_df, model_type, classifier_dir)
    save_test_results(best_model, test_loader, model_type, classifier_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train and evaluate a classifier model.')
    parser.add_argument('input_file', type = str,
                        help = 'Path to the input CSV file.')
    parser.add_argument('encoder_checkpoint_file', type = str,
                        help = 'Path to the encoder checkpoint file.')
    parser.add_argument('output_dir', type = str,
                        help = 'Directory to save the results.')
    parser.add_argument('model_type', type = str, choices = ['cnn', 'mlp_mixer', 'vit'],
                        help = 'Type of model to train.')
    parser.add_argument('--num_epochs', type = int, default = 10,
                        help = 'Number of training epochs.')
    parser.add_argument('--batch_size', type = int, default = 32,
                        help='Batch size for training and evaluation.')
    
    args = parser.parse_args()
    classifier_dir = args.output_dir
    os.makedirs(classifier_dir, exist_ok = True)
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # Split data into train, validation, and test sets
    train_df, val_df, test_df = train_test_val_split(args.input_file)

    # Create dataloaders
    train_loader = convert_images_to_tensors(train_df, batch_size = batch_size)
    val_loader = convert_images_to_tensors(val_df, batch_size = batch_size)
    test_loader = convert_images_to_tensors(test_df, batch_size = batch_size)

    run_classifier(args, train_loader, val_loader, test_loader, classifier_dir)