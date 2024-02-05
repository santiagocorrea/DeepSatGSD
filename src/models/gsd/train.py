"""
Trainer for the DeepSatGSD model
"""

__author__ = "Santiago Correa"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import rasterio
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

tqdm.pandas()

class SensorDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_black_pixels=0.4, max_white_pixels=0.4, prune=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([filename for filename in os.listdir(root_dir) if filename.startswith("GSD")], key=lambda x: int(x.split('_')[1][:-2]))
        self.filepaths = []
        self.labels = []
        self.prune = prune

        for i, sensor_class in enumerate(self.classes):
            class_dir = os.path.join(root_dir, sensor_class)
            file_list = [f for f in os.listdir(class_dir) if f.startswith("row")]
            sampled_files = random.sample(file_list, k=int(len(file_list) * 0.1))
            for filename in sampled_files:
                filepath = os.path.join(class_dir, filename)
                self.filepaths.append(filepath)
                self.labels.append(i)
                
        # Prune the dataset based on black and white pixel percentages
        if self.prune:
            self.prune_dataset(max_black_pixels, max_white_pixels)
        # Calculate mean and standard deviation of the dataset
        # self.mean, self.std = self.calculate_statistics()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        label = self.labels[index]

        with rasterio.open(filepath, 'r') as img:
            image = img.read().transpose(1, 2, 0)

        if self.transform:
            image = self.transform(image)
            #print(f" in transform: {image.shape}")

        return image, label
    
    def prune_dataset(self, max_black_pixels, max_white_pixels):
        pruned_filepaths = []
        pruned_labels = []
        channels_sum = np.zeros((3,))
        channels_squared_sum = np.zeros((3,))

        for filepath, label in tqdm(zip(self.filepaths, self.labels)):
            with rasterio.open(filepath, 'r') as img:
                image = img.read().transpose(1, 2, 0)  # Transpose to (H, W, C) (512,512,3)

            # Calculate the percentage of black and white pixels
            total_pixels = np.prod(image.shape)
            black_pixels = np.sum(image == 0)
            white_pixels = np.sum(image == 255)
            black_pixel_percentage = black_pixels / total_pixels
            white_pixel_percentage = white_pixels / total_pixels

            # Prune the image if it falls outside the specified range
            if black_pixel_percentage <= max_black_pixels and white_pixel_percentage <= max_white_pixels:
                pruned_filepaths.append(filepath)
                pruned_labels.append(label)
                
                channels_sum += np.sum(image, axis=(0, 1))
                channels_squared_sum += np.sum(np.square(image), axis=(0, 1))
                

        self.filepaths = pruned_filepaths
        self.labels = pruned_labels
        num_samples = len(self.filepaths)
        self.mean = channels_sum / (num_samples * image.shape[0] * image.shape[1])
        variance = (channels_squared_sum / (num_samples * image.shape[0] * image.shape[1])) - np.square(self.mean)
        variance = np.maximum(variance, 1e-8)  # Clip variance to a minimum value of epsilon
        self.std = np.sqrt(variance)
        
        self.mean = self.mean.tolist()
        self.std = self.std.tolist()
        


def main(args):
    """ Main entry point of the app """
    print("Calculating mean and std for normalization...")
    
    # Set the path to your train, validation, and test directories
    train_dir = "/work/scorreacardo_umass_edu/DeepSatGSD/data/processed/train"
    validation_dir = "/work/scorreacardo_umass_edu/DeepSatGSD/data/processed/validation"
    test_dir = "/work/scorreacardo_umass_edu/DeepSatGSD/data/processed/test"
    
    # Define the image transformation
    image_transform = ToTensor()  # Add other transformations as needed

    # # Create the datasets
    train_dataset_normalization = SensorDataset(train_dir, transform=image_transform, prune=args.prune)

    # # Create the data loaders for batching and shuffling
    train_loader_normalization = torch.utils.data.DataLoader(train_dataset_normalization, batch_size=1, shuffle=False)
    
    mean = 0.0
    meansq = 0.0
    count = 0
    import pdb
    for index, data in tqdm(enumerate(train_loader_normalization)):
        #pdb.set_trace()
        batch_size = data[0].shape[0]
        data_batch = data[0]  # Concatenate tensors from the list
        mean += data_batch.sum(dim=(0, 2, 3))
        meansq += (data_batch **2).sum(dim=(0, 2, 3))
        count += batch_size * data_batch .shape[2] * data_batch.shape[3]

    total_mean = mean / count
    total_var = (meansq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    print("mean: " + str(total_mean.tolist()))
    print("std: " + str(total_std.tolist()))
    print(args)
    
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the training parameters
    batch_size = int(args.batchsize) #bajar
    num_epochs = int(args.epochs)
    learning_rate = float(args.lr)

    # Define the image transformations
    transform = torchvision.transforms.Compose([
        ToTensor(),
        Normalize(total_mean.tolist(), 
                  total_std.tolist())  # Normalize image tensors
    ])

    # Create custom datasets
    print('loading datasets...')
    train_dataset = SensorDataset(train_dir, transform=transform)
    validation_dataset = SensorDataset(validation_dir, transform=transform)
    test_dataset = SensorDataset(test_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the pre-trained ResNet model
    model_dic = {
        'resnet18': torchvision.models.resnet18(pretrained=args.pretrained),
        'resnet34': torchvision.models.resnet34(pretrained=args.pretrained),
        'resnet50': torchvision.models.resnet50(pretrained=args.pretrained),
    }
    model = model_dic[args.backbone]
    num_classes = len(train_dataset.classes)
    
    if args.reg:
        # Modify the last fully connected layer to match the number of classes
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),  # Add dropout layer with a dropout rate defined in the input
            nn.Linear(256, num_classes)
        )
    else:
        # Modify the last fully connected layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the device
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.reg:
        # Add weight decay (L2 regularization) to the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    learning_courve = []
    # Training loop
    print('start training loop...')
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            if args.reg:
                # Add regularization term to the loss
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += 0.001 * l2_reg  # Adjust the regularization strength (0.001) as needed

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print the loss every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{len(train_loader)}], Loss: {loss.item()}")
                learning_courve.append(loss.item())

        # Print the training loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        learning_courve.append(loss.item())
    
    plt.plot(learning_courve, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Evaluation on validation set
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    validation_accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {validation_accuracy}")

    # Compute additional performance metrics
    classification_metrics = classification_report(true_labels, predicted_labels, target_names=validation_dataset.classes)
    print("Classification Report:")
    print(classification_metrics)

    # Evaluation on test set
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    test_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {test_accuracy}")

    # Compute additional performance metrics
    classification_metrics = classification_report(true_labels, predicted_labels, target_names=test_dataset.classes)
    print("Classification Report:")
    print(classification_metrics)
    
    # Compute the confusion matrix
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix using seaborn
    class_names = train_dataset.classes  # Assuming you have access to the class names
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
   
    
    


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional argument
    parser.add_argument("exp", help="Required positional argument: experiment number")
    
    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-b", "--backbone", action="store", dest="backbone")
    
    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-bs", "--batch-size", action="store", dest="batchsize")
    
    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-l", "--lr", action="store", dest="lr")
    
    # Optional argument flag which defaults to False
    parser.add_argument("-r", "--reg", action="store_true", default=False)
    
    # Optional argument flag which defaults to False
    parser.add_argument("-d", "--dropout", action="store", dest="dropout")
    
    # Optional argument flag which defaults to False
    parser.add_argument("-p", "--prune", action="store_true", default=False)
    
     # Optional argument flag which defaults to False
    parser.add_argument("-pt", "--pretrained", action="store_true", default=False)
    
    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-e", "--epochs", action="store", dest="epochs")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)