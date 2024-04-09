import torch
import torchaudio
import torchvision
import torch.nn as nn
from torchvision import transforms
from Pipeline import *
from torch.utils.data import Dataset, DataLoader,random_split
from torchaudio.datasets import SPEECHCOMMANDS

import math
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# Image Downloader :: 10 classes; torch.Size([3, 32, 32]) :: Load CIFAR-10 dataset
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root='./data',  # specify the root directory where the dataset will be downloaded
    train=True,      # set to True for the training set, False for the test set
    download=True,    # set to True to download the dataset if not already downloaded
    transform = transforms.ToTensor()
)

# Audio Downloader :: 35 classes; torch.Size([1, 16000]) :: Load SPEECH-COMMANDS dataset
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',    # specify the root directory where the dataset will be downloaded
    url= 'speech_commands_v0.02',
    download=True     # set to True to download the dataset if not already downloaded
)

class ImageDataset(Dataset):
    def __init__(self, split: str = "train", train_percentage=0.8) -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split

        # Load CIFAR-10 dataset
        self.dataset = image_dataset_downloader

        num_samples = len(self.dataset)

        val_percentage = 1 - (train_percentage)
        train_size = int(train_percentage * num_samples)
        val_size = num_samples - train_size # int(val_percentage * num_samples)

        torch.manual_seed(0)
        train_dataset, val_dataset = random_split(self.dataset,[train_size,val_size])
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

        if self.datasplit == "train":
            self.dataset = train_dataset
        elif self.datasplit == "val":
            self.dataset = val_dataset
        elif self.datasplit == "test":
            self.dataset = test_dataset
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx] 
        return image, label

class AudioDataset(Dataset):
    def __init__(self, split: str = "train", train_percentage=0.8, val_percentage = 0.1, uniform_length=16000) -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        self.dataset = audio_dataset_downloader
        self.uniform_length = uniform_length
        
        num_samples = len(self.dataset)
        test_percentage = 1 - (train_percentage + val_percentage)

        torch.manual_seed(0) 
        train_size = int(num_samples * train_percentage)
        tmp_size = num_samples - train_size
        test_size = int(tmp_size * (test_percentage / (1 - train_percentage)))
        val_size = tmp_size - test_size
        train_dataset, tmp_dataset = random_split(self.dataset, [train_size, tmp_size])
        test_dataset, val_dataset = random_split(tmp_dataset, [test_size, val_size])
        
        if split == "train":
            self.dataset = train_dataset
        elif split == "val":
            self.dataset = val_dataset
        elif split == "test":
            self.dataset = test_dataset
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

        self.labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, _, label, *_ = self.dataset[idx]
        label = self.label_encoder.transform([label])[0]

        if waveform.shape[1] < self.uniform_length:
            padding = torch.zeros(1, self.uniform_length - waveform.shape[1])
            waveform = torch.cat((waveform, padding), dim=1)

        return waveform, label


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False, is_audio = False):
        super(ResNetBlock, self).__init__()
        if is_audio:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
        
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias) if not conv2d else nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # self.bn1 = nn.BatchNorm1d(out_channels) if not conv2d else nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias) if not conv2d else nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # self.bn2 = nn.BatchNorm1d(out_channels) if not conv2d else nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
        
class Resnet_Q1(nn.Module):
    def __init__(self, out_channels = 1):
        super(Resnet_Q1, self).__init__()

        self.relu = nn.ReLU()

        number_of_blocks = 18

        self.conv2d = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(out_channels)
        self.residual_block2d = nn.Sequential(*[ResNetBlock(out_channels, out_channels, is_audio = False) for _ in range(number_of_blocks)])
        self.fc2 = nn.Linear(out_channels*32*32,10)


        self.conv1d = nn.Conv1d(1, out_channels, kernel_size=3,padding=1)
        self.bn1d = nn.BatchNorm1d(out_channels)
        self.residual_block_Ad = nn.Sequential(*[ResNetBlock(out_channels, out_channels, is_audio = True) for _ in range(number_of_blocks)])
        audio_features = 16000 # 16000 
        self.fc1 = nn.Linear(out_channels*audio_features,35)

    def forward(self, x):
        is_audio = (len(x.shape) == 3 and x.shape[1] == 1)

        if is_audio:
            out = self.conv1d(x)
            out = self.bn1d(out)
            out = self.relu(out)
            out = self.residual_block_Ad(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
        else:
            out = self.conv2d(x)
            out = self.bn2d(out)
            out = self.relu(out)
            out = self.residual_block2d(out)
            out = out.view(out.size(0), -1)
            out = self.fc2(out)
        return out


class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        initial_channels = 8        # hyperparameter
        initial_kernel_size = 3     # given

        kernels = [int(math.ceil(initial_kernel_size * (1.25 ** i))) for i in range(5)]
        channels = [int(math.ceil(initial_channels * (0.65 ** i))) for i in range(5)]

        self.features2d = nn.Sequential(
            self._create_conv_block(3, channels[0], kernels[0], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._create_conv_block(channels[0], channels[1], kernels[1], 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._create_conv_block(channels[1], channels[2], kernels[2], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._create_conv_block(channels[2], channels[3], kernels[3], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._create_conv_block(channels[3], channels[4], kernels[4], 3),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.features1d = nn.Sequential(
            self._create_conv_block(1, channels[0], kernels[0], 2, True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self._create_conv_block(channels[0], channels[1], kernels[1], 2,True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self._create_conv_block(channels[1], channels[2], kernels[2], 3, True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self._create_conv_block(channels[2], channels[3], kernels[3], 3,True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            self._create_conv_block(channels[3], channels[4], kernels[4], 3,True),
            nn.MaxPool1d(kernel_size=2, stride=2),)

        neurons_fc2,neurons_fc3,classify_classes = 64, 64, 10
        self.classifier_image = nn.Sequential(
                    nn.Linear(18, neurons_fc2), 
                    nn.ReLU(True), 
                    # nn.Dropout(),

                    nn.Linear(neurons_fc2, neurons_fc3), 
                    nn.ReLU(True), 
                    # nn.Dropout(),

                    nn.Linear(neurons_fc3, classify_classes),
                )

        neurons_fc2, neurons_fc3, classify_classes = 256, 128, 35  
        self.classifier_audio = nn.Sequential(
                    nn.Linear(1004, neurons_fc2), 
                    nn.ReLU(True), 
                    # nn.Dropout(),

                    nn.Linear(neurons_fc2, neurons_fc3), 
                    nn.ReLU(True), 
                    # nn.Dropout(),

                    nn.Linear(neurons_fc3, classify_classes),
                )

    def _create_conv_block(self, in_channels, out_channels, kernel_size, num_convs, is_audio = False):
        if is_audio:
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)]
            
            for _ in range(1, num_convs):
                layers.extend([nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(inplace=True)])
                
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)]
            for _ in range(1, num_convs):
                layers.extend([nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)])
                
        return nn.Sequential(*layers)

    def forward(self, x):
        is_audio = (len(x.shape) == 3 and x.shape[1] == 1)

        if is_audio:
            # num_flat_features =  x.shape[1] * x.shape[2] # == 1004     # # after convolutions of kernel as per architecture on 16000 feaures to 1004
            # neurons_fc2, neurons_fc3, classify_classes = 256, 128, 35  
            x = self.features1d(x)
            x = torch.flatten(x, 1)
            x = self.classifier_audio(x)
        else: 
            # num_flat_features = 18 
            # num_flat_features = x.shape[1] * x.shape[2] * x.shape[3]
            # neurons_fc2,neurons_fc3,classify_classes = 64, 32, 10
            x = self.features2d(x)
            x = torch.flatten(x, 1)
            x = self.classifier_image(x)

        return x

        
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_audio = False,
                *args, **kwargs) -> None:
        super(InceptionBlock, self).__init__(*args, **kwargs)
        
        self.is_audio = is_audio

        # for image
        # 1x1 CNA branch
        self.branch1_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3X3 CNA 5X5 CNA branch
        self.branch2_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3X3 CNA 5X5 CNA branch
        self.branch3_2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3X3 M (maxpooling) branch
        self.branch4_2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # for audio
        # 1x1 convolution branch
        self.branch1_1d = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3X3 CNA 5X5 CNA branch
        self.branch2_1d = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3X3 CNA 5X5 CNA branch
        self.branch3_1d = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3X3 M (maxpooling) branch
        self.branch4_1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if self.is_audio:
            branch1 = self.branch1_1d(x)    # 1 CNA
            branch2 = self.branch2_1d(x)    # 3 CNA 5 CNA
            branch3 = self.branch3_1d(x)    # 3 CNA 5 CNA
            branch4 = self.branch4_1d(x)    # 3 M

        else:
            branch1 = self.branch1_2d(x)    # 1 CNA
            branch2 = self.branch2_2d(x)    # 3 CNA 5 CNA
            branch3 = self.branch3_2d(x)    # 3 CNA 5 CNA
            branch4 = self.branch4_2d(x)    # 3 M

        return torch.cat([branch1, branch2, branch3, branch4], 1)

class Inception_Q3(nn.Module):
    def __init__(self, in_channels=3):
        super(Inception_Q3, self).__init__()

        # for image (2d)
        self.inception_blocks_image = nn.Sequential(
            InceptionBlock(3 , 8),
            InceptionBlock(27, 12),
            InceptionBlock(63, 8),
            InceptionBlock(87, 87)
        )
        number_image_features_extracted = 356352
        number_image_classes = 10
        self.fc_image = nn.Linear(number_image_features_extracted, number_image_classes)
        
        # for audio (1d)
        self.inception_blocks_audio = nn.Sequential(
            InceptionBlock(1 , 2, is_audio = True),
            InceptionBlock(7 , 2, is_audio = True),
            InceptionBlock(13, 2, is_audio = True),
            InceptionBlock(19, 2, is_audio = True),
        )
        number_audio_features_extracted = 400000
        number_audio_classes = 35
        self.fc_audio = nn.Linear(number_audio_features_extracted, number_audio_classes)

    def forward(self, x):
        is_audio = (len(x.shape) == 3 and x.shape[1] == 1)

        if is_audio:
            x = self.inception_blocks_audio(x)
            x = x.view(x.size(0),-1)
            x = self.fc_audio(x)
        else:
            x = self.inception_blocks_image(x)
            x = x.view(x.size(0),-1)
            x = self.fc_image(x)

        return x


class CustomNetwork_Q4(nn.Module):
    def __init__(self):
        super(CustomNetwork_Q4, self).__init__()

        # for image (2d)
        image_channels = 3
        self.inception_blocks_image = nn.Sequential(
            ResNetBlock(image_channels,3),
            ResNetBlock(3,3),
            InceptionBlock(3,5),
            InceptionBlock(18,5),
            ResNetBlock(33,33),
            InceptionBlock(33,15),
            ResNetBlock(78,78),
            InceptionBlock(78,20),
            ResNetBlock(138,138),
            InceptionBlock(138,32)
        )
        self.fc_layer_image = nn.Linear(239616,10)

        # for audio (1d)
        audio_channel = 1
        self.inception_blocks_audio = nn.Sequential(
            ResNetBlock(audio_channel,1, is_audio = True),
            ResNetBlock(1,1, is_audio = True),
            InceptionBlock(1,1, is_audio = True),
            InceptionBlock(4,1, is_audio = True),
            ResNetBlock(7,7, is_audio = True),
            InceptionBlock(7,1, is_audio = True),
            ResNetBlock(10,10, is_audio = True),
            InceptionBlock(10,1, is_audio = True),
            ResNetBlock(13,13, is_audio = True),
            InceptionBlock(13,1, is_audio = True),
        )

        self.fc_layer_audio = nn.Linear(256000,35)

    def forward(self, x):
        is_audio = len(x.shape) == 3 and x.shape[1] == 1

        if is_audio:
            x = self.inception_blocks_audio(x)

            # Classification Network
            x = x.view(x.size(0),-1)
            x = self.fc_layer_audio(x)

        else:
            x = self.inception_blocks_image(x)

            # Classification Network
            x = x.view(x.size(0),-1)
            x = self.fc_layer_image(x)

        return x

def plot_graphs(epoch_losses, epoch_accuracies, label):
    import matplotlib.pyplot as plt
    # Plotting
    plt.figure(figsize=(10, 5))

    l = [i for i in range(1, EPOCH + 1)]

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(l, epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(l, epoch_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{label}_Evaluations.png")
    plt.show()
    
from tqdm import tqdm
def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None,
            earlybreak = False):
    
    device = torch.device("cuda:0" if gpu == "T" else "cpu")
    
    network = network.to(device)

    network_class = network.__class__.__name__
    dataset_class = dataloader.dataset.__class__.__name__
    model_path = f"Networks/{network_class}_{dataset_class}_model.pth"
    
    epoch_losses, epoch_accuracies = [], []  # List to store epoch losses and epoch accuracies
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss = loss.item() * inputs.size(0)
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1) # outputs.max(1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)   

            if earlybreak: break   

        # # checkpoint
        # torch.save(network.state_dict(), model_path)

        # Calculate epoch statistics
        epoch_accuracy = 100. * correct_predictions / total_samples

        # Store epoch loss and accuracy
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Print epoch information
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch + 1,
            epoch_loss,
            epoch_accuracy
        ))

    torch.save({
        'model_state_dict':network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, model_path)
    
    plot_graphs(epoch_losses, epoch_accuracies,label = f"Evaluations/{network_class}_{dataset_class}_Training")


def validator(gpu="F",
        dataloader=None,
        network=None,
        criterion=None,
        optimizer=None,):
    
    device = torch.device("cuda:0" if gpu == "T" else "cpu")

    criterion = nn.CrossEntropyLoss()
    network = network.to(device)

    network_class = network.__class__.__name__
    dataset_class = dataloader.dataset.__class__.__name__
    model_path = f"Networks/{network_class}_{dataset_class}_model.pth"
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = network(inputs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_accuracy = 100. * correct / total

        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch+1,
            epoch_loss,
            epoch_accuracy
        )) 
        break

    torch.save({
        'model_state_dict':network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, model_path)
        

    

def evaluator(gpu = "F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None,):

    # device = next(network.parameters()).device
    device = torch.device("cuda:0" if gpu == "T" else "cpu")

    criterion = nn.CrossEntropyLoss()
    network = network.to(device)

    network_class = network.__class__.__name__
    dataset_class = dataloader.dataset.__class__.__name__
    model_path = f"Networks/{network_class}_{dataset_class}_model.pth"
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = network(inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_accuracy = 100. * correct / total

            print("[Loss: {}, Accuracy: {}]".format(
                epoch_loss,
                epoch_accuracy
            ))
            break