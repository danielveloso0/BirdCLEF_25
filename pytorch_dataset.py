import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from Audio_Pipeline import AudioProcessing
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

# EfficientNet Takes 224x224 images as input, so we resize all of them
data_transforms = {
    'train': T.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        T.RandomResizedCrop((224,224)),
        T.RandomHorizontalFlip(),
        #T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
     'val': T.Compose([
        T.Resize((224,224)),
        #T.CenterCrop(224),
        #T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'test': T.Compose([
        T.Resize((224,224)),
        #T.CenterCrop(224),
       #T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
}

# Dataset para Pytorch

class BirdDataset(Dataset):
    def __init__(self, df,ffts,f0,f, transformer=None):
        super().__init__()
        self.path = df.filename
        self.labels = df.primary_label
        self.processing = AudioProcessing(n_ffts=ffts,
                                          fmin=f0, fmax=f)
        self.transform = transformer
        self.num_class = 206
       # self.fold = fold
    def __len__(self):
        return len(self.path)
    def __getitem__(self,idx):
        origin_path = os.path.join('/kaggle/input/birdclef-2025/train_audio',self.path[idx])
        if os.path.exists(origin_path):
            ys, fs = self.processing.select_window(origin_path)
            image = self.processing.melspec_to_image(self.processing.mel_spectrogram(ys,fs))
            label = nn.functional.one_hot(torch.tensor(self.labels[idx],dtype=torch.int64),num_classes=self.num_class)
            if self.transform:
                image = self.transform(image)
                return image,label
            return image, label
        else:
            print(f"Audio não encontrado: {origin_path}")
        
    