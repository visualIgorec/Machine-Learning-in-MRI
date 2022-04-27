import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision.transforms.transforms import ToPILImage
from skimage import io
import os


class CustomDataTransform_test(Dataset):
    def __init__(self, df, features_transform=None, label_transform=None):
        self.df = df
        self.features_transform = features_transform
        self.label_transform = label_transform
        self.root_dir_x = 'data/us'
        self.root_dir_y = 'data/fs'

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self,index):
        img_path_x = os.path.join(self.root_dir_x, self.df.iloc[index, 0])
        img_path_y = os.path.join(self.root_dir_y, self.df.iloc[index, 0])
        image_x = io.imread(img_path_x)
        image_y = io.imread(img_path_y)
        
        if self.features_transform is not None:
            image_x = self.features_transform(image_x)

        if self.label_transform is not None:
            image_y = self.label_transform(image_y)

        return (image_x, image_y)
