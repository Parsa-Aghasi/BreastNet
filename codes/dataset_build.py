import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
import os
import pandas as pd
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class mias(Dataset):
    """
    Custom dataset class

    Args:
        annotations_file: address of the dataframe file that contains the annotations.
        img_dir: address of directory that contains mammogram images.
        transform: transform applied to images e.g. Rescale, Crop, etc.
        target_trasnsform: transform applied to labels.
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0]) + '.jpeg'
        image = decode_image(img_path)
        labels = self.img_labels.iloc[index, 1:]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels

a = mias('labels/dataset_annotations_2.csv', '')
print(len(a))
figure, ax = plt.subplots()

#ax.imshow(a[0][0].squeeze())
print(a[0][0].shape)
#plt.show()