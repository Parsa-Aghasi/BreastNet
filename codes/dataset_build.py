import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import decode_image
import os
import pandas as pd
import matplotlib.pyplot as plt


class MySubset(Subset):
    """
    Customised version of Subset to access show_image method for train and test split returned values
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        
    def show_image(self, index):
        original_idx = self.indices[index]
        return self.dataset.show_image(original_idx)

class mias(Dataset):
    """
    Custom dataset class

    Args:
        annotations_file: address of the dataframe file that contains the annotations.
        img_dir: address of directory that contains mammogram images.
        transform: transform applied to images e.g. Rescale, Crop, etc.
        target_trasnsform: transform applied to labels.
    
    Labels of image: ref_num,back_tissue_chr,class,severity,x,y,r
    'back_tissue_chr': {'G': 0, 'D': 1, 'F': 2}
    'class': {'CIRC': 0, 'NORM': 1, 'MISC': 2, 'ASYM': 3, 'ARCH': 4, 'SPIC': 5, 'CALC': 6}
    'severity': {'B': 0, 'M': 1}
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        
        elif isinstance(index, list):
            return [self[i] for i in index]
        
        elif isinstance(index, int):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0]) + '.jpeg'
            image = decode_image(img_path).type(torch.float)
            labels = self.img_labels.iloc[index, 1:]
            labels = torch.tensor(labels.to_numpy(dtype=float))
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                labels = self.target_transform(labels)
            return image, labels
        else:
            raise TypeError(f"{type(self).__name__} indices must be integers or slices or list, not {type(index).__name__}, got {index}")
    
    def show_image(self, index):
        fig, ax = plt.subplots()
        ax.imshow(self[index][0].squeeze())
        plt.show()
        plt.close(fig)

def build_train_test_datasets(
    annotations_file,
    img_dir,
    train_transform=None,
    test_transform=None,
    test_size=0.2,
    seed=42) -> tuple[MySubset, MySubset]:
    """
    Create train and test datasets with different transforms.
    
    returns two dataset objects:
        one with train transforms
        one with test transforms
    """
    base_dataset = mias(annotations_file, img_dir)
    dataset_size = len(base_dataset)

    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed)).tolist()
    split = int(dataset_size * (1 - test_size))

    train_indices = indices[:split]
    test_indices = indices[split:]

    train_dataset = mias(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=train_transform,
    )
    test_dataset = mias(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=test_transform,
    )

    train_subset = MySubset(train_dataset, train_indices)
    test_subset = MySubset(test_dataset, test_indices)

    return train_subset, test_subset


def main():
    from torchvision.transforms import Compose, RandomHorizontalFlip, Normalize
    from transform import Rotate
    
    a = mias('dataset_all_mias/labels/label_encoded_dataset.csv', 
             'dataset_all_mias/dataset_jpeg')
    print(len(a))
    a.show_image(0)

    tforms = Compose([
    Rotate(-10, 10, 0.7),
    RandomHorizontalFlip(0.5),
    Normalize((0,), (1,))
    ])

    train_dataset, test_dataset = build_train_test_datasets(
    annotations_file='dataset_all_mias/labels/label_encoded_dataset.csv',
    img_dir='dataset_all_mias/dataset_jpeg',
    train_transform=tforms,
    test_size=0.2,
    seed=42,)

    print(f'Train size: {len(train_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    train_image, train_label = train_dataset[0]
    test_image, test_label = test_dataset[0]

    trains = train_dataset[:3]
    images = a[:3]
    print(len(trains))
    print(len(images))

    print(f'Train image shape: {train_image.shape}')
    print(f'Train image label: {train_label}')
    print(f'Test image shape: {test_image.shape}')
    print(f'Test image label: {test_label}')

if __name__ == '__main__':
    main()