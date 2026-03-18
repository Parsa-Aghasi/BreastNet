from skimage import transform
import numpy as np
import torch
from dataset_build import mias
import matplotlib.pyplot as plt
from math import ceil
from typing import Dict

# for in real use.
class Rescale:
    """
    Rescale to a given size, to adjust CNN input size.
    Args: 
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.squeeze().shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        return torch.from_numpy(transform.resize(image.squeeze().numpy(), (new_h, new_w)))[None , : , :]
    
#for data augmentation
class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        h, w = image.squeeze().shape
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h + 1, (1,))
        left = torch.randint(0, w - new_w + 1, (1,))

        image = image[:, top: top + new_h,
                      left: left + new_w]

        return image

#TODO add random rotation in a range
class Rotate:
    """
    Rotate by a given angle in radians.
    Args: 
        rotation: angle in degrees.
    """
    def __init__(self, rotation_angle_start: float, rotation_angle_stop: float):
        assert isinstance(rotation_angle_start, (float, int))
        assert isinstance(rotation_angle_stop, (float, int))

        self.rotation_angle_start = rotation_angle_start
        self.rotation_angle_stop = rotation_angle_stop

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        a = image.squeeze()
        angle = ((self.rotation_angle_start - self.rotation_angle_stop) * 
                torch.rand((1,))) + self.rotation_angle_stop
        return torch.from_numpy(transform.rotate(a, angle)[None, :, :])

class Horizontal_Flip:
    """
    Flips the image horizontally by a certain probability.
    Args:  
        bernouli_trial_probability: probability of flip
    """
    def __init__(self, bernouli_trial_probability: float):
        self.probability = bernouli_trial_probability
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
            if torch.rand((1,)) <= self.probability:
                a = image.squeeze().numpy()
                return torch.from_numpy(a[None, :, ::-1].copy())
            else: return image
    


def show(rows, columns, **images: torch.Tensor) -> torch.Tensor:
    fig, axes = plt.subplots(rows, columns)
    fig.tight_layout()
    titles = list(images)
    if isinstance(axes,plt.Axes):
        axes.imshow(images[titles[0]].squeeze())
        axes.set_title(titles[0])
        plt.show()
        return
    axes = axes.reshape(rows, columns)

    for i in range(rows):
        for j in range(columns):
            if i*columns + j >= len(titles): 
                plt.show()
                return
            axes[i, j].imshow(images[titles[i*columns + j]].squeeze())
            axes[i, j].set_title(titles[i*columns + j])
    
    plt.show()



def main():
    dataset = mias('dataset_all_mias/labels/dataset_annotations_2.csv', 'dataset_all_mias/dataset_jpeg')
    resize = Rescale((600, 600))
    crop = RandomCrop((400, 700))
    rotate_5 = Rotate(0,180)
    flip = Horizontal_Flip(0.5)
    
    print(dataset[0][0].shape)
    print(resize(dataset[0][0]).shape)
    print(crop(dataset[0][0]).shape)
    print(rotate_5(dataset[0][0]).shape)
    print(flip(dataset[0][0]).shape)
    
    show(3,3,Original=dataset[0][0], 
         Resized=resize(dataset[0][0]), 
         Cropped=crop(dataset[0][0]),
         Rotated_1=rotate_5(dataset[0][0]),
         Rotated_2=rotate_5(dataset[0][0]),
         Rotated_3=rotate_5(dataset[0][0]),
         Flipped_1=flip(dataset[0][0]),
         Flipped_2=flip(dataset[0][0]),
         Flipped_3=flip(dataset[0][0]),)

if __name__ == '__main__':
    main()