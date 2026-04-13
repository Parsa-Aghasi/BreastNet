from torchvision.transforms import RandomRotation, RandomAffine
import torch
from dataset_build import mias
import matplotlib.pyplot as plt
    
#for data augmentation
class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, bernouli_trial_probability: float = 1):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(bernouli_trial_probability, (int, float))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.probability = bernouli_trial_probability

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand((1,)) <= self.probability:
            h, w = image.shape[-2:]
            new_h, new_w = self.output_size

            top = torch.randint(0, h - new_h + 1, (1,))
            left = torch.randint(0, w - new_w + 1, (1,))

            image = image[..., top: top + new_h,
                        left: left + new_w]

        return image

class Rotate:
    """
    Rotate by a given angle in degrees.
    Args: 
        rotation: angle in degrees.
    """
    def __init__(self, rotation_angle_start: float, rotation_angle_stop: float
                 ,bernouli_trial_chance: float = 1, centre_of_rotation: tuple = None):
        assert isinstance(rotation_angle_start, (float, int))
        assert isinstance(rotation_angle_stop, (float, int))
        assert isinstance(centre_of_rotation, (type(None), tuple))
        assert isinstance(bernouli_trial_chance, (float, int))
        

        self.rotation_angle_start = rotation_angle_start
        self.rotation_angle_stop = rotation_angle_stop
        self.centre_of_rotation = centre_of_rotation
        self.probability = bernouli_trial_chance

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand((1,)) <= self.probability:
            tform = RandomRotation((self.rotation_angle_start, self.rotation_angle_stop), center=self.centre_of_rotation)
            return tform(image)
        else:
            return image

class RandomShift:
    def __init__(self, x, y, bernouli_trial_probability: float = 1):
        self.x = x
        self.y = y
        self.probability = bernouli_trial_probability
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand((1,)) <= self.probability:
            x = self.x/image.shape[-2]
            y = self.y/image.shape[-1]
            tform = RandomAffine(degrees=0, translate=(x, y))
            return tform(image)
        else: return image
       

def show(rows, columns, **images: torch.Tensor):
    fig, axes = plt.subplots(rows, columns)
    fig.tight_layout()
    titles = list(images)
    if isinstance(axes,plt.Axes):
        axes.imshow(images[titles[0]].permute(1,2,0))
        axes.set_title(titles[0])
        plt.show()
        return
    axes = axes.reshape(rows, columns)

    for i in range(rows):
        for j in range(columns):
            if i*columns + j >= len(titles): 
                plt.show()
                return
            axes[i, j].imshow(images[titles[i*columns + j]].permute(1,2,0))
            axes[i, j].set_title(titles[i*columns + j])
    
    plt.show()



def main():
    from torchvision.transforms import RandomRotation, Resize , RandomHorizontalFlip
    dataset = mias('dataset_all_mias/labels/label_encoded_dataset.csv',
                   'dataset_all_mias/dataset_jpeg')
    resize = Resize((600, 600))
    crop = RandomCrop((400, 700))
    rotate_5 = Rotate(0, 40)
    flip = RandomHorizontalFlip(0.5)
    shift = RandomShift(1000, 1000)
    
    print(dataset[0][0].shape)
    print(resize(dataset[0][0]).shape)
    print(crop(dataset[0][0]).shape)
    print(rotate_5(dataset[0][0]).shape)
    print(flip(dataset[0][0]).shape)
    
    show(4,3,Original=dataset[0][0], 
         Resized=resize(dataset[0][0]), 
         Cropped=crop(dataset[0][0]),
         Rotated_1=rotate_5(dataset[0][0]),
         Rotated_2=rotate_5(dataset[0][0]),
         Rotated_3=rotate_5(dataset[0][0]),
         Flipped_1=flip(dataset[0][0]),
         Flipped_2=flip(dataset[0][0]),
         Flipped_3=flip(dataset[0][0]),
         shifted = shift(dataset[0][0]))

if __name__ == '__main__':
    main()