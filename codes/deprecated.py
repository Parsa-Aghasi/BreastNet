#all deprecated code

#region first rotate class
# Deprecated rotate class due to existence of torch.transforms.RandomRotation
# class Rotate:
#     """
#     Rotate by a given angle in radians.
#     Args: 
#         rotation: angle in degrees.
#     """
#     def __init__(self, rotation_angle_start: float, rotation_angle_stop: float):
#         assert isinstance(rotation_angle_start, (float, int))
#         assert isinstance(rotation_angle_stop, (float, int))

#         self.rotation_angle_start = rotation_angle_start
#         self.rotation_angle_stop = rotation_angle_stop

#     def __call__(self, image: torch.Tensor, centre_of_rotation: tuple = None) -> torch.Tensor:
#         """
#         Args:
#             image: image to be transformed
#             centre_of_rotation: position of pixel relative to which rotation occurs. default is centre pixel
#         """
#         if centre_of_rotation:
#             x, y = centre_of_rotation
#             height, width = image.squeeze().shape
#             shift1 = Shift(int(x)-height//2,width//2-int(y))
#             rotate_norm = Rotate(self.rotation_angle_start, self.rotation_angle_stop)
#             shift2 = Shift(-int(x)+height//2,-width//2+int(y))
#             composed = Compose([shift1, rotate_norm, shift2])
#             return composed(image)

#         else:
#             a = image.squeeze()
#             angle = ((self.rotation_angle_start - self.rotation_angle_stop) * 
#                     torch.rand((1,))) + self.rotation_angle_stop
#             return torch.from_numpy(transform.rotate(a, angle)[None, :, :])
# endregion

#region second rotate class
# class Rotate:
#     """
#     Rotate by a given angle in radians.
#     Args: 
#         rotation: angle in degrees.
#     """
#     def __init__(self, rotation_angle_start: float, rotation_angle_stop: float
#                  ,centre_of_rotation: tuple = None):
#         assert isinstance(rotation_angle_start, (float, int))
#         assert isinstance(rotation_angle_stop, (float, int))
#         assert isinstance(centre_of_rotation, tuple)
        

#         self.rotation_angle_start = rotation_angle_start
#         self.rotation_angle_stop = rotation_angle_stop
#         self.centre_of_rotation = centre_of_rotation

#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             image: image to be transformed
#             centre_of_rotation: position of pixel relative to which rotation occurs. default is centre pixel
#         """
#         tform = RandomRotation((self.rotation_angle_start, self.rotation_angle_stop), center=self.centre_of_rotation)
#         return tform(image)
#endregion

#region Rescale

# # for in real use.

# class Rescale:
#     """
#     Rescale to a given size, to adjust CNN input size.
#     Args: 
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         h, w = image.squeeze().shape
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
        
#         new_h, new_w = int(new_h), int(new_w)

#         return torch.from_numpy(transform.resize(image.squeeze().numpy(), (new_h, new_w)))[None , : , :]

#endregion

#region Shift classes

# class Shift:
#     """
#     shifts image x pixels to left and y pixels upward
#     """
#     def __init__(self, x: int, y: int):
#         assert isinstance(x, int)
#         assert isinstance(y, int)
#         self.x = x
#         self.y = y

#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         tform = AffineTransform(translation=(self.x,self.y))
#         return warp(image.squeeze().numpy(), tform, mode='wrap', preserve_range=True)[None, :, :]

# class RandomShift:
#     """
#     shifts image x pixels to left and y pixels upward
#     """
#     def __init__(self, range_x: tuple, range_y, bernouli_trial_probability: float = 1):
#         assert isinstance(range_x[0], int)
#         assert isinstance(range_y[0], int)
#         assert isinstance(range_x[1], int)
#         assert isinstance(range_y[1], int)
        
#         self.x = torch.randint(range_x[0], range_x[1])
#         self.y = torch.randint(range_y[0], range_y[1])
#         self.probability = bernouli_trial_probability
#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         if torch.rand((1,)) <= self.probability:
#             tform = Shift(self.x, self.y)
#             return tform(image)
#         else: return image

#endregion

#region Horizontal Flip

# class Horizontal_Flip:
#     """
#     Flips the image horizontally by a certain probability.
#     Args:  
#         bernouli_trial_probability: probability of flip
#     """
#     def __init__(self, bernouli_trial_probability: float = 1):
#         assert isinstance(bernouli_trial_probability, (int, float))
#         self.probability = bernouli_trial_probability
#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#             if torch.rand((1,)) <= self.probability:
#                 return torch.from_numpy(image.numpy()[..., :, ::-1].copy())
#             else: return image

#endregion

#region Resize if not class

# class ResizeIfNot:
#     def __init__(self, size: tuple):
#         self.size = size
#         self.resize = Resize(size=size)
#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         if image.shape[-2:] == self.size: return image
#         else: return self.resize(image)

#endregion