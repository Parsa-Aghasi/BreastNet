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
