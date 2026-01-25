"""
Modified data loaders which also return the image path to aid with caching.

Code based on: 
https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
https://discuss.pytorch.org/t/using-imagefolder-without-subfolders-labels/67439/2

Code by: Matthew Walmer
"""
import os
import torch
from torchvision import datasets
from PIL import Image


# from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# based on: https://discuss.pytorch.org/t/using-imagefolder-without-subfolders-labels/67439/2
# Data loader for a single image folder with images. No labels will be loaded. Will also return
# the image path by default
class SingleFolderWithPaths(torch.utils.data.Dataset):
    def __init__(self, image_root, transform=None):
        self.image_root = image_root
        self.image_paths = sorted(os.listdir(self.image_root))
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_path = os.path.join(self.image_root, image_path)
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        # make a new tuple that includes original and the path
        tuple_with_path = ((x,) + (image_path,))
        return tuple_with_path
    
    def __len__(self):
        return len(self.image_paths)