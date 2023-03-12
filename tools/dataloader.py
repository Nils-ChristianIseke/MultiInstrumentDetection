# Imports
import os
import sys

import cv2
import random
import numpy as np
from PIL import Image

import json
import torch
from torchvision.transforms import functional as func
from torchvision import transforms
from torch.utils.data import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('RGB')


def mask_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('L')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def mask_np_loader(path):
    # The np mask has 3 channels but we need only a single channel
    return np.load(path)[..., 0]

def json_loader(path):
    with open(path, 'rb') as file:
        return json.load(file)

class EndoDataset(Dataset):
    """ The endoscopic dataset requires the following folder structure:
    Surgery --> Video --> images which contain the images to be loaded
    This mono class, works with split files which are text files that contain the
    relative path and the image name. (It is a mono class because it returns a single image)
    The class reads the image file paths that are specified in the text file and loads the images.
    It applies the specified transforms to the image, else it just converts it into a tensor.
    :returns Pre-processed and augmented image as a tensor
    """

    def __init__(self, data_root_folder=None,
                 filenames=None,
                 height=448,
                 width=448,
                 image_aug=None,
                 aug_prob=0.5,
                 camera="left",
                 image_ext='.png'):
        super(EndoDataset).__init__()
        self.data_root_folder = data_root_folder
        self.filenames = filenames
        self.height = height
        self.width = width
        self.image_ext = image_ext
        self.camera = camera

        # Pick image loader based on image format
        self.image_loader = np.load if self.image_ext == '.npy' else pil_loader
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.cam_to_side = {"left": "l", "right": "r"}

        # Image pre-processing options
        self.image_aug = image_aug
        self.aug_prob = aug_prob

        # Pick resize function based on image format
        if self.image_ext == '.png':
            # Output: Resized PIL Image
            self.resize = transforms.Resize((self.height, self.width), interpolation=Image.LINEAR)
            # Resize to dims slightly larger than given dims
            # Sometimes useful for aug together with crop function
            self.resize_bigger = transforms.Resize((int(self.height * 1.2),
                                                    int(self.width * 1.2)))
        elif self.image_ext == '.npy':
            self.resize = lambda x: cv2.resize(x, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            self.resize_bigger = lambda x: cv2.resize(x, (self.width*1.2, self.height*1.2),
                                                      interpolation=cv2.INTER_NEAREST)

    def get_split_filename(self, filename):
        """ Splits a filename string comprising of "relative_path <space> image_name"
        :param filename A string comprising of "relative_path <space> image_name"
        :return split_filename- "relative_path, image_name"
        """
        split_filename = filename.split()
        # folder, frame_num, side
        if len(split_filename) == 2: return split_filename[0], split_filename[1], self.cam_to_side[self.camera]
        else: return split_filename

    def make_image_path(self, data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of image
        """
        frame_name = "{:06d}{}".format(int(image_name), self.image_ext)
        return os.path.join(data_root, rel_folder,
                            "image_0{}".format(self.side_map[side]), frame_name)

    def preprocess(self, image):
        image = self.resize(image)
        # if self.image_aug and random.random() > self.aug_prob: image = self.image_aug(image)
        if self.image_aug: image = self.image_aug(image=np.asarray(image))["image"]  # alb needs np input
        return func.to_tensor(image)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns the image with pre-proc transforms + aug applied to it"""
        args = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        image = self.preprocess(image)
        return image, self.filenames[index]


class EndoMaskDataset(EndoDataset):
    """ Loads an image and its corresponding mask
    Some aug is performed common to both image and mask
    Some aug like color aug is performed only on the image
    :returns Pre-proc+aug image, mask
    """

    def __init__(self, mask_transform=None,
                 image_mask_aug=None,
                 mask_path_suffix="",
                 **kwargs):
        super(EndoMaskDataset, self).__init__(**kwargs)
        self.mask_path_suffix = mask_path_suffix
        self.image_mask_aug = image_mask_aug
        self.mask_loader = mask_np_loader if self.image_ext == '.npy' else mask_loader
        self.mask_transform = transforms.ToTensor() if mask_transform is None else transforms.Compose(mask_transform)

    def make_mask_path(self, data_root, rel_folder, image_name, side=None):
        """Combines the relative path with the data root to get the complete path of mask
        """
        frame_name = "{:06d}{}".format(int(image_name), self.image_ext)
        return os.path.join(data_root, rel_folder,
                            "mask_0{}".format(self.side_map[side])+self.mask_path_suffix, frame_name)

    def preprocess_image_mask(self, image, mask):
        image = self.resize(image)
        if self.image_aug: image = self.image_aug(image=np.asarray(image))["image"]
        if self.image_mask_aug:
            augmented = self.image_mask_aug(image=np.asarray(image), mask=np.asarray(mask))
            # alb needs np input
            image = augmented["image"]
            mask = augmented["mask"]
        image = func.to_tensor(np.array(image))
        mask = func.to_tensor(np.array(mask))
        return image, mask

    def __getitem__(self, index):
        args = self.get_split_filename(self.filenames[index])
        image = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        mask = self.mask_loader(self.make_mask_path(self.data_root_folder, *args))
        image, mask = self.preprocess_image_mask(image=image, mask=mask)
        return image, mask, self.filenames[index]


class StereoEndoDataset(EndoDataset):
    def get_split_filename(self, filename):
        """ Splits a filename string comprising of "relative_path <space> image_name"
        :param filename A string comprising of "relative_path <space> image_name"
        :return split_filename- "relative_path, image_name"
        """
        split_filename = filename.split()
        if len(split_filename) == 2: return split_filename[0], split_filename[1]
        else: return split_filename

    def preprocess(self, image_left, image_right):
        image_left = self.resize(image_left)
        image_right = self.resize(image_right)
        # if self.image_aug and random.random() > self.aug_prob: image = self.image_aug(image)
        if self.image_aug: image_left = self.image_aug(image=np.asarray(image_left))["image"]  # alb needs np input
        if self.image_aug: image_right = self.image_aug(image=np.asarray(image_right))["image"]  # alb needs np input
        image = np.concatenate((np.array(image_left), np.array(image_right)), axis=2)
        return func.to_tensor(image)

    def __getitem__(self, index):
        """Returns the image with pre-proc transforms + aug applied to it"""
        args = self.get_split_filename(self.filenames[index])
        image_left = self.image_loader(self.make_image_path(self.data_root_folder, *["l" if idx == 3 else arg for idx, arg in enumerate(args)]))
        image_right = self.image_loader(self.make_image_path(self.data_root_folder, *["r" if idx == 3 else arg for idx, arg in enumerate(args)]))
        image = self.preprocess(image_left, image_right)
        return image, self.filenames[index]


class StereoEndoMaskDataset(EndoMaskDataset):
    def __init__(self, **kwargs):
        super(StereoEndoMaskDataset, self).__init__(**kwargs)
        self.other_cam = {"l": "r", "r": "l"}

    def preprocess_image_mask(self, image_A, image_B, mask_A, mask_B):
        image_A = self.resize(image_A)
        image_B = self.resize(image_B)
        if self.image_aug:
            image_A = self.image_aug(image=np.asarray(image_A))["image"]
        if self.image_aug:
            image_B = self.image_aug(image=np.asarray(image_B))["image"]
        if self.image_mask_aug:
            augmented_A = self.image_mask_aug(image=np.asarray(image_A), mask=np.asarray(mask_A))
            augmented_B = self.image_mask_aug(image=np.asarray(image_B), mask=np.asarray(mask_B))
            # alb needs np input
            image_A = augmented_A["image"]
            mask_A = augmented_A["mask"]
            image_B = augmented_B["image"]
        mask_A = mask_A[..., np.newaxis]
        image = func.to_tensor(np.concatenate((np.array(image_A), np.array(image_B)), axis=2))
        mask_A = func.to_tensor(np.array(mask_A))
        return image, mask_A

    def __getitem__(self, index):
        args = self.get_split_filename(self.filenames[index])
        image_A = self.image_loader(self.make_image_path(self.data_root_folder, *args))
        mask_A = self.mask_loader(self.make_mask_path(self.data_root_folder, *args))
        image_B = self.image_loader(self.make_image_path(self.data_root_folder,
                                                      *[self.other_cam[self.cam_to_side[self.camera]] if idx == 3 else
                                                        arg for idx, arg in enumerate(args)]))
        mask_B = self.mask_loader(self.make_mask_path(self.data_root_folder,
                                                      *[self.other_cam[self.cam_to_side[self.camera]] if idx == 3 else
                                                        arg for idx, arg in enumerate(args)]))
        image, mask = self.preprocess_image_mask(image_A, image_B, mask_A, mask_B)
        return image, mask, self.filenames[index]
