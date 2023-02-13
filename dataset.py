from PIL import Image
from PIL import ImageFile
import numpy as np
import albumentations as A
import torch
import config
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    """
    to return image tensors and target tensors for the data loader in pytorch
    Args:
    image paths(list): path to the image file
    targets (list): respective target of images
    to_augment (bool): to implement image augmentation
    test_data (bool): if input data is test data, then return only image tensors
    """
    def __init__(self, image_paths, targets=None, to_augment=True, test_data=None):
        self.image_paths = image_paths
        self.targets = targets
        self.to_augment = to_augment
        self.test_data = test_data
        self.normalize = A.Compose([A.Normalize(always_apply=True)])
        # image augmentations
        self.augment = A.Compose([
            # to zoom the image randomly
            A.Affine(scale=(1.5, 2), keep_ratio=True),
            A.SafeRotate(limit=60, p=0.6),
            A.RandomBrightnessContrast(p=0.5)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        image = np.array(image)
        image = self.normalize(image=image)['image']
        if self.to_augment:
            image = self.augment(image=image)['image']
        # in pytorch the input image aray should be in format NCHW
        # batch N, channels C, height H, width W.
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        result = {
            "images": torch.tensor(image, dtype=torch.float),
        }
        if self.test_data:
            return result
        targets = self.targets[item]
        result.update({"targets": torch.tensor(targets, dtype=torch.long)})
        return result
