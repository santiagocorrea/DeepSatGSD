from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import albumentations as albu
import config
import sys
import os
import pdb

# Create directories and copy patches to processed data for training
raw_data_path = config.RAW_IMAGERY_PATH[:-4]
processed_data_path = config.PROCESSED_DATA_PATH

# x_train_dir = os.path.join(processed_data_path, 'imgs')
# y_train_dir = os.path.join(processed_data_path, 'masks')

# x_val_dir = os.path.join(processed_data_path, 'val_img')
# y_val_dir = os.path.join(processed_data_path, 'val_mask')

# x_test_dir = os.path.join(processed_data_path, 'test_img')
# y_test_dir = os.path.join(processed_data_path, 'test_mask')


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.im_patches = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.mask_patches = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = imread(self.im_patches[i])[:,:,:3].astype('float32')/255
        mask = imread(self.mask_patches[i], as_gray=True)[:,:,np.newaxis]
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_training_augmentation():
    
    # training_transform = albu.Compose([
    #     albu.HorizontalFlip(p=0.5),
    #     albu.RandomBrightnessContrast(p=0.2),
    #     albu.ColorJitter(), 
    #     albu.MedianBlur(blur_limit=3)
    # ], p=0.9)

    # training_transform = [
    #     albu.HorizontalFlip(p=0.5),
    #     albu.RandomBrightnessContrast(p=0.5),
    #     albu.ColorJitter(), 
    #     albu.MedianBlur(blur_limit=3)
    # ]
    training_transform = [
        #albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
        albu.ColorJitter(), 
        albu.MedianBlur(blur_limit=3),
        albu.Blur()
    ]

    #return training_transform
    return albu.Compose(training_transform)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    train_dataset = Dataset(
                x_train_dir, 
                y_train_dir, 
                augmentation=get_training_augmentation(), 
                #preprocessing=get_preprocessing(preprocessing_fn),
                #classes=CLASSES,
                #transform =get_training_augmentation()
            )
    pdb.set_trace()
