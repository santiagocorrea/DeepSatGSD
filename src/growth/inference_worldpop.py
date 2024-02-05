import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
from skimage.io import imread
import pdb
from skimage.filters import laplace
from skimage.color import rgb2gray
from collections import defaultdict
from shapely.geometry import Polygon
from skimage.draw import line, polygon, ellipse
import rasterio
from rasterio import features
import segmentation_models_pytorch as smp
import albumentations as albu
import torch
from tqdm import tqdm
import argparse
import pdb
tqdm.pandas()

def infer(exp_name, list_chips):
    list_preds = []
    model = torch.load(f'/home/scorreacardo/projects/urban_growth/urbano/models/{exp_name}_best_model.pth')
    # model = torch.load(f'/work/scorreacardo_umass_edu/DeepSatGSD/models/segmentation_{exp_name}_legacy_best_model.pth')
    model = model.float()
    model = model.to('cuda')  # Move the model to GPU
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', "imagenet")
    _transform = [albu.Lambda(image=preprocessing_fn)]
    aug = albu.Compose(_transform)

    for file in list_chips:
        
        chip = file[:,:,:3]
        aug_chip = aug(image=chip)
        x_tensor_chip = torch.from_numpy(aug_chip['image']).to('cuda').unsqueeze(0)
        pr_mask_chip = model.predict(x_tensor_chip.permute(0,3,1,2).float())
        pr_mask_chip = (pr_mask_chip.squeeze().cpu().numpy().round())
        pr_mask_chip_th = sigmoid(pr_mask_chip)
        pr_mask_chip_th = pr_mask_chip_th > 0.5
        pr_mask_chip_th = np.uint8(np.multiply(pr_mask_chip_th, 255))
        list_preds.append(pr_mask_chip_th)
        
    return list_preds

def sigmoid(X):
    return 1/(1+np.exp(-X))

def stitch(list_chips, mask=False):
    if mask:
        image = np.zeros((2048, 2048))
    else:
        image = np.zeros((2048, 2048, 3))
        
    H, W = image.shape[0], image.shape[1]
    counter = 0
    for i in range(0, H, 256):
        for j in range(0, W, 256):
            if mask:
                image[i:i+256, j:j+256] = list_chips[counter]
            else:
                image[i:i+256, j:j+256, :] = list_chips[counter]
            counter +=1
            
    if mask:
        return image[:1923,:1923]
    else:
        return image[:1923, :1923, :]

def create_chips(image):
    mod_y = image.shape[0]%256
    mod_x = image.shape[1]%256
    diff_y = 256 - mod_y
    diff_x = 256 - mod_x
    base_img = np.zeros((image.shape[0] + diff_y, image.shape[1] + diff_x, 3), dtype=np.uint8)
    base_img[:image.shape[0], :image.shape[1], :] = image[:,:,:3]
    H, W = image.shape[0], image.shape[1]
    list_chips = []
    for i in range(0, H, 256):
        for j in range(0, W, 256):
            single_chip_img = base_img[i:i+256, j:j+256, :]
            list_chips.append(single_chip_img)

    return list_chips

def main(args):

    print(args)
    dd = defaultdict(list)
    out_dir = '/gypsum/eguide/projects/scorreacardo/urbano/data/worldpop_centered/'
    
    meta_df = pd.read_csv(f'/gypsum/home/scorreacardo/projects/urban_growth/data/processed/{args.group}_{args.case}_metadata.csv')
    print(meta_df.shape)
    for idx, item in tqdm(meta_df.iterrows()):
        #print(idx)
        if item.valid == False:
            dd['num_bldgs'].append(False)
            continue
        try:
            img = imread(os.path.join(out_dir, 'interim',args.case, args.group, item.tile_id,  item.path_to_file.split('/')[-1][:-4] + '.png'))
        except:
            print(item.path_to_file.split('/')[-1][:-4])
            dd['num_bldgs'].append(False)
            continue

        chips = create_chips(img)
        chips_pred = infer(args.exp_name, chips)
        pred_mask = stitch(chips_pred, mask=True)

        #calculating the number of structures in the ground truth dataset
        pred_copy = pred_mask.copy()
        pred_copy = np.uint8(pred_copy)
        buildings, _ = cv2.findContours(pred_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dd['num_bldgs'].append(len(buildings))

        os.makedirs(os.path.join(out_dir, 'predictions_gsd', args.case, args.group, item.tile_id), exist_ok=True)
        plt.imsave(os.path.join(out_dir, 'predictions_gsd', args.case, args.group, item.tile_id,  item.path_to_file.split('/')[-1][:-4] + '.png'), \
                      pred_mask, cmap="gray")


    meta_df['num_bldgs'] = dd['num_bldgs']
    meta_df.to_csv(f'/gypsum/home/scorreacardo/projects/urban_growth/data/processed/{args.group}_{args.case}_metadata_predictions_gsd.csv', index=False)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument("exp_name", help="experiment name")

    # Required positional argument
    parser.add_argument("case", help="case study")

    # Required positional argument
    parser.add_argument("group", help="treatment, control or control no mpi group")

    args = parser.parse_args()

    main(args)