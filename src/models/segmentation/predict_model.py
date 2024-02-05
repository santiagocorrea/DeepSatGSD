"""
Module Docstring
"""

__author__ = "Santiago Correa"
__version__ = "0.1.0"
__license__ = "MIT"

import torch
import numpy as np
import pandas as pd
import argparse
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from collections import defaultdict
from dataset import *

class Predictor:
    def __init__(self, exp_name, encoder_name, loss, metric, manual, tuning):
        self.exp_name = exp_name
        self.encoder_name = encoder_name
        self.loss = get_loss(loss)
        self.metrics = get_metric(metric)
        self.model = get_model(exp_name)
        self.tuning = tuning
        self.manual = manual

    def run(self):
        #if self.tuning == 1 or self.manual == 1:
        x_test_dir = os.path.join(processed_data_path, 'test_img_tuning')
        y_test_dir = os.path.join(processed_data_path, 'test_mask_tuning')

        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder_name, "imagenet")
        
        test_dataset = Dataset(
            x_test_dir, 
            y_test_dir, 
            #augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            #classes=CLASSES,
        )

        test_dataloader = DataLoader(test_dataset)

        # evaluate model on test set
        test_epoch = smp.utils.train.ValidEpoch(
            model=self.model,
            loss=self.loss,
            metrics=self.metrics,
            device='cuda',
        )

        logs = test_epoch.run(test_dataloader)
        
        dd = defaultdict(list)
        
        #calculate metrics for building count detection and area
        for i in range(len(test_dataset)):
            image_test, gt_mask_test = test_dataset[i]
            gt_mask_test = gt_mask_test.squeeze()
            x_tensor_test = torch.from_numpy(image_test).to('cuda').unsqueeze(0)
            pr_mask_test = self.model.predict(x_tensor_test)
            pr_mask_test = (pr_mask_test.squeeze().cpu().numpy().round())

            pr_mask_test_th = sigmoid(pr_mask_test)
            pr_mask_test_th = pr_mask_test_th > 0.5
            pr_mask_test_th = np.uint8(np.multiply(pr_mask_test_th, 255))

            gt_mask_test = np.uint8(gt_mask_test)
            #calculating the number of structures in the ground truth dataset
            gt_buildings, _ = cv2.findContours(gt_mask_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dd['gt_buildings'].append(len(gt_buildings))
            
            #calculating the ratio prediction/ground_truth per patch
            dd['perc_area'].append(perc_area(pr_mask_test_th, gt_mask_test))

            #calculating the squared error for the number of buildings in each patch
            dd['SE'].append(se(pr_mask_test_th, gt_mask_test))

            #calculating the percent error for the number of buildings in each patch
            dd['perc_error'].append(perc_error(pr_mask_test_th, gt_mask_test))

            #calculating IoU per patch:
            dd['iou'].append(iou(pr_mask_test_th, gt_mask_test))


        print(f'IoU model {self.exp_name}:')
        print(logs['iou_score'])

        df = pd.DataFrame(dd)
        print(f'MAPE: {df.perc_error.mean()}')
        print(f'MSE: {df.SE.mean()}')
        df.to_csv(f'../reports/{self.exp_name}.csv', index=False)

        #visualize predictions
        # test dataset without transformations for image visualization
        test_dataset_vis = Dataset(
            x_test_dir, y_test_dir, 
        )
        for i in range(5):
            n = np.random.choice(len(test_dataset))
            
            image_vis = test_dataset_vis[n][0]
            image, gt_mask = test_dataset[n]
            
            gt_mask = gt_mask.squeeze()
            
            x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
            pr_mask = self.model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                
            visualize(self.exp_name, n, 
                image=image_vis, 
                ground_truth_mask=gt_mask, 
                predicted_mask=pr_mask
            )


def get_model(exp_name):
    model = torch.load(f'../models/{exp_name}_best_model.pth')
    #model = torch.load(f'../models/unet_resnet34_16_40_best_model.pth')
    return model

def get_loss(loss):
    if loss == 'dice':
        return smp.utils.losses.DiceLoss(activation='sigmoid')
    else:
        return smp.utils.losses.JaccardLoss(activation='sigmoid')

def get_metric(metric):
    if metric == 'iou':
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        return metrics

def sigmoid(X):
    return 1/(1+np.exp(-X))

def se(pr_mask, gt_mask):
    pr_contours, _ = cv2.findContours(pr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq_error = (len(pr_contours) - len(gt_contours))**2
    return sq_error

def perc_error(pr_mask, gt_mask):
    pr_contours, _ = cv2.findContours(pr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perc_error = (len(pr_contours) - len(gt_contours))/len(gt_contours)
    return perc_error*100

def perc_area(pr_mask, gt_mask):
    #area as percentage of total area
    pr_pixels = cv2.countNonZero(pr_mask)
    gt_pixels = cv2.countNonZero(gt_mask)
    return (pr_pixels/gt_pixels)*100

def iou(pr_mask, gt_mask):
    pr_mask_area = np.count_nonzero(pr_mask)
    gt_mask_area = np.count_nonzero(gt_mask)
    intersection = np.count_nonzero(np.logical_and(pr_mask, gt_mask))
    iou = intersection/(pr_mask_area + gt_mask_area - intersection)
    return iou

def mask_to_vect(mask):
    pass

# helper function for data visualization
def visualize(exp_name, sample_n, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(f'../reports/figures/{exp_name}_predictor_{sample_n}.png', dpi=500)
    

def main(args):
    """ Main entry point of the app """
    print(args)
    print(torch.cuda.is_available())
    segmentation_model = Predictor(args.exp_name, args.encoder, args.loss, args.metric, int(args.manual), int(args.tuning))
    segmentation_model.run()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("exp_name", help="name of the experiment")
    parser.add_argument("encoder", help="Encoder name e.g. 'resnet34'")
    parser.add_argument("loss", help="e.g. 'dice' or 'jaccard'")
    parser.add_argument("metric", help="e.g. 'iou'")
    parser.add_argument("manual", help="flag for training only with manually labeled images")
    parser.add_argument("tuning", help="flag for tuning with the best model (exp5)")

    args = parser.parse_args()
    main(args)
