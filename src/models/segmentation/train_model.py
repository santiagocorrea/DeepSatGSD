"""
Module Docstring
"""

__author__ = "Santiago Correa"
__version__ = "0.1.0"
__license__ = "MIT"

import torch
import numpy as np
import argparse
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
import matplotlib.pyplot as plt
from dataset import *
import pdb
import config

class Trainer:
    def __init__(self, exp_name, gsd, model_name, encoder_name, loss, metric, batch_size, n_epochs, lr, aug=None, manual=None, tuning=None):
        self.exp_name = exp_name
        self.gsd = gsd
        self.model = get_model(model_name, encoder_name, tuning)
        self.loss = get_loss(loss)
        self.metrics = get_metric(metric)
        self.optimizer = torch.optim.Adam([ dict(params=self.model.parameters(), lr=lr), ])
        self.lr = lr
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.encoder_name = encoder_name
        self.tuning = tuning
        self.aug = aug
        self.manual = manual
        self.model_name = model_name

    def train(self):
        if self.gsd == 'legacy':
            processed_data_path = config.PROCESSED_DATA_PATH_LEGACY
            if self.tuning == 1 or self.manual == 1:
                x_train_dir = os.path.join(processed_data_path, 'train_img_tuning')
                y_train_dir = os.path.join(processed_data_path, 'train_masks_tuning')

                x_val_dir = os.path.join(processed_data_path, 'val_img_tuning')
                y_val_dir = os.path.join(processed_data_path, 'val_mask_tuning')

            else:
                x_train_dir = os.path.join(processed_data_path, 'train_img')
                y_train_dir = os.path.join(processed_data_path, 'train_masks')

                x_val_dir = os.path.join(processed_data_path, 'val_img')
                y_val_dir = os.path.join(processed_data_path, 'val_masks')
            
        else:
            processed_data_path = config.PROCESSED_DATA_PATH
            processed_data_path = os.path.join(processed_data_path, self.gsd)

            if self.tuning == 1 or self.manual == 1:
                x_train_dir = os.path.join(processed_data_path, 'train_img_tuning')
                y_train_dir = os.path.join(processed_data_path, 'train_masks_tuning')

                x_val_dir = os.path.join(processed_data_path, 'val_img_tuning')
                y_val_dir = os.path.join(processed_data_path, 'val_mask_tuning')

            else:
                x_train_dir = os.path.join(processed_data_path, 'train/imgs')
                y_train_dir = os.path.join(processed_data_path, 'train/masks')

                x_val_dir = os.path.join(processed_data_path, 'valid/imgs')
                y_val_dir = os.path.join(processed_data_path, 'valid/masks')

        #x_test_dir = os.path.join(processed_data_path, 'test_img_tuning')
        #y_test_dir = os.path.join(processed_data_path, 'test_mask_tuning')
        
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder_name, "imagenet")

        if self.aug == 1:
            train_dataset = Dataset(
                x_train_dir, 
                y_train_dir, 
                augmentation=get_training_augmentation(), 
                preprocessing=get_preprocessing(preprocessing_fn),
                #classes=CLASSES,
            )
        else:
            train_dataset = Dataset(
                x_train_dir, 
                y_train_dir, 
                #augmentation=get_training_augmentation(), 
                preprocessing=get_preprocessing(preprocessing_fn),
                #classes=CLASSES,
            )

        valid_dataset = Dataset(
            x_val_dir, 
            y_val_dir, 
            #augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            #classes=CLASSES,
        )

        # test_dataset = Dataset(
        #     x_test_dir, 
        #     y_test_dir, 
        #     #augmentation=get_validation_augmentation(), 
        #     preprocessing=get_preprocessing(preprocessing_fn),
        #     #classes=CLASSES,
        # )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

        #create the epoch objects for train and validation 
        train_epoch = utils.train.TrainEpoch(
            self.model, 
            loss=self.loss, 
            metrics=self.metrics, 
            optimizer=self.optimizer,
            device='cuda',
            verbose=True,
        )

        valid_epoch = utils.train.ValidEpoch(
            self.model, 
            loss=self.loss, 
            metrics=self.metrics, 
            device='cuda',
            verbose=True,
        )
        max_score = 0
        
        train_scores = []
        val_scores = []
        for i in range(0, self.epochs):
            
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            #pdb.set_trace()
            valid_logs = valid_epoch.run(valid_loader)
            train_scores.append(train_logs['iou_score'])
            val_scores.append(valid_logs['iou_score'])

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, \
                    #f'../models/unet_{self.encoder_name}_{self.batch_size}_{self.epochs}_best_model.pth')
                    #f'../models/{self.exp_name}_best_model.pth')
                    f'/work/scorreacardo_umass_edu/DeepSatGSD/models/segmentation_{self.exp_name}_best_model.pth')
                print('Model saved!')
                
            if i == 25:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
            
            #if i > -1:
            if i%6 == 1 and i > 0:

                # load best saved checkpoint
                # best_model = torch.load(f'../models/unet_{self.encoder_name}_{self.batch_size}_{self.epochs}_best_model.pth')
                # dataset without transformations for image visualization
                #test_dataset_vis = Dataset(x_test_dir, y_test_dir,) #classes=CLASSES,)
                train_dataset_vis = Dataset(x_train_dir, y_train_dir)
                val_dataset_vis = Dataset(x_val_dir, y_val_dir)

                if self.tuning == 1 or self.manual == 1:
                    #image_vis_test = test_dataset_vis[10][0]
                    image_vis_train = train_dataset_vis[30][0]
                    image_vis_val = val_dataset_vis[30][0]

                    #image_test, gt_mask_test = test_dataset[10]
                    image_train, gt_mask_train = train_dataset[30]
                    image_val, gt_mask_val = valid_dataset[30]
                else:
                    #image_vis_test = test_dataset_vis[10][0]
                    image_vis_train = train_dataset_vis[30][0]
                    image_vis_val = val_dataset_vis[30][0]

                    #image_test, gt_mask_test = test_dataset[10]
                    image_train, gt_mask_train = train_dataset[30]
                    image_val, gt_mask_val = valid_dataset[30]

                #gt_mask_test = gt_mask_test.squeeze()
                gt_mask_train = gt_mask_train.squeeze()
                gt_mask_val = gt_mask_val.squeeze()

                #x_tensor_test = torch.from_numpy(image_test).to('cuda').unsqueeze(0)
                x_tensor_train = torch.from_numpy(image_train).to('cuda').unsqueeze(0)
                x_tensor_val = torch.from_numpy(image_val).to('cuda').unsqueeze(0)

                #pr_mask_test = best_model.predict(x_tensor_test)
                #pr_mask_test = self.model.predict(x_tensor_test)
                #pr_mask_test = (pr_mask_test.squeeze().cpu().numpy().round())
                #pr_mask_train = best_model.predict(x_tensor_train)
                pr_mask_train = self.model.predict(x_tensor_train)
                pr_mask_train = (pr_mask_train.squeeze().cpu().numpy().round())
                #pr_mask_val = best_model.predict(x_tensor_val)
                pr_mask_val = self.model.predict(x_tensor_val)
                pr_mask_val = (pr_mask_val.squeeze().cpu().numpy().round())
                
                #save image with predictions to visualize progress of the learning approach
                # visualize(self.exp_name, "test", self.encoder_name, self.batch_size, i,
                #     image=image_vis_test, 
                #     ground_truth_mask=gt_mask_test, 
                #     predicted_mask=pr_mask_test
                # )

                visualize(self.exp_name, "train", self.encoder_name, self.batch_size, i,
                    image=image_vis_train, 
                    ground_truth_mask=gt_mask_train, 
                    predicted_mask=pr_mask_train
                )

                visualize(self.exp_name, "val", self.encoder_name, self.batch_size, i,
                    image=image_vis_val, 
                    ground_truth_mask=gt_mask_val, 
                    predicted_mask=pr_mask_val
                )

        #create and save graph
        plt.figure()
        plt.plot(range(0, self.epochs), train_scores, 'y', label='Training IOU')
        plt.plot(range(0, self.epochs), val_scores, 'r', label='Validation IOU')
        plt.title('Training and validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.legend()
        plt.savefig(f'/work/scorreacardo_umass_edu/DeepSatGSD/reports/figs/train_curve_segmentation_{self.exp_name}.png', \
            dpi=500)



def get_model(model_name, encoder_name, tuning=None):
    if tuning == 1:
        model = torch.load(f'../models/unet_exp5_best_model.pth')
    else:
        if model_name == 'unet':
            model = smp.Unet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif model_name == 'deeplabv3':
            model = smp.DeepLabV3(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif model_name == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif model_name == 'pspnet':
            model = smp.PSPNet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif model_name == 'pan':
            model = smp.PAN(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
    return model

def get_loss(loss):
    if loss == 'dice':
        return utils.losses.DiceLoss(activation='sigmoid')
    else:
        return utils.losses.JaccardLoss(activation='sigmoid')

def get_metric(metric):
    if metric == 'iou':
        metrics = [
            utils.metrics.IoU(threshold=0.5),
        ]
        return metrics

# helper function for data visualization
def visualize(exp_name, img_set, encoder_name, batch_size, epoch, **images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    #plt.savefig(f'../reports/figures/{exp_name}_{img_set}_{epoch}.png', dpi=500)
    plt.savefig(f'/work/scorreacardo_umass_edu/DeepSatGSD/reports/figs/segmentation_{exp_name}_{img_set}_{epoch}.png', dpi=500)


def main(args):
    """ Main entry point of the app """
    print(args)
    print(torch.cuda.is_available())
    segmentation_model = Trainer(args.exp_name, args.model_name, args.encoder, args.loss, args.metric, int(args.bs), int(args.epochs), \
        float(args.lr), int(args.aug), int(args.manual), int(args.tuning))
    segmentation_model.train()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("exp_name", help="name of the experiment")
    parser.add_argument("gsd", help="ground sample distance to train")
    parser.add_argument("model_name", help="name of the model e.g. unet, deeplabv3, deeplabv3plus")
    parser.add_argument("encoder", help="Encoder name e.g. 'resnet34'")
    parser.add_argument("loss", help="e.g. 'dice' or 'jaccard'")
    parser.add_argument("metric", help="e.g. 'iou'")
    parser.add_argument("bs", help="Batch size e.g. '16'")
    parser.add_argument("epochs", help=" e.g. '40'")
    parser.add_argument("lr", help="Learning rate e.g. '0.0001'")
    parser.add_argument("aug", help="Augmentation flag")
    parser.add_argument("manual", help="flag for training only with manually labeled images")
    parser.add_argument("tuning", help="flag for tuning with the best model (exp5)")

    args = parser.parse_args()
    main(args)