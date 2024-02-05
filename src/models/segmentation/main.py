
__author__ = "Santiago Correa"
__version__ = "0.1.0"
__license__ = "MIT"

import torch
import numpy as np
import argparse
from train_model import Trainer
from predict_model import Predictor

def main(args):
    """ Main entry point of the app """
    print(args)
    print(torch.cuda.is_available())
    segmentation_model = Trainer(args.exp_name, args.gsd, args.model_name, args.encoder, args.loss, args.metric, int(args.bs), int(args.epochs), \
        float(args.lr), int(args.aug), int(args.manual), int(args.tuning))
    segmentation_model.train()

    #Now let's see how the model predicts in each image:
    #best_model = Predictor(args.exp_name, args.encoder, args.loss, args.metric, int(args.manual), int(args.tuning))
    #best_model.run()

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