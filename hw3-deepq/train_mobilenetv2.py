# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

import torch
import random
import numpy as np
import datetime
from torchvision import transforms

resize_size = 224
model_pick = ["mobilenetv2", "resnet50"]   
model_picknum = 0# 0 is 
dropout_value = 0.5

def get_random_seed():
    """
    this function is called once before the training starts

    returns:
        an integer as the random seed, or None for random initialization
    """
    seed = None
    return seed

def get_model_spec():
    """
    this function is called once for setting up the model

    returns:
        a dictionary contains following items:
            model_name: one of the 'resnet50' and 'mobilenetv2'
            pretrained: a boolean value indicating whether to use pre-trained model
                        if False is returned, default initialization will be used
    """
    print('use', model_pick[model_picknum], 'to train this time')
    return {"model_name": model_pick[model_picknum], "pretrained": True}

def get_optimizer(params):
    """
    this function is called once for setting up the optimizer

    args:
        params: the set of the parameters to be optimized
                should be passed to torch.optim.Optimizer

    returns:
        an torch.optim.Optimizer optimizing the given params

    notes:
        don't modify params
    """

    if model_pick[model_picknum] is not "resnet50":
        optimizer = torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.003, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(params, lr=0.001)  # 0.1
    
    return optimizer

def get_eval_spec():
    """
    this function is called once for setting up evaluation / inferencing

    returns:
        a dictionary contains following items:
            transform: a transform used to preprocess evalutaion / inferencing images
                       should be a callable which takes a PIL image of 3 x 256 x 256
                       and produce either a 3 x 244 x 244 tensor or a NC x 3 x 244 x 244 tensor
                        in the latter case:
                            NC stands for the number of crops of an image,
                            NC predictions will be inferenced on the NC crops
                            then those predictions will be average to produce a final prediction
            batchsize: an integer between 1 and 64
    """
    transform =  transforms.Compose([
        transforms.Resize(resize_size),
        # transforms.RandomRotation(20),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        # transforms.RandomAffine(10),
        transforms.ToTensor()
        ])
    return {"transform": transform, "batchsize": 32}


def before_epoch(train_history, validation_history):
    """
    this function is called before every training epoch

    args:
        train_history:
            a 3 dimensional python list (i.e. list of list of list)
            the j-th element in i-th list is a list containing two entries,
            stands for the [accuracy, loss] for the j-th batch in the i-th epoch

            len(train_history) indicate the index of current epoch
            this value should be within the range of 1~50

        validation_history:
            a 2 dimentsional python list (i.e. list of list)
            the i-th element is a list containing two entry,
            stands for the [accuracy, loss] for the validation result of the i-th epoch

    returns:
        a dictionary contains following items:
            transform: a transform used to preprocess training images
                       should be a callable which takes a PIL image of 3 x 256 x 256
                       and produces a 3 x 224 x 224 tensor
            batchsize: an integer between 1 and 64
    """
    if model_pick[model_picknum] is not "resnet50":
        transform = transforms.Compose([
            transforms.Resize(resize_size),
        # transforms.FiveCrop(),
        # transforms.RandomCrop(),
        
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.ToTensor(),
        # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            # transforms.FiveCrop(),
            # transforms.RandomCrop(100),
            # transforms.RandomApply([transforms.RandomGrayscale(3)], 0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(18),
            transforms.RandomAffine(18, translate=(0.2,0.2)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.5, hue=0.1),
            transforms.ToTensor(),
        # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
    n_epoch = len(train_history)

    _batch_size = 32
    if model_pick[model_picknum] is not "resnet50":
        cof = n_epoch // 10
        _batch_size = _batch_size + 10*cof
        if _batch_size >= 64:
            _batch_size = 64

    # print(train_history)
    return {"transform": transform, "batchsize": _batch_size}

def before_batch(train_history, validation_history):
    """
    this function is called before each training batch

    args: please refer to before_epoch()

    returns:
        a dictionary contains the following items:
            optimizer: a dictionary of optimizer hyperparameters
            batch_norm: a float stands for the value of momentum in all batch normalization layers
                        or None indicating no changes should be made
            drop_out: a float stands for the value of drop out probability
                      or None indicating no changes should be made

    notes:
        drop_out should always be None when using resnet50 since there are no dropout layers in it
    """
    n_epoch = len(train_history)
    _lr = 0.001
    
    if model_pick[model_picknum] is not "resnet50":
        dropout = dropout_value
        cof_lr = n_epoch // 15
        _lr = _lr / 10 ** cof_lr
    else:
        dropout = None

    return {"optimizer": {"lr": _lr}, "batch_norm": 0.5, "drop_out": dropout}

def save_model_as(train_history, validation_history):
    """
    this function is called after each epoch's training
    the returned value will be used to determine whether to save the model at this point or not

    args: please refer to before_epoch()

    returns:
        a string, the filename as which the model is going to be saved
        or None indicating no saving is desired for this epoch
    """
    best_val = 0.0
    best_n = 50
    n_epoch = len(train_history)
    print(n_epoch)
    # validation_history = np.array(validation_history)
    # print('validation_history_shape', validation_history)
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d_%H:%M:%S")

    if n_epoch <= 50:
        if n_epoch == 50:
            for n in range(n_epoch):
                if validation_history[n][0] > best_val:
                    best_val = validation_history[n][0]
                    best_n = n + 1
            print('best_val:', best_val)
            print('best_n:', best_n)


        return 'save_'+model_pick[model_picknum]+'_'+str(n_epoch)+'_'+str(otherStyleTime)+'_'+'.pth.tar'
    else:
        return None
    


