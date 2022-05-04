# William Collins
# Image prep methods for creating tensor files for training/testing networks.
# Code used from by https://github.com/praritagarwal/Visualizing-CNN-Layers/blob/master/Activation%20Maximization.ipynb

import torch
from torchvision import transforms

import numpy as np


#normalize the image tensor by subtracting mean and dividing by std
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# undo the above normalization if and when the need arises 
denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                   std = [1/0.229, 1/0.224, 1/0.225] )


def random_image(ht=28, wd=28):
    '''Create a noise image of specific dimensions'''
    # we need the pixel values to be of type float32
    img = np.single(np.random.uniform(0,1, (3, ht, wd))) 
    
    # normalize the image to have requisite mean and std. dev.
    im_tensor = normalize(torch.from_numpy(img)).to(device).requires_grad_(True) 
    print("img_shape:{}, img_dtype: {}".format(im_tensor.shape, im_tensor.dtype ))

    return im_tensor


def image_converter(im):
    '''Convert img_tensor for input to plt.imshow()'''
    # for plt.imshow() the channel-dimension is the last therefore use transpose to permute axes
    im_copy = denormalize(im.clone().detach()).numpy().transpose(1,2,0)
    
    # clip negative values as plt.imshow() only accepts floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1) 
    
    return im_copy

