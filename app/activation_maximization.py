# William Collins
# Image prep methods for creating tensor files for training/testing networks.
# Code inspired but not copied directly from https://github.com/praritagarwal/Visualizing-CNN-Layers/blob/master/Activation%20Maximization.ipynb

import sys

import cv2

import torch
from torch import nn, optim

import matplotlib.pyplot as plt

import image_utils as utils



def maximize_activation_simple(model, activation, layer_name, unit_idx=0,
                               optim_steps=40, optim_lr=0.4, model_name='inception'):
    '''Maximize the activation of a chosen neuron in a chosen layer having a forward_hook to store
    the activation in the activation dict.
    
    Penalize only the negative RMS of the activation value.
    '''
    # the neuron to visualize
    model.eval()
    
    #create a random noise image
    img_tensor = random_image(256, 256)
    
    #the optimizer is only optimizing the input image and must be 
    optimizer = optim.Adam([img_tensor], lr=optim_lr)

    #iteratively optimize the input image
    for opt_epoch in range(optim_steps):
        optimizer.zero_grad()

        #when model runs, the hooked layer is hit, which populates the activation dict with results
        model(img_tensor.unsqueeze(0))

        #the activation output of the hooked layer that we want to maximize
        layer_out = activation[layer_name]

        #calculate the root mean square of the activation of the selected neuron
        if unit_idx < 0:
            #if -1 is passed in as the unit, then the whole layer is optimized
            rms = torch.pow((layer_out[0, :]**2).mean(), 0.5)
        else:
            rms = torch.pow((layer_out[0, unit_idx]**2).mean(), 0.5)
            
        # terminate if rms is nan
        if torch.isnan(rms):
            print('Error: rms was Nan; Terminating ...')
            sys.exit()

        loss = -act_wt*rms

        #calculate the gradients
        loss.backward()

        #update the input image
        optimizer.step()

    #convert image to be viewable
    img = image_converter(img_tensor)
    
    #display and save image
    plt.imshow(img)
    plt.savefig('%s_%s_%d.png' % (model_name, layer_name, unit_idx), bbox_inches='tight')
    plt.show()
    
    

def maximize_activation_with_scaling(model, gradLayer, activation, layer_name, unit_idx=0,
                            upscaling_steps=20, upscaling_factor=1.05, optim_steps=40, optim_lr=0.4, act_wt=0.5,
                            model_name='inception', penalize_intensity=True, penalize_sharp_gradients=True):
    '''Maximize the activation of a chosen neuron in a chosen layer having a forward_hook to store
    the activation in the activation dict. 
    
    Penalize the negative RMS of the activation value, with options to additionally penalize pixel intensity 
    and sharp gradients. The sharp gradient penalty makes a huge improvement in quality.
    
    Starts with a small noise image and progressively scales the result to smooth out gradient updates.'''
    #set model to eval mode
    model.eval()
    
    #create a random noise image
    img_tensor = utils.random_image(80, 80)
    
    #start with a small image and upscale repeatedly to smooth out the resulting pattern
    for mag_epoch in range(upscaling_steps+1):
        
        #the optimizer is only optimizing the input image and must be 
        optimizer = optim.Adam([img_tensor], lr=optim_lr)

        #iteratively optimize the input image
        for opt_epoch in range(optim_steps):
            optimizer.zero_grad()
            
            #when model runs, the hooked layer is hit, which populates the activation dict with results
            model(img_tensor.unsqueeze(0))
            
            #the activation output of the hooked layer that we want to maximize
            layer_out = activation[layer_name]

            #calculate the root mean square of the activation of the selected neuron
            if unit_idx < 0:
                #if -1 is passed in as the unit, then the whole layer is optimized
                rms = torch.pow((layer_out[0, :]**2).mean(), 0.5)
            else:
                rms = torch.pow((layer_out[0, unit_idx]**2).mean(), 0.5)
                
            # terminate if rms is nan
            if torch.isnan(rms):
                print('Error: rms was Nan; Terminating ...')
                sys.exit()
                
            loss = -act_wt*rms

            # pixel intensity penalty
            if penalize_intensity:
                pxl_inty = torch.pow((img_tensor**2).mean(), 0.5)
                
                # terminate if pxl_inty is nan
                if torch.isnan(pxl_inty):
                    print('Error: Pixel Intensity was Nan; Terminating ...')
                    sys.exit()
                loss += pxl_inty

            # image gradients-compute gradient loss of an image using the above defined gradLayer
            if penalize_sharp_gradients:
                gradSq = gradLayer(img_tensor.unsqueeze(0))**2
                im_grd = torch.pow(gradSq.mean(), 0.5)
                
                # terminate is im_grd is nan
                if torch.isnan(im_grd):
                    print('Error: image gradients were Nan; Terminating ...')
                    sys.exit()
                    
                loss += im_grd
   
            # print activation at the beginning of each mag_epoch
            if opt_epoch == 0 and mag_epoch%5==0:
                print('begin mag_epoch {}, activation: {}'.format(mag_epoch, rms))
                
            #calculate the gradients
            loss.backward()
            
            #update the input image
            optimizer.step()

        #convert image from tensor form to a form that can be manipulated by cv2
        img = utils.image_converter(img_tensor)

        # scale up
        img = cv2.resize(img, dsize = (0,0), fx = upscaling_factor, fy = upscaling_factor)
        
        #transform back into torch tensor (move the color axis to be the first)
        img_tensor = utils.normalize(torch.from_numpy(img.transpose(2,0,1))).to(device).requires_grad_(True)

    #convert image to be viewable
    img = utils.image_converter(img_tensor)
    
    #display and save the image
    plt.imshow(img)
    plt.savefig('%s_%s_%d.png' % (model_name, layer_name, unit_idx), bbox_inches='tight')
    plt.show()
    
    
    
def maximize_activation_octaves(model, gradLayer, activation, layer_name, unit_idx=-1, im_size=(256,256), 
                                 num_octaves=4, octave_scale=1.4, optim_steps=20, optim_lr=0.4, act_wt=0.75,
                               model_name='inception'):
    # the neuron to visualize
    model.eval()
    
    img_tensor = utils.random_image(im_size[0], im_size[1])
    img = utils.image_converter(img_tensor)
    print('original im size ', img.shape)
    
    #img is shrunk
    img, octaves = get_octaves(img, num_octaves, octave_scale)
    img_tensor = utils.normalize(torch.from_numpy(img.astype(np.float32).transpose(2,0,1))).to(device).requires_grad_(True)
    print('img size after octaves: ', img_tensor.size())
    print('octaves', [o.shape for o in octaves])   
    
    # generate details octave by octave
    for i in range(octave_n):
        print('octave %d, im size %s' % (i, str(img_tensor.size())))
        if i>0:
            hi = octaves[-i] #smallest first
            
            #grow the image, add hi back in
            img = utils.image_converter(img_tensor)
            print('resizing image ', img.shape)
            img = cv2.resize(img, dsize=(hi.shape[:2])) + hi #*0.5
            img = cv2.blur(img, (5,5))
            print('to ', img.shape)
            img_tensor = utils.normalize(torch.from_numpy(img.astype(np.float32).transpose(2,0,1))).to(device).requires_grad_(True)
        
        optimizer = optim.Adam([img_tensor], lr=1.0-(i*0.2)) #optim_lr)
        #print('lr', (0.5-(i*0.1)))
        
        #get grad for current size image
        for opt_epoch in range(optim_steps - (i*25)):
            optimizer.zero_grad()
            
            #when model runs, the hooked layer is hit, which populates the activation dict with results
            model(img_tensor.unsqueeze(0))
            
            #the activation output of the hooked layer that we want to maximize
            layer_out = activation[layer_name]
            
            if unit_idx < 0:
                rms = torch.pow((layer_out[0, :]**2).mean(), 0.5)
            else:
                rms = torch.pow((layer_out[0, unit_idx]**2).mean(), 0.5)
            # terminate if rms is nan
            if torch.isnan(rms):
                print('Error: rms was Nan; Terminating ...')
                sys.exit()

            # pixel intensity
            pxl_inty = torch.pow((img_tensor**2).mean(), 0.5)
            # terminate if pxl_inty is nan
            if torch.isnan(pxl_inty):
                print('Error: Pixel Intensity was Nan; Terminating ...')
                sys.exit()

            # image gradients-compute gradient loss of an image using the above defined gradLayer
            gradSq = gradLayer(img_tensor.unsqueeze(0))**2
            im_grd = torch.pow(gradSq.mean(), 0.5)
            # terminate is im_grd is nan
            if torch.isnan(im_grd):
                print('Error: image gradients were Nan; Terminating ...')
                sys.exit()

            loss = -act_wt*rms + pxl_inty + im_grd        
            # print activation at the beginning of each mag_epoch
            if opt_epoch % 5 == 0:
                print('begin opt_epoch {}, activation: {}'.format(opt_epoch, rms))
            loss.backward()
            optimizer.step()
        
        img = utils.image_converter(img_tensor)
        plt.imshow(img)
        plt.savefig('%s_%s_%d.png' % (model_name, layer_name, unit_idx), bbox_inches='tight')
        plt.show()


def get_octaves(img, num_octaves=4, octave_scale=1.4):
    octaves = []
    for i in range(num_octaves-1):
        h,w,c = img.shape

        #shrink
        lo = cv2.resize(img, dsize=(0,0), fx=1/octave_scale, fy=1/octave_scale)#.transpose(2,0,1)
        print(lo.shape)

        #expand and remove expanded version from original version
        hi = img - cv2.resize(lo, dsize=(w,h))

        #store difference
        octaves.append(hi)

        #set img to shrunk one
        img = lo
        
    return img, octaves
    
    
    
    