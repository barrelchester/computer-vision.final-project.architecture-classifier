# William Collins
# Image prep methods for creating tensor files for training/testing networks.

import os, shutil, math, random
import numpy as np
import torch
import cv2



def get_image_sizes(images_path):
    '''Get a dictionary of image sizes mapped to list of image paths'''
    im_sizes = {}
    
    #images are in class folders
    for cls in os.listdir(images_path):
        if '.' in cls:
            continue
            
        print(cls)
        
        cls_path = '%s/%s' % (images_path, cls)
        
        #read each image in the class folder and record its size
        for fn in os.listdir(cls_path):
            fp = '%s/%s' % (cls_path, fn)
            
            #read the image with opencv2
            try:
                im = cv2.imread(fp)
            except Exception as ex:
                print('couldnt open %s: %s' % (fp, str(ex)))
                continue
                
            if im is None:
                print('couldnt open %s' % (fp))
                continue
                
            #get and store the image size
            sz = im.shape
            if sz not in im_sizes:
                im_sizes[sz]=[]
                
            im_sizes[sz].append('%s/%s' % (cls, fn))
        
    return im_sizes


def resize_and_crop(im_sizes, originals_path):
    '''Resize images so that the lesser dimension is 128 and the larger dimension is center cropped.
    Store original images in originals_path rather than overwriting them.'''
    i=0
    
    #iterate over the sizes and path lists stored in im_sizes dict
    for sz, paths in im_sizes.items():
        #if this is already the correct size, continue
        if sz[0]==128 and sz[1]==128:
            continue
            
        #iterate over the images to be resized
        for p in paths:
            if not os.path.exists(p):
                continue
                
            i+=1
            if i%100==0:
                print(i, p)
                
            #read the image with opencv2
            im = cv2.imread(p)
            
            #in case it was already resized, continued
            sz = im.shape
            if sz[0]==128 and sz[1]==128:
                print('already resized', p)
                continue

            #resize the image so that the smaller dimension is 128
            wd = sz[0]
            ht = sz[1]
            
            #if a dimension is already 128 resizing is not needed
            if sz[0]>128 and sz[1]>128:
                
                #width to 128, height to new val
                if sz[0] < sz[1]:
                    wd = 128
                    ht = (128*sz[1])//sz[0]
                    
                    #resize the image
                    im = cv2.resize(im, (ht, wd), interpolation = cv2.INTER_LINEAR)
                    
                #height to 128, width to new val
                else:
                    wd = (128*sz[0])//sz[1]
                    ht = 128
                    
                    #resize the image
                    im = cv2.resize(im, (ht, wd), interpolation = cv2.INTER_LINEAR)
                    
            #crop the larger dimension so that it is centered
            if wd>ht: 
                #crop width to 128, centered
                side = math.ceil((wd - ht)/2)
                im = im[side:, :, :]
                im = im[:128, :, :]
            else:
                #crop height to 128, centered
                side = math.ceil((ht - wd)/2)
                im = im[:, side:, :]
                im = im[:, :128, :]
            
            #move original file to originals folder rather than overwriting
            if im.shape[0]==128 and im.shape[1]==128:
                cls, fn = p.split('/')[-2:]
                
                if not os.path.exists('%s/%s' % (originals_path, cls)):
                    os.mkdir('%s/%s' % (originals_path, cls))
                
                #move the original image
                shutil.move(p, '%s/%s/%s' % (originals_path, cls, fn))
                
                #write the resized cropped image to the source folder
                cv2.imwrite(p, im)
            else:
                print('image is wrong size')
                continue
                
                
def get_lab_to_idx(images_path):
    '''Get mapping of class name to index, for creating y tensors.'''
    #for every class folder name, take the lower case and replace spaces with underbar
    return {cls.replace(' ','_'):i for i,cls in enumerate(sorted([fn.lower() for fn in os.listdir(images_path)]))}
                
                
def images_to_tensors(images_path, tensors_path, lab2idx, im_size=128, max_per_file=2000):
    '''Convert resized image files to tensors for training/testing models.'''
    im_arrays = []
    y=[]
    
    #imagenet mean and std: x = (x/255-mean)/std
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    #get all paths, shuffle, save max_per_file in each
    paths = get_images_paths(images_path)
    random.shuffle(paths)
    
    #batch using max_per_file
    for i in range(0, len(paths), max_per_file):
        
        #iterate over a batch of image paths
        for cls_fn in paths[i:i+max_per_file]:
            cls, fn = cls_fn.split('/')
            
            #normalize the class name; lowercase with ' ' replaced with '_'
            cls_name = cls.replace(' ','_').lower()
            
            fp = '%s/%s' % (images_path, cls_fn)
            
            #read image and convert to RGB
            im = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            
            #add image and label index to lists
            im_arrays.append(im)
            y.append(lab2idx[cls_name])
        
        #shape image arrays to batch,size,size,3
        im_arrays = np.reshape(im_arrays, (-1, im_size, im_size, 3))
        
        #convert it to a torch tensor
        im_arrays = torch.from_numpy(im_arrays)
        
        #normalize: divide by 255, subtract mean, divide by std
        im_arrays = ((im_arrays/255)-mean)/std
        
        #move channel axis: batch,channel,size,size
        im_arrays = torch.swapaxes(im_arrays, 3, 1)

        #save
        print('saving %s/x_%d.npy' % (tensors_path, i))
        torch.save(im_arrays, '%s/x_%d.pt' % (tensors_path, i))
        torch.save(torch.tensor(y), '%s/y_%d.pt' % (tensors_path, i))
        
        im_arrays=[]
        y=[]
        
        
def get_images_paths(images_path):
    '''Get list of image paths.'''
    paths = []
    
    for cls in os.listdir(images_path):
        if '.' in cls:
            continue
            
        cls_p = '%s/%s' % (images_path, cls)
        
        for fn in os.listdir(cls_p):
            paths.append('%s/%s' % (cls, fn))
            
    return paths


def get_training_cv_test_data(tensor_path):
    '''Get cross validation (cv) and test tensors and list of tensor paths containing training data.'''
    #get the list of tensor files and sort them
    xs = list(sorted([fn for fn in os.listdir(tensor_path) if 'x_' in fn and not '_0.' in fn]))
    ys = list(sorted([fn for fn in os.listdir(tensor_path) if 'y_' in fn and not '_0.' in fn]))
    
    #choose first file as test and second file as cv
    x_test = torch.load('%s/%s' % (tensor_path, xs[0]))
    y_test = torch.load('%s/%s' % (tensor_path, ys[0]))
    x_cv  = torch.load('%s/%s' % (tensor_path, xs[1]))
    y_cv = torch.load('%s/%s' % (tensor_path, ys[1]))
    
    #exclude test and cv paths from the paths of training data
    xs=xs[2:]
    ys=ys[2:]
    
    return xs, ys, x_cv, y_cv, x_test, y_test
    

