#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/ImageClassifier/train.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Karry Harsh
# DATE CREATED: 12/02/2019
# REVISED DATE:

# Imports python modules

import torch
from torch import nn
from torch import optim
import json
import torchvision 
from torchvision import datasets, transforms, models
import time
import numpy as np
import pandas as pd
import argparse
import os
from PIL import Image

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. dataset directory as --data_dir with default value 'flowers'
      2. Path to an image as --path_to_image with default value '/15/image_06351.jpg'
      3. Path to JSON file as --category_names with default value cat_to_name.json
      4. Run model on CPU or GPU as --to_device with default value 'gpu'
      5. Top k most likely classes prediction as --top_k with default value 5
      6. Save model as --sav_dir with deault value 'checkpoint.pth'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    arg_parser = argparse.ArgumentParser(description='Predicting file')
    
    # command line options
    arg_parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                         help = 'Flower images directory')
    arg_parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                         help = 'Model checkpoints directory')
    arg_parser.add_argument('--path_to_image', type = str, default = '/15/image_06351.jpg', 
                         help = 'Path to an image')  
    arg_parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                         help = 'Path to JSON file containing category labels')
    arg_parser.add_argument('--device_type', type = str, default = 'gpu', 
                         help = 'Run model on CPU or GPU')
    arg_parser.add_argument('--top_k', type = int, default = 5, 
                         help = 'Top k most likely classes prediction')
    return arg_parser.parse_args()

def load_saved_checkpoint(model_path):
    """
    loads a saved checkpoint and rebuilds the model
    """
    # Load the saved model from the checkpoint
    checkpoint = torch.load(model_path)
    arch_name = checkpoint['arch_name']
    if (arch_name == 'densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    criterion = nn.NLLLoss()
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['model_class_index']
    #freeze the parameter
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint['model_class_index']

def process_image(image):
    '''
    Defined to pre process the image, it will crop the image to select the center portion of
    the image the convert the image to numpy set of array then normalize the array between 0-1
    after that calculate the standard deviation of the numpy array and transpose the image
    '''
    # Read the Image
    pil_image = Image.open(image)
    # Resize the image
    width, height = pil_image.size
    short_side = min(width, height)
    pil_image = pil_image.resize( (int((width / short_side)*256), int((height / short_side)*256)) )
    # Crope the image to get the center portion
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # convert image into numpy array and normalize it
    np_image = np.array(pil_image)    
    np_image = np_image / 255
    # calculate the standard deviation of the numpy array and transpose 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return torch.Tensor(np_image)

def predict(image_path, model,device_type,topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    # Determine the type of device processing
    if (device_type == 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model.to(device)
    model.eval() # inference mode
    # implement the code to predict the class from an image file
    img = process_image(image_path)

    img = img.to(device)
    

    img = img.unsqueeze(0)  
    # temporarily set all the requires_grad flag to false
    with torch.no_grad():
        output = model.forward(img)
        # https://pytorch.org/docs/stable/torch.html#torch.topk
        # returns the topk largest elements of the given input tensor
        probs, probs_labels = torch.topk(output, topk)
        probs = probs.exp() # calc all exponential of all elements
        class_to_idx = model.class_to_idx
     
    # Use Tensor.cpu() to copy the tensor to host memory first.
    
    probs = probs.cpu().numpy()
    probs_labels = probs_labels.cpu().numpy()
    
    # gets the indexes in numerical order: 0 to 101
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    
    classes_list = list()
    
    for label in probs_labels[0]:
        
        classes_list.append(idx_to_class[label])
        
    return (probs[0], classes_list)

def show_prediction(probs, classes, json_category_names):
    """
    Display the top probabilites of the Image with its name.
    """
    # Load the File which contains the image name.
    with open(json_category_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[i] for i in classes]

    result = pd.DataFrame(
        {'flowers': pd.Series(data=flower_names),
         'probabilities': pd.Series(data=probs, dtype='float64')
        })
    print(result)
def main():
    
    # Define get_input_args function within the file train.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable args
    arg = get_input_args()
    # Define path to the image in variable
    data_dir = arg.data_dir
    test_dir = data_dir + 'test/'
    image = arg.path_to_image
    print('')
    print('data_dir: {}'.format(data_dir))
    print('test_dir: {}'.format(test_dir))
    print('save_dir: {}'.format(arg.save_dir))
    print('image_file: {}'.format(test_dir+image))
    print('category_names: {}'.format(arg.category_names))
    print('device_type: {}'.format(arg.device_type))
    print('top_k: {}'.format(arg.top_k))
    print('')
    # make sure checkpoint exists
    if os.path.exists('checkpoint.pth'):
        # Define load_saved _checkpoint function with in predict.py
        # it will rebuild the model for prediction
        model, class_to_idx = load_saved_checkpoint(arg.save_dir)
        # Define predict function with in predict.py
        # it will predict the top list of flowers with its probablities of correct prediction
        probs, classes = predict(test_dir+image, model, arg.device_type, arg.top_k)
        # Define show_predictyion function with in predict.py
        # it will show the prediction with respect to the file provided.
        show_prediction(probs, classes, arg.category_names)
        
    else:
        print('Oops, checkpoint does NOT exist! ({}) please provide correct valid checkpoint'.format(arg.save_dir))
   
    return 
    

# let's run this thing
if __name__ == '__main__':
    main()




