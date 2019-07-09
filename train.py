#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/ImageClassifier/train.py
#
# TODO 0: Add your information below for Programmer & Date Created.                                                                             
# PROGRAMMER: Karry Harsh
# DATE CREATED: 30/01/2019
# REVISED DATE:

# Imports python modules
from time import time, sleep
import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os

def args_paser():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. dataset directory as --data_dir with default value 'flowers'
      2. Processing Device as --gpu with default value True
      3. Learning rate as --lr with default value 0.001
      4. Epochs as --epoch with default value 10
      5. CNN architecture as --arch with default value 'vgg16'
      6. Hidden units for layer as --hidden_units with default value 512
      7. Save model as --sav_dir with deault value 'checkpoint.pth'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    paser = argparse.ArgumentParser(description='trainer file')
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    paser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=str, default='gpu', help='Run model on CPU or GPU')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    paser.add_argument('--arch', type=str, default='vgg16', help='architecture')
    paser.add_argument('--hidden_units', type=int, default=[512,128,102], help='hidden units for layer')
    paser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
    args = paser.parse_args()
    return args

def process_data(train_dir, test_dir, valid_dir):
    '''
    The dataset is split into three parts, training, validation, and testing. For the training,
    apply transformations such as random scaling, cropping, and flipping. This will help the network generalize
    leading to better performance.the input data is resized to 224x224 pixels as required by the pre-trained networks.

    The validation and testing sets are used to measure the model's performance on data it hasn't seen yet.
    For this don't want any scaling or rotation transformations, but need to resize then crop the images to the appropriate size.

    The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately.
    For all three sets need to normalize the means and standard deviations of the images to what the network expects.
    For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225],
    calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.
    '''
    # Define transforms for the training data, testing data and validating data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    
    return trainloaders, testloaders, validloaders

def basic_model(arch):
    '''
    load the pretraind CNN arch provided by the user or if not use the default arch defined in args_praser()
    '''
    # Load pretrained_network
    if arch == None or arch == 'vgg':
        load_model = models.vgg16(pretrained=True)
        print('Use vgg16')
    else:
        print('Please vgg16 or desnenet only, defaulting to vgg16')
        load_model = models.vgg16(pretrained=True)
           
    return load_model

def set_classifier(model, hidden_units):
    '''
    set_classifier function will re-define the classifier of a pretrained model with new hidden
    layers provided by the user 
    '''
    if hidden_units == None:
        hidden_units = [512,128,102]
    input = model.classifier[0].in_features
    # Redefining the classifier with new hidden layers
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input, hidden_units[0], bias=True)),
                                                  ('relu1', nn.ReLU()),
                                                  ('dropout', nn.Dropout(p=0.5)),
                                                  ('fc2', nn.Linear(hidden_units[0], hidden_units[1], bias=True)),
                                                  ('relu2', nn.ReLU()),
                                                  ('dropout', nn.Dropout(p=0.5)),
                                                  ('fc3', nn.Linear(hidden_units[1], hidden_units[2], bias=True)),
                                                  ('output', nn.LogSoftmax(dim=1))
                                                  ]))
    model.classifier= classifier
    return model
#print the epoch and training loss with respect to the training of model
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/print_every:.3f}.. ")
    
def train_model(epochs, trainloaders, validloaders,gpu,model,optimizer,criterion):
    '''
    Training the new model with set of Image dataset, with GPU to do the calculation is will 
    uses CUDA to efficiently compute the forward and backwards passes on the GPU. epoch is used
    to calculate the training loss, valid loss and valid accuracy of the model after training
    '''
    if type(epochs) == type(None):
        epochs = 10
        print("Epochs = 10")

    if (gpu == 'gpu'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    steps = 0
    model.to(device)
    running_loss = 0
    print_every = 60
    # Calculating running loss
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
#            if gpu==True:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Calculating test loss
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval() # inference mode
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {test_loss/len(validloaders):.3f}.."
                          f"Valid accuracy: {accuracy/len(validloaders):.3f}")
                    running_loss = 0
                model.train()
    return model

def valid_model(Model, testloaders, gpu):
    if (gpu == 'gpu'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_loss = 0
    accuracy = 0
    Model.eval()
    with torch.no_grad():
        for inputs, labels in testloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = Model.forward(inputs)
            criterion = nn.NLLLoss()
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
def save_checkpoint(model, train_datasets, save_dir):
    '''
    save_checkpoint function is used to rebuild the model exactly as it was when trained.
    Information about the model architecture needs to be saved in the checkpoint, along with the state dict.
    build a dictionary with all the information you need to compeletely rebuild the model.
    '''
    # mapping of classes to indices from one of the image datasets
    model.class_to_idx = train_datasets.class_to_idx
    # Define a Dictionary to save all the required information to rebuild the model 
    checkpoint = {'arch_name': 'vgg16',
                'classifier': model.classifier,
                'model_class_index': model.class_to_idx,
                'model_state': model.state_dict()}
    return torch.save(checkpoint, save_dir)

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    sleep(10)
    # Define args_paser function within the file train.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable args
    args = args_paser()
    # Define path to the image in variable train_dir, valid_dir, test_dir
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    print('')
    print('data dir: {}'.format(data_dir))
    print('train dir: {}'.format(train_dir))
    print('valid dir: {}'.format(valid_dir))
    print('Basic model arch: {}'.format(args.arch))
    print('hidden layers: {}'.format(args.hidden_units))
    print('Epochs: {}'.format(args.epochs))
    print('Learning Rate: {}'.format(args.lr))
    print('Mode of device: {}'.format(args.gpu))
    print('Path to save model: {}'.format(args.save_dir))
    
    
    # Define process_data function within train.py
    # it pre process the set of image for testing, training and validating
    trainloaders, testloaders, validloaders = process_data(train_dir, test_dir, valid_dir)
    
    # Define basic_model function with in train.py
    # Load the pretrained model
    model = basic_model(args.arch)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define set_classifier function with in train.py
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout 
    model = set_classifier(model,args.hidden_units)
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    # Define train_model function with in train.py
    # It is created to train the set of Images with with new defined Classifier in GPU
    trained_model = train_model(args.epochs,trainloaders, validloaders, args.gpu,model,optimizer,criterion)
    
    # Define valid_model function with in train.py
    # It is created to validate the newly trained model with a new 
    # set of images to estimate model performance
    valid_model(trained_model, testloaders, args.gpu)
    
    # Pre process the set of Images.
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    
    # Define save_checkpoint function with in train.py
    # Save the model so you can load it later for making predictions
    save_checkpoint(trained_model, train_datasets, args.save_dir)
    print('')
    print('Training the model is Completed and saved in a  checkpoint path!')
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Call to main function to run the program    
if __name__ == '__main__': main()
 


