#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:40:57 2019

@author: tirsgaard
"""
import re
import os
import glob
import numpy as np
import torch
import sys
import torch.optim as optim
import warnings

sys.path.append("../model")
from model import go_model

def load_saved_games(N_data_points, board_size):
    # Construct large numpy array of data
    S_array = np.empty((N_data_points, 17, board_size, board_size), dtype=bool)
    Pi_array = np.empty((N_data_points, board_size**2+1), dtype=float)
    z_array = np.zeros((N_data_points), dtype=float)
    
    subdirectory = "games_data/"
    
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory+"*.npz"):
        files.append(file)
        
    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "",file)))
    # Get max number
    latest_games = max(number_list)
    
    # Counter for keeping track of large array
    data_counter = 0
    while (data_counter<N_data_points):
        # Load data
        file_name = subdirectory+"game_data_"+str(latest_games)+".npz"
        latest_games -= 1
        # Case where not enough data is generated
        if (latest_games<0):
            S_array = S_array[0:data_counter]
            Pi_array = Pi_array[0:data_counter]
            z_array = z_array[0:data_counter]
            
            return S_array, Pi_array, z_array, data_counter+1

        data = np.load(file_name)
        S = data['S']
        Pi = data['P']
        z = data['z']
        for i in range(z.shape[0]):
            # Add data to large arrays
            S_array[data_counter] = S[i]
            Pi_array[data_counter] = Pi[i]
            z_array [data_counter] = z[i] 
            # Increment counter
            data_counter += 1
            # Check if large arrays are filled
            if (data_counter>=N_data_points):
                break
    
    return S_array, Pi_array, z_array, data_counter+1

def save_model(model):
    subdirectory = "model/saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory+"*.model"):
        files.append(file)
        
    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "",file)))
        
    if (number_list==[]):
        number_list = [0]
    # Get max number
    latest_new_model = max(number_list)+1
    
    save_name = subdirectory + "model_" + str(latest_new_model) + ".model"
    torch.save(model, save_name)
    

def load_latest_model():
    subdirectory = "model/saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory+"*.model"):
        files.append(file)
        
    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "",file)))
    
    if (number_list==[]):
        warnings.warn("No model was found in path " + subdirectory, RuntimeWarning)
        return None
    # Get max number
    latest_model = max(number_list)
    
    load_name = subdirectory+"model_" + str(latest_model) + ".model"
    print("Loading model " + load_name)
    sys.path.append("model")
    if (torch.cuda.is_available()):
        model = torch.load(load_name)
    else:
        model = torch.load(load_name, map_location=torch.device('cpu'))
    return model

def loss_function(Pi, z, P, v, batch_size, board_size):
    v = torch.squeeze(v)
    value_error = torch.mean((v - z)**2)
    inner = torch.log(1e-8 + P)
    policy_error = torch.bmm(Pi.view(batch_size, 1, board_size**2+1), inner.view(batch_size, board_size**2+1, 1)).mean()
    total_error = value_error - policy_error
    return total_error, value_error, -policy_error

class model_trainer:
    def __init__(self, writer, MCTS_settings, N_turns=5*10**5, num_epochs = 320, train_batch_size = 512):
        self.writer = writer
        self.MCTS_settings = MCTS_settings
        self.criterion = loss_function
        self.N_turns = N_turns
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.board_size = self.MCTS_settings["board_size"]
        self.training_counter = 0

        # Check for cuda
        # GPU things
        self.cuda = torch.cuda.is_available()


    def train(self, training_model):
        # Select learning rate
        if (self.training_counter < 10 ** 5):
            learning_rate = 0.25 * 10 ** -2
        elif (self.training_counter < 1.5 * 10 ** 5):
            learning_rate = 0.25 * 10 ** -3
        else:
            learning_rate = 0.25 * 10 ** -4

        training_model.train()
        optimizer = optim.SGD(training_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=10**-4)
        # Load newest data
        S, Pi, z, n_points = load_saved_games(self.N_turns, self.board_size)
        S = torch.from_numpy(S).float()
        Pi = torch.from_numpy(Pi).float()
        z = torch.from_numpy(z).float()
        # Convert to cuda if GPU support
        if self.cuda:
            S = S.cuda()
            Pi = Pi.cuda()
            z = z.cuda()

        # Train
        for i in range(self.num_epochs):
            self.training_counter += 1

            # generate batch
            index = np.random.randint(0, n_points - 1, size=self.train_batch_size)
            Pi_batch = Pi[index]
            z_batch = z[index]
            S_batch = S[index]

            # Optimize
            optimizer.zero_grad()
            P_batch, v_batch = training_model.forward(S_batch)
            loss, v_loss, P_loss = self.criterion(Pi_batch, z_batch, P_batch, v_batch, self.train_batch_size, self.board_size)
            loss.backward()
            optimizer.step()

            self.writer.add_histogram('Output/v', v_batch, self.training_counter)
            self.writer.add_histogram('Output/P', P_batch, self.training_counter)
            self.writer.add_histogram('Output/Pi', Pi_batch, self.training_counter)
            self.writer.add_scalar('Total_loss/train', loss, self.training_counter)
            self.writer.add_scalar('value_loss/train', v_loss, self.training_counter)
            self.writer.add_scalar('Policy_loss/train', P_loss, self.training_counter)

            if (i % 100 == 0):
                print("Fraction of training done: ", i / self.num_epochs)

