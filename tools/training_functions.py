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


def load_saved_games(N_data_points, board_size, subdir, only_last_epochs=False):
    # Find subdirectory to load games from
    subdirectory = "games_data/" + subdir + "/"

    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory + "*.npz"):
        files.append(file)

    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "", file)))
    # Get max number
    latest_games = max(number_list)

    # Case where only last epoch data is wanted
    if only_last_epochs:
        file_name = subdirectory + "game_data_" + str(latest_games) + ".npz"
        data = np.load(file_name)
        S = data['S']
        Pi = data['P']
        z = data['z']
        return S, Pi, z, S.shape[0]

    # Construct large numpy array of data
    S_array = np.empty((N_data_points, 17, board_size, board_size), dtype=bool)
    Pi_array = np.empty((N_data_points, board_size ** 2 + 1), dtype=float)
    z_array = np.zeros((N_data_points), dtype=float)
    # Counter for keeping track of large array
    data_counter = 0
    while (data_counter < N_data_points):
        # Load data
        file_name = subdirectory + "game_data_" + str(latest_games) + ".npz"
        latest_games -= 1
        # Case where not enough data is generated
        if (latest_games < 0):
            S_array = S_array[0:data_counter]
            Pi_array = Pi_array[0:data_counter]
            z_array = z_array[0:data_counter]

            return S_array, Pi_array, z_array, data_counter + 1

        data = np.load(file_name)
        S = data['S']
        Pi = data['P']
        z = data['z']
        for i in range(z.shape[0]):
            # Add data to large arrays
            S_array[data_counter] = S[i]
            Pi_array[data_counter] = Pi[i]
            z_array[data_counter] = z[i]
            # Increment counter
            data_counter += 1
            # Check if large arrays are filled
            if (data_counter >= N_data_points):
                break

    return S_array, Pi_array, z_array, data_counter + 1


def save_model(model):
    subdirectory = "model/saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory + "*.model"):
        files.append(file)

    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "", file)))

    if (number_list == []):
        number_list = [0]
    # Get max number
    latest_new_model = max(number_list) + 1

    save_name = subdirectory + "model_" + str(latest_new_model) + ".model"
    torch.save(model, save_name)


def load_latest_model():
    subdirectory = "model/saved_models/"
    os.makedirs(subdirectory, exist_ok=True)
    # Find larges number of games found
    # Get files
    files = []
    for file in glob.glob(subdirectory + "*.model"):
        files.append(file)

    # get numbers from files
    number_list = []
    for file in files:
        number_list.append(int(re.sub("[^0-9]", "", file)))

    if (number_list == []):
        warnings.warn("No model was found in path " + subdirectory, RuntimeWarning)
        return None
    # Get max number
    latest_model = max(number_list)

    load_name = subdirectory + "model_" + str(latest_model) + ".model"
    print("Loading model " + load_name)
    sys.path.append("model")
    if (torch.cuda.is_available()):
        model = torch.load(load_name)
    else:
        model = torch.load(load_name, map_location=torch.device('cpu'))
    return model


def loss_function(Pi, z, P, v, batch_size, board_size):
    v = torch.squeeze(v)
    value_error = torch.mean((v - z) ** 2)
    inner = torch.log(1e-8 + P)
    policy_error = torch.bmm(Pi.view(batch_size, 1, board_size ** 2 + 1),
                             inner.view(batch_size, board_size ** 2 + 1, 1)).mean()
    total_error = value_error - policy_error
    return total_error, value_error, -policy_error


class model_trainer:
    def __init__(self, writer, MCTS_settings, training_settings, N_turns=5*10 ** 5):
        self.writer = writer
        self.MCTS_settings = MCTS_settings
        self.criterion = loss_function
        self.N_turns = N_turns
        self.num_epochs = training_settings["max_training_epochs"]
        self.first_training_epoch = training_settings["first_training_epoch"]
        self.train_batch_size = training_settings["train_batch_size"]
        self.board_size = self.MCTS_settings["board_size"]
        self.rotate = training_settings["use_rotation"]
        self.patience = training_settings["patience"]
        self.use_early_stopping = training_settings["use_early_stopping"]
        self.training_counter = 0

        # Check for cuda
        # GPU things
        self.cuda = torch.cuda.is_available()

    def convert_torch(self, S, Pi, z):
        S = torch.from_numpy(S).float()
        Pi = torch.from_numpy(Pi).float()
        z = torch.from_numpy(z).float()
        # Convert to cuda if GPU support
        if self.cuda:
            S = S.cuda()
            Pi = Pi.cuda()
            z = z.cuda()

        return S, Pi, z

    def gen_epoch(self, S, Pi, z, batch_size):
        # Note this does not permutate samples!
        N_samples = S.shape[0]
        samples = []
        n_batches = int(np.ceil(N_samples / batch_size))

        index = 0
        for i in range(n_batches):
            end_index = index + batch_size
            S_batch = S[index:end_index]
            Pi_batch = Pi[index:end_index]
            z_batch = z[index:end_index]
            samples.append([S_batch, Pi_batch, z_batch])
            index += batch_size
        return samples

    def rotate_batch(self, Pi_batch, S_batch):
        # First convert back to 2D board
        Pi_actions = torch.reshape(Pi_batch[:, 0:(self.board_size ** 2)],
                                   (-1, self.board_size, self.board_size))
        rotations = torch.randint(0, 4, size=(self.train_batch_size,))
        # This works by sampling number of different rotation types, and splitting the tensor
        zeroes = torch.sum(rotations == 0)
        ones = torch.sum(rotations == 1)
        twos = torch.sum(rotations == 2)
        threes = torch.sum(rotations == 3)
        rots = [zeroes, ones, twos, threes]

        Pi_chunks = list(torch.split(Pi_actions, rots))
        Pi_chunks[1] = torch.rot90(Pi_chunks[1], 1, dims=(1, 2))
        Pi_chunks[2] = torch.rot90(Pi_chunks[2], 1, dims=(1, 2))
        Pi_chunks[3] = torch.rot90(Pi_chunks[3], 1, dims=(1, 2))

        S_chunks = list(torch.split(S_batch, rots))
        S_chunks[1] = torch.rot90(S_chunks[1], 1, dims=(2, 3))
        S_chunks[2] = torch.rot90(S_chunks[2], 1, dims=(2, 3))
        S_chunks[3] = torch.rot90(S_chunks[3], 1, dims=(2, 3))

        Pi_actions = torch.cat(Pi_chunks, dim=0)
        S_batch = torch.cat(S_chunks, dim=0)

        # Now repeat same trick with reflection
        # First generate index for reflection (this is also a binomial sampling)

        # Rotation will be seperated into 4 chuncks as torch.rot90 does not support different rotations in sample sample
        reflects = torch.randint(0, 2, size=(self.train_batch_size,)) == 1
        Pi_actions[reflects] = torch.flip(Pi_actions[reflects], dims=(1,))
        S_batch[reflects] = torch.flip(S_batch[reflects], dims=(2,))

        # Put the board- and pass actions back together
        Pi_actions = Pi_actions.reshape(-1, self.board_size ** 2)
        Pi_batch = torch.cat([Pi_actions, Pi_batch[:, self.board_size ** 2, None]], dim=1)

        return Pi_batch, S_batch

    def train(self, training_model):
        # Select learning rate
        if (self.training_counter < 10 ** 5):
            learning_rate = 0.25 * 10 ** -2
        elif (self.training_counter < 1.5 * 10 ** 5):
            learning_rate = 0.25 * 10 ** -3
        else:
            learning_rate = 0.25 * 10 ** -4
        best_model = training_model # Initialise the best model as the training model
        training_model.train()
        optimizer = optim.SGD(training_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=10 ** -4)
        # Load newest data
        S, Pi, z, n_points = load_saved_games(self.N_turns, self.board_size, "train")
        S, Pi, z = self.convert_torch(S, Pi, z)

        last_lowest_loss = torch.tensor([float("Inf")])
        increasing_loss_streak = 0

        length_training = self.first_training_epoch if (self.training_counter == 0) else self.num_epochs
        # Train
        for i in range(length_training):
            self.training_counter += 1

            # Generate batch. Note we uniform sample instead of epoch as in the original paper
            index = np.random.randint(0, n_points - 1, size=self.train_batch_size)
            Pi_batch = Pi[index]
            z_batch = z[index]
            S_batch = S[index]

            if self.rotate:
                # Case where rotation is used as data argmentation
                with torch.no_grad():
                    Pi_batch, S_batch = self.rotate_batch(Pi_batch, S_batch)

            # Optimize
            optimizer.zero_grad()
            P_batch, v_batch = training_model.forward(S_batch)
            loss, v_loss, P_loss = self.criterion(Pi_batch, z_batch, P_batch, v_batch, self.train_batch_size,
                                                  self.board_size)
            loss.backward()
            optimizer.step()

            self.writer.add_histogram('Output/v', v_batch, self.training_counter)
            self.writer.add_histogram('Output/P', P_batch, self.training_counter)
            self.writer.add_histogram('Output/Pi', Pi_batch, self.training_counter)
            self.writer.add_scalar('Total_loss/train', loss, self.training_counter)
            self.writer.add_scalar('value_loss/train', v_loss, self.training_counter)
            self.writer.add_scalar('Policy_loss/train', P_loss, self.training_counter)

            if ((i % 10 == 0) and (self.training_counter >= self.num_epochs)):
                # number of epochs should be above one loop, so validation data exists

                print("Fraction of training done: ", i / self.num_epochs)

                # Run validation
                S_val, Pi_val, z_val, n_points = load_saved_games(self.N_turns, self.board_size, "validation",
                                                                  only_last_epochs=True)
                S_val, Pi_val, z_val = self.convert_torch(S_val, Pi_val, z_val)
                epoch_samples = self.gen_epoch(S_val, Pi_val, z_val, self.train_batch_size)
                loss_list, v_loss_list, P_loss_list = [], [], []
                training_model.eval()

                with torch.no_grad():
                    for S_batch, Pi_batch, z_batch in epoch_samples:
                        P_batch, v_batch = training_model.forward(S_batch)
                        loss, v_loss, P_loss = self.criterion(Pi_batch, z_batch, P_batch, v_batch,
                                                              S_batch.shape[0],
                                                              self.board_size)
                        loss_list.append(loss)
                        v_loss_list.append(v_loss)
                        P_loss_list.append(P_loss)

                    loss_val = torch.mean(torch.tensor(loss_list))
                    v_loss_val = torch.mean(torch.tensor(v_loss_list))
                    P_loss_val = torch.mean(torch.tensor(P_loss_list))

                self.writer.add_scalar('Total_loss/validation', loss_val, self.training_counter)
                self.writer.add_scalar('value_loss/validation', v_loss_val, self.training_counter)
                self.writer.add_scalar('Policy_loss/validation', P_loss_val, self.training_counter)

                print("Lowest loss value: " + str(last_lowest_loss))
                print("Loss value: " + str(loss_val))
                if last_lowest_loss >= loss_val:
                    last_lowest_loss = loss_val
                    increasing_loss_streak = 0
                    best_model = training_model.deep_copy() # New best model
                else:
                    increasing_loss_streak += 1
                if (increasing_loss_streak >= self.patience) and self.use_early_stopping:
                    # Case where no improvement of validation loss is observed for self.patience iterations
                    print("breaking at: " + str(i))
                    break
        self.writer.add_scalar('Training_length', i, self.training_counter)
        return best_model

