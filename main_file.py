#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:44:33 2019

@author: tirsgaard
"""
# File for running the entire program

from model.go_model import ResNet, ResNetBasicBlock
from MCTS.MCTS2 import sim_games
from tools.training_functions import save_model, load_latest_model, model_trainer

import torch
from torch.utils.tensorboard import SummaryWriter
from tools.elo import elo_league

if __name__ == '__main__':
    def resnet40(in_channels, filter_size=128, board_size=9, deepths=[19]):
        return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)


    ## Hyper parameters
    number_of_threads = 16 # Number of threads / games of go to run in parallel
    n_parallel_explorations = 4 # Number of pseudo-parrallel runs of the MCTS, note >16 reduces accuracy significantly
    N_MCTS_sim = 100 # Number of MCTS simulations for each move
    board_size = 5 # Board size of go
    N_training_games = 200 # Number of games to run each
    #MCTS_queue = 8

    N_duel_games = 100 # Number of games to play each duel
    N_turns = 500000
    train_batch_size = 512
    num_epochs = 320
    elo_league = elo_league()

    MCTS_settings = {"number_of_threads" : number_of_threads,
                         "n_parallel_explorations" : n_parallel_explorations,
                        "board_size" : board_size,
                        "N_training_games" : N_training_games,
                        "N_MCTS_sim" : N_MCTS_sim}

    # GPU things
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    ## Load model if one exists, else define a new
    writer = SummaryWriter()
    best_model = load_latest_model()
    training_model = load_latest_model()

    if (best_model == None):
        best_model = resnet40(17, 128, board_size=board_size)
        save_model(best_model)
        training_model = load_latest_model()

    if cuda:
        best_model.cuda()
        training_model.cuda()

    trainer = model_trainer(writer)

    ## define variables to be used
    v_resign = float("-inf")
    loop_counter = 1
    training_counter = 0

    # This is in the case of early termination
    skip_self_play = False
    skip_training = False

    ## Running loop
    while True:
        if ((not skip_self_play) | (training_counter > 0)):
            print("Beginning loop", loop_counter)
            print("Beginning self-play")
            ## Generate new data for training
            with torch.no_grad():
                v_resign = sim_games(N_training_games,
                                     best_model,
                                     v_resign,
                                     MCTS_settings)

            writer.add_scalar('v_resign', v_resign, loop_counter)

        print("Begin training")
        ## Now train model
        trainer.train(training_model)

        print("Begin evaluation")
        ## Evaluate training model against best model
        # Below are the needed functions
        best_model.eval()
        training_model.eval()
        with torch.no_grad():
            scores = sim_games(N_duel_games,
                               training_model,
                               v_resign,
                               MCTS_settings,
                               model2=best_model,
                               duel=True)

        # Find best model
        # Here the model has to win atleast 60% of the games

        best_model = training_model
        save_model(best_model)

        new_elo, model_iter_counter = elo_league.common_duel_elo(scores[0] / (scores[1]+scores[0]))

        # Store statistics
        writer.add_scalar('Elo', new_elo, model_iter_counter)
        loop_counter += 1
