#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:51:02 2019

@author: tirsgaard
"""

import sys
# Relative paths for some reason do not work :(
sys.path.append('/Users/tirsgaard/Google Drive/Alphago_zero')
from model.go_model import ResNet, ResNetBasicBlock
from tools.go import go_board
from multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
from MCTS.MCTS2 import gpu_worker, data_handler, sim_game
from tools.training_functions import load_latest_model

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from functools import partial




def resnet40(in_channels, filter_size=128, board_size=9, deepths=[19]):
    return ResNet(in_channels, filter_size, board_size, block=ResNetBasicBlock, deepths=deepths)

if __name__ == "__main__":
    #model = resnet40(17, 128,9)
    model = load_latest_model()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    N_MCTS_queue = 2
    
    # Make queues for sending data
    gpu_Q = Queue()
    data_Q = Queue()
    # Also make pipe for receiving v_resign
    conn_rec, conn_send = Pipe(False)

    with torch.no_grad():
        # Make process for gpu worker and data_handler
        processes = []
        p = Process(target=gpu_worker, args=(gpu_Q, 2, 9, model))
        processes.append(p)
        p = Process(target=data_handler, args=(data_Q, 1, conn_send))
        processes.append(p)
        # Start gpu and data_loader worker
        for p in processes:
            p.start()
        
        sim_game(gpu_Q, 400, data_Q, -float("inf"), N_MCTS_queue)
    
    # Close processes
    av_resign = conn_rec.recv()
    processes[1].join()
    processes[0].terminate()
