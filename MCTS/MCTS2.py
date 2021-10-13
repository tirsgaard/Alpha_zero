import numpy as np
from collections import deque, Counter
from torch.multiprocessing import Process, Queue, Pipe, Value, Lock, Manager, Pool
import time
import re
import os
import glob
from tqdm import tqdm
import sys

# Import torch things
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from functools import partial


from model.go_model import ResNet, ResNetBasicBlock
from tools.go import go_board

class state_node:
    def __init__(self, go_state, P, color):
        board_size = 9
        self.go_state = go_state
        self.N_total = 0
        self.N = np.zeros((board_size*board_size+1))
        self.N_inv = np.ones((board_size*board_size+1))
        self.Q = np.zeros((board_size*board_size+1))
        self.W = np.zeros((board_size*board_size+1))
        self.illigal_board = None
        self.U = P
        self.P = P
        self.turn_color = color
        self.action_edges = {}
        self.game_over = False

class end_node:
    def __init__(self, z):
        self.game_over = True
        self.z = z
    
def gpu_worker(gpu_Q, job_size ,board_size, model):
    with torch.no_grad():
        
        cuda = torch.cuda.is_available()
        num_eval = 0
        pipe_queue = deque([])
        if cuda:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
        pipe_queue = deque([])

        # Calibrate queue size
        calibration_done = False
        MCTS_queue = 1
        t = time.time()
        # The +1 next line is to take cases where a small job is submitted
        batch = torch.empty((job_size*(MCTS_queue+1), 17, board_size, board_size))
        batch_test_length = 1000
        speed = float("-inf")

        while True:
            # Loop to get data
            i = 0
            jobs_indexes = []
            queue_size = job_size*MCTS_queue
            while(i<queue_size):
                try:
                    gpu_jobs, pipe = gpu_Q.get(True, 0.0001) # Get jobs
                    pipe_queue.append(pipe) # Track job index with pipe index
                    num_jobs = gpu_jobs.shape[0] # Get number of jobs send
                    jobs_indexes.append(num_jobs) # Track number of jobs send
                    batch[i:(i+num_jobs)] = torch.from_numpy(gpu_jobs) # Add jobs to batch
                    num_eval += num_jobs # Increase number of evaluations for speed check
                    i += num_jobs
                    
                except:
                    if (i!=0):
                        break
            # Evaluate
            result = model.forward(batch[0:i])
            P = result[0].cpu().numpy()
            v = result[1].cpu().numpy()
            index_start = 0
            for jobs_index in jobs_indexes:
                index_end = index_start+jobs_index
                pipe_queue.popleft().send([P[index_start:index_end], v[index_start:index_end]])
                index_start = index_end


            if ((num_eval % (batch_test_length + 1)) == 0) and (not calibration_done):
                # Update calibration
                time_spent = time.time() - t
                t = time.time()
                calibration_done = (num_eval / time_spent) < speed
                speed = num_eval / time_spent

                MCTS_queue += 1 - 2*calibration_done # The calibration is for going back to optimal size
                batch = torch.empty((job_size * (MCTS_queue+1), 17, board_size, board_size))


                f = open("speed.txt", "a")
                print("Queue size:" + str(MCTS_queue-1), file=f)
                print("Number of games pr. week:" + str(7 * 24 * 3600 * num_eval / (time_spent * 80 * 400)), file=f)
                print("Time pr. eval: " + str(time_spent / num_eval), file=f)
                f.close()
                num_eval = 0
    
    
def rotate_S(S):
    # Function for rotating and reflecting a 17*board_size*board_size array
    # The chance of any rotation and reflection is equally likely
    rotation = np.random.randint(4)
    reflection = np.random.randint(2)
    
    # Potetinal rotation
    S = np.rot90(S, rotation, axes=(1,2))
    # Reflection
    if (reflection==1):
        S = np.flip(S, axis=(2))
    return S, rotation, reflection

def reverse_rotate(P, rotation, reflection, board_shape):
    # The indexing is because of the posibility of a pass action
    total_board_size = board_shape*board_shape
    temp_P = np.reshape(P[0:total_board_size],(board_shape, board_shape))
    if (reflection==1):
        temp_P = np.flip(temp_P, axis=(1))
        
    # Reverse rotation 
    P[0: total_board_size] = np.ndarray.flatten(np.rot90(temp_P, rotation, axes=(1, 0)))
    return P
    
    
def MCTS(root_node, gpu_Q, n_MCTS, color, number_passes, MCTS_queue):
    n_vl = 3 #Virtual loss count
    board_size = root_node.go_state.board_size
    n_positions = board_size*board_size
    start_color = color
    turn_switcher = {"black": "white",
                     "white": "black"}
    # Switch for adding calculated v relative to black
    relative_value =  {"black": 1,
                     "white": -1}
    # Define pipe for GPU process
    conn_rec, conn_send = Pipe(False)
    # Define variables to be used
    legal_board = np.empty((n_positions+1), dtype=float)
    
    # Begin simulations
    i = 0
    while i<n_MCTS: # Keep number of simulations below the set N
        stored_jobs = []
        empty = False
        for j in range(MCTS_queue):
            current_path = deque([])
            current_node = root_node
            color = start_color
            
            # Code for breaking simuntanius searches if 
            if ( (empty == True ) & (stored_jobs==[]) ):
                break

            while True: # Continue to traverse tree untill new edge is found
                current_node.N_total += 1
                # Choose action
                current_node.U = (5*np.sqrt(current_node.N_total))*current_node.P*current_node.N_inv
                
                # Depending on current color Q_values are multiplied by -1
                if (color=="black"):
                    a_chosen = np.argmax(current_node.U+current_node.illigal_board+current_node.Q)
                else:
                    a_chosen = np.argmax(current_node.U+current_node.illigal_board-current_node.Q)
                
                current_node.N_total += n_vl-1 # Add virtual loss
                
                # Add action and node to path and change color
                # Add current node to path
                current_path.append((current_node, a_chosen, color))
                
                ## Update stored values using virtual loss
                current_node.W[a_chosen] = -n_vl*relative_value[color]
                current_node.Q[a_chosen] = (current_node.W[a_chosen])/(current_node.N[a_chosen]+n_vl)
            
                #### Now coontinue based if edges are explored or not
                if (current_node.N[a_chosen]!=0): # Case for explored edge
                    # Update current node, color of turn, and repeat
                    color = turn_switcher[color]
                    current_node.N[a_chosen] += n_vl
                    current_node.N_inv[a_chosen] = 1/(current_node.N[a_chosen]+1)
                    
                    # Game is not done and search can progress
                    try:
                        current_node = current_node.action_edges[a_chosen]
                        
                        if (current_node.game_over): # A done game is encountered
                            v = current_node.z
                            for node, action, color in current_path:
                                    node.N_total += -n_vl+1
                                    node.N[action] += -n_vl+1
                                    node.N_inv[action] = 1/(node.N[action]+1)
                                    node.W[action] += v + n_vl*relative_value[color]
                                    node.Q[action] = node.W[action]/node.N[action]
                                
                            empty = True
                            break
                    except:
                        # Case where same new edge is explored during parallel
                        # Undo virtual loss
                        for node, action, color in current_path:
                                node.N_total -= n_vl
                                node.N[action] -= n_vl
                                node.N_inv[action] = 1/(node.N[action]+1)
                                node.W[action] += n_vl*relative_value[color]
                                node.Q[action] = node.W[action]/node.N[action]
                        
                        empty = True
                        break
                        
                    continue
                
                else: # Not explored case
                    # Virtual loss update could be skipped for the last node
                    #   but it makes the code simpler to include it
                    current_node.N[a_chosen] += n_vl
                    current_node.N_inv[a_chosen] = 1/(current_node.N[a_chosen]+1)
                
                    # Prepare new go game to store
                    new_go_state = current_node.go_state.copy_game()
                    
                    
                    if (a_chosen!=(board_size*board_size)):
                        the_move = np.unravel_index(a_chosen, (board_size, board_size))
                        new_go_state.move(the_move, color)
                            
                    else:
                        game_done = False
                        if (number_passes==1) & (len(current_path)==1):
                            # Case where last action in game was a pass
                            game_done = True
                        else: 
                            # Normal rule of each game done after two passes in a row
                            try:
                                game_done = (n_positions==current_path[-1][1]==current_path[-2][1])
                            except:
                                game_done = False
                        
                        # Count backwards if game has ended
                        if not game_done:
                            # Take pass move
                            new_go_state.move('pass', color)

                        else:
                            ## Game has ended
                            
                            # Calculate who won from stored values or directly
                            # Compute who won
                            counted_points = new_go_state.count_points()
                            v = (counted_points>0)-int(counted_points<0)
                            
                            # Construct new game over edge 
                            new_node = end_node(v)
                            current_node.action_edges[n_positions] = new_node
                            
                            # Backup
                            v = current_node.W[n_positions]/(current_node.N[n_positions]-1+n_vl)
                            for node, action, color in current_path:
                                node.N_total += -n_vl+1
                                node.N[action] += -n_vl+1
                                # Skip inverse since it will be updated later
                                node.N_inv[action] = 1/(node.N[action]+1)
                                node.W[action] += v + n_vl*relative_value[color]
                                node.Q[action] = node.W[action]/node.N[action]
                            
                            empty = True
                            break
                    
                    
                    
                    # Get state
                    color = turn_switcher[color]
                    S = new_go_state.get_state(color)
                    # Save values for later construction and backup of node
                    stored_jobs.append([S.copy(), color, new_go_state, current_path, current_node, a_chosen])
                    break
                
        if ( (empty == True ) & (stored_jobs==[]) ):
                empty = False
                break
        
        # Store values for updating
        rotation_list = []
        reflection_list = []
        S_array = np.empty((len(stored_jobs),17, board_size, board_size), dtype=bool)
        k = 0
        for S, color, new_go_state, current_path, current_node, a_chosen in stored_jobs:
            # Rotate and reflect state randomly
            S, rotation, reflection  = rotate_S(S)
            rotation_list.append(rotation)
            reflection_list.append(reflection)
            S_array[k] = S
            k += 1
        
        # Get policy and value
        gpu_Q.put([S_array, conn_send])
        P_array, v_array = conn_rec.recv() 
        
        # Expand tree and backtrack each node in queue
        k = 0
        for S, color, new_go_state, current_path, current_node, a_chosen in stored_jobs:
            i += 1 # Update i for each simulation
            P = P_array[k]
            v = v_array[k][0]
            v = relative_value[color]*v
            # Reverse rotation of P 
            P = reverse_rotate(P, rotation_list[k], reflection_list[k], board_size)

            # Rescale P based on legal moves
            legal_board[0:n_positions] = np.ndarray.flatten(new_go_state.get_legal_board(color))
            legal_board[n_positions] = 1
            P = np.multiply(P,legal_board)
            P = P/np.sum(P)
        
            # Generate new node
            new_node = state_node(new_go_state, P, color)
        
            # Make large n_roundsnegative penalty to stop choosing illigal moves
            new_node.illigal_board = (legal_board-1)*1000

            # Add new node to tree
            current_node.action_edges[a_chosen] = new_node
            
            # Now back up and remove virtual loss
            for node, action, color in current_path:
                node.N_total += -n_vl + 1
                node.W[action] += v + n_vl*relative_value[color]
                node.N[action] += -n_vl + 1
                node.N_inv[action] = 1/(node.N[action]+1)
                node.Q[action] = node.W[action]/node.N[action]
            
    return root_node
            
def data_handler(data_Q, num_games, conn_v):
    i = 0
    game_list = []
    total_moves = 0
    while i<num_games:
        # Get data
        data = data_Q.get(True)
        # Get number of moves in game
        total_moves += data[2].shape[0]
        game_list.append(data)
        i+= 1

    board_size = data[0].shape[2] # Infer board size from given data
    n_positions = board_size*board_size
    # Construct large numpy array of data
    S_array = np.empty((total_moves, 17, board_size, board_size), dtype=bool)
    P_array = np.empty((total_moves, n_positions+1), dtype=float)
    z_array = np.zeros((total_moves), dtype=float)
    
    v_list = []
    
    # Now loop list to fill in array
    i = 0
    for S, P , z, false_v in game_list:
        for j in range(z.shape[0]):
            # Store values for training
            S_array[i,:,:,:] = S[j]
            P_array[i,:] = P[j]
            z_array[i] = z[j]
            i += 1
        # Store values for calculating v_resign
        if (false_v!=None):
            v_list.append(false_v)
    
    
    v_array = np.array(v_list)
    v_array = np.sort(v_array)
    # Calculate resign value
    print(0.05*v_array.shape[0])
    new_index = np.ceil(0.05*v_array.shape[0])-1
    try:
        new_v_resign = v_array[int(new_index)]
    except:
        new_v_resign = -float("inf")
    print(new_v_resign)
    # Make directory if missing
    subdirectory = "games_data/"
    os.makedirs(subdirectory, exist_ok=True)
    
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
    try:
        new_iter = max(number_list)+1
    except:
        new_iter = 1
    
    # Save new data
    name = subdirectory+"game_data_" + str(new_iter)
    np.savez(name, S=S_array, P=P_array, z=z_array)
    # Load data with np.load(temp_name)
    conn_v.send(new_v_resign)
    
def sim_game_worker(gpu_Q, n_MCTS, data_Q, v_resign, n_games, lock, game_counter, seed, MCTS_queue=4, board_size=9):
    np.random.seed(seed)
    while True:
        with lock:
            game_counter.value += 1
            val = game_counter.value
        if not (val>n_games):
            #print("Beginning game :", val, " out of: ", n_games)
            sim_game(gpu_Q, n_MCTS, data_Q, v_resign, MCTS_queue, board_size)
        else:
            return

def sim_duel_game_worker(gpu_Q1, gpu_Q2, N, winner_Q, n_games, lock, game_counter, seed, MCTS_queue=4, board_size=9):
    np.random.seed(seed)
    while True:
        with lock:
            done = game_counter.value>n_games
            game_counter.value += 1
        if not done:
            duel_game(gpu_Q1, gpu_Q2, N, winner_Q, MCTS_queue, board_size)
        else:
            return

    

def sim_game(gpu_Q, n_MCTS, data_Q, v_resign, MCTS_queue=4, board_size=9):
    no_resign = np.random.rand(1)[0]>0.95
    
    # Hyperparameters
    temp_switch = 16  #Number of turns before other temperature measure is used
    eta_par = 0.03
    epsilon = 0.25
    
    # Switch for adding calculated v relative to black
    relative_value =  {"black": 1,
                     "white": -1}
    turn_switcher = {"black": "white",
                     "white": "black"}
    n_positions = board_size*board_size
    # Define pipe for GPU process
    conn_rec, conn_send = Pipe(False)
    
    # List for storing resignation values
    resign_list_black = []
    resign_list_white = []
    
    # Start game
    n_rounds = 0
    turn_color = "white"
    number_passes = 0
    data = []
    resign = False

    # Start game
    go_game = go_board(board_size)
    # Evalute first node
    S = np.expand_dims(go_game.get_state(turn_switcher[turn_color]), 0)
    # Get policy and value
    
    gpu_Q.put([S, conn_send])
    P, v = conn_rec.recv()
    # Remove padded dimensions
    P = np.squeeze(P)
    v = v[0, 0]
    
    # Generate start node
    root_node = state_node(go_game, P, turn_switcher[turn_color])
    root_node.illigal_board = np.zeros(n_positions+1)
    # Run next moves
    while True:
        n_rounds += 1
        turn_color = turn_switcher[turn_color]
        
        # Case where early temperature is used
        if (n_rounds<=temp_switch):
            # Simulate MCTS
            root_node = MCTS(root_node, gpu_Q , n_MCTS, turn_color, number_passes, MCTS_queue)

            # Compute legal policy
            pi_legal = root_node.N/root_node.N_total
            # Selecet action
            action = np.random.choice(n_positions+1, size=1, p=pi_legal)[0]
            
        # Case where later temperature is used
        else:
            # Get noise
            eta = np.random.dirichlet(np.ones(n_positions+1)*eta_par)
            root_node.P = (1-epsilon)*root_node.P+epsilon*eta
            
            # Simulate MCTS
            root_node = MCTS(root_node, gpu_Q, n_MCTS, turn_color, number_passes, MCTS_queue)
            
            # Compute legal actions visit count (needed for storing)
            pi_legal = root_node.N/root_node.N_total
            
            # Pick move
            action = np.argmax(root_node.N)
        
        # Save Data
        S = go_game.get_state(turn_color)
        data.append([S.copy(), pi_legal.copy(), turn_color])
        
        # Check for resignation
        if (turn_color=="black"):
            try:
                resign_req = max([np.max(root_node.action_edges[action].Q), root_node.Q[action]])
            except:
                resign_req = relative_value[turn_color]*root_node.Q[action]
        else:
            try:
                # To account for flipped v
                resign_req = -1*min([np.min(root_node.action_edges[action].Q), root_node.Q[action]])
            except:
                resign_req = relative_value[turn_color]*root_node.Q[action]
            
        # Add resign values for color
        if (turn_color=="black"):
            resign_list_black.append(resign_req)
        else:
            resign_list_white.append(resign_req)
        
        # Check if game ends
        if ((no_resign==False) & (resign_req<v_resign)):
            # resign
            resign = True
            break
        
        # Convert and take action
        #print("Move n. ",n_rounds, "New move was: ", action, "color was: ", turn_color)
        if (action==n_positions):
                go_game.move('pass', turn_color)
                number_passes += 1
        else:
                go_game.move(np.unravel_index(action, (board_size, board_size)), turn_color)
                number_passes = 0
                
        # Check if game is over or too long (9*9*2)
        if ((number_passes==2) | (n_rounds>162)):
            break
        # Pick move
        root_node = root_node.action_edges[action]
        
    # Game is over
    # Find winner
    if (resign==True):
        # Set winner depending on resigned color
        if (turn_color=="black"):
            z = {"black": -1,
                 "white": 1}
        else:
            z = {"black": 1,
                 "white": -1}
    else:
        # No resignation, 
        points = go_game.count_points()
        
        # Black is winner
        if (points>0):
            z = {"black": 1,
                 "white": -1}
        else:
            z = {"black": -1,
                 "white": 1}
        
    # Define data arrays
    S_array = np.empty((n_rounds, 17, board_size, board_size), dtype=bool)
    P_array = np.empty((n_rounds, n_positions+1), dtype=float)
    z_array = np.empty((n_rounds), dtype=int)
    
    # Loop over each move and fill in arrays
    i = 0
    for S, P , turn_color in data:
        S_array[i] = S
        P_array[i] = P
        z_array[i] = z[turn_color]
        i += 1
    # Send data
    
    # In case game was used to check for false positives, compute lowest value
    if (no_resign==True):
        if (z["black"]==1):
            false_positive = min(resign_list_black)
        else:
            false_positive = min(resign_list_white)
        # Send data
        data_Q.put([S_array, P_array, z_array, false_positive])
    else:
        data_Q.put([S_array, P_array, z_array, None])


def duel_game(gpu_Q1, gpu_Q2, N, winner_Q, MCTS_queue, board_size=9):
    def gen_node(gpu_Q, go_game, color, conn_rec, conn_send, game_beginning):
        # A function for generating a node of If no node of board state exists
        
        relative_value =  {"black": 1,
                           "white": -1}
        S = go_game.get_state(color)
        S, rotation, reflection = rotate_S(S)
        
        S = np.expand_dims(S,0)
        
        gpu_Q.put([S, conn_send])
        P, v = conn_rec.recv()
        
         # Remove padded dimensions
        P = np.squeeze(P)
        v = v[0,0]
        
        P = reverse_rotate(P, rotation, reflection, go_game.board_size)
        v = relative_value[color]*v
    
        # Add legal board attribute
        if (game_beginning):
            # Generate start node
            root_node = state_node(go_game, P, color)
            root_node.illigal_board = np.zeros(n_positions+1)
        else:
            # Ensure illigal moves are removed
            legal_board = np.empty((n_positions+1), dtype=float)
            legal_board[0:n_positions] = np.ndarray.flatten(go_game.get_legal_board(color))
            legal_board[n_positions] = 1
            P = np.multiply(P,legal_board)
            P = P/np.sum(P)
            
            # Generate start node
            root_node = state_node(go_game, P, color)
            # Make large negative penalty to stop choosing illigal moves
            root_node.illigal_board = (legal_board-1)*1000
            
        return root_node, v

    n_positions = board_size*board_size
    # Select player colors
    coin_toss = np.random.randint(2)
    if (coin_toss==0):
        player1 = "black"
        black = gpu_Q1
        white = gpu_Q2
    else:
        player1 = "white"
        black = gpu_Q2
        white = gpu_Q1
    
    turn_switcher = {"black": "white",
                     "white": "black"}
    # Define pipe for GPU process
    conn_rec, conn_send = Pipe(False)

    # Start game
    n_rounds = 0
    turn_color = "white"
    number_passes = 0
    go_game = go_board()
    
    # Evalute first node
    root_node, v = gen_node(black, go_game, "black", conn_rec, conn_send, True)
    # Define dummy root node for white, as it will be generated later
    other_root_node = None
    
    # Run next moves
    while True:
        n_rounds += 1
        turn_color = turn_switcher[turn_color]
        
        # Select if the current player is black or white
        if (turn_color == "black"):                        
            player = black
            not_player = white
        else:
            player = white
            not_player = black
         
        # Simulate MCTS
        root_node = MCTS(root_node, player, N, turn_color, number_passes, MCTS_queue)
        
        # Compute legal policy
        pi_legal = root_node.N/root_node.N_total
        
        # Selecet action
        action = np.random.choice(n_positions+1, size=1, p=pi_legal)[0]
        
        # Convert and take action
        #print("Move n. ",n_rounds, "New move was: ", action, "color was: ", turn_color)
        if (action==n_positions):
                go_game.move('pass', turn_color)
                number_passes += 1
        else:
                go_game.move(np.unravel_index(action, (board_size, board_size)), turn_color)
                number_passes = 0
        # Check if game is over
        if ((number_passes==2) | (n_rounds>162)):
            break
        
        # Pick move
        try:
            root_node = root_node.action_edges[action]
        except:
            print("Color is: ", turn_color)
            print("pi_legal is: ", pi_legal)
            print("Root node w is", root_node.W)
            print("Root node Q is", root_node.Q)
            print("Root node n is", root_node.N)
            print("Action edges is", root_node.action_edges)
            print("Root node turn color is", root_node.turn_color)
            print("illigal board is", root_node.illigal_board)
            print("number of passes was", number_passes)
            
            
        try:
            # If any exploration for the oppponents chosen action has been done
            other_root_node = other_root_node.action_edges[action]
        except:
            # Evalute first node
            other_root_node, v = gen_node(not_player, go_game, turn_switcher[turn_color], conn_rec, conn_send, False)
            
        # Switch trees
        temp = root_node
        root_node = other_root_node
        other_root_node = temp

        
        
    ##### Game is over
    
    # Find winner
    # No resignation, 
    points = go_game.count_points()
    
    # Is black is winner
    if (points>0):
        if (player1=="black"):
            player1_won = 1
        else:
            player1_won = 0
    else:
        if (player1=="white"):
            player1_won = 1
        else:
            player1_won = 0
    winner_Q.put(player1_won)
    return
        
def sim_games(N_games, n_MCTS, model, number_of_processes, v_resign, model2 = None, duel=False, batch_size = 8, board_size = 9):
    #### Function for generating games
    process_workers = []
    torch.multiprocessing.set_start_method('spawn', force=True) # This is important for generating a worker with torch support

    model.eval() # Set model for evaluating
    # Make queues for sending data
    gpu_Q = Queue()
    if (duel==False):
        data_Q = Queue()
        # Also make pipe for receiving v_resign
        conn_rec, conn_send = Pipe(False)
        
        p_data = Process(target=data_handler, args=(data_Q, N_games, conn_send))
        process_workers.append(p_data)
    else:
        winner_Q = Queue()
        gpu_Q2 = Queue()
        process_workers.append(Process(target=gpu_worker, args=(gpu_Q2, batch_size, board_size, model2)))
    # Make counter and lock
    game_counter = Value('i', 0)
    lock = Lock()
    
    # Make process for gpu worker and data_loader
    
    process_workers.append(Process(target=gpu_worker, args=(gpu_Q, batch_size, board_size, model)))
    # Start gpu and data_loader worker
    for p in process_workers:
        p.start()
    # Construct tasks for workers
    procs = []
    torch.multiprocessing.set_start_method('fork', force=True)
    for i in range(number_of_processes):
        seed = np.random.randint(int(2**31))
        if (duel==True):
            procs.append(Process(target=sim_duel_game_worker, args=(gpu_Q, gpu_Q2, n_MCTS, winner_Q, N_games, lock, game_counter, seed, batch_size, board_size)))
        else:
            procs.append(Process(target=sim_game_worker, args=(gpu_Q, n_MCTS, data_Q, v_resign, N_games, lock, game_counter, seed, batch_size, board_size)))
     # Begin running games
    for p in procs:
        p.start()
    # Join processes

    if (duel==False):
        with tqdm(total=100, update_to=True) as pbar:
            while True:
                try:
                    v_resign = conn_rec.get(True, 0.1) # Receive new v_resign
                    break
                except:
                    pbar.update(game_counter.value)



    else:
        player1_wins = 0
        player2_wins = 0
        for i in tqdm(range(N_games)):
            player1_won = winner_Q.get(True)
            if (player1_won==1):
                player1_wins += 1
            else:
                player2_wins += 1
    
    for p in procs: 
        p.join()
                
    # Close processes
    for p in process_workers:
        p.terminate()
    
    # Returns v_resign if training else winrate when dueling
    if (duel==False):
        return v_resign
    else:
        return player1_wins, player2_wins



