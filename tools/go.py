import numpy as np
import pickle as cPickle
from collections import deque, Counter
import itertools
def deep_copy_array(board):
    board_size = board.shape[0]
    copy_dic = {}
    new_array = np.empty(board.shape, dtype=board.dtype)
    # First find unique elements and make a copy by reference
    for i in range(board_size):
        for j in range(board_size):
            test = id(board[i, j])
            if test not in copy_dic:
                copy_dic[test] = board[i, j].copy()
            new_array[i, j] = copy_dic[test]
    """ 
    # Fill in new array
    map_func = lambda x: copy_dic[id(x)]
    new_array = np.vectorize(map_func, otypes=[list])(board)
    The above is equal to
        new_array = np.empty(board.shape, dtype=board.dtype)
        for i in range(board_size):
            for j in range(board_size):
                new_array[i,j] = copy_dic[id(board[i, j])]
    """

    return new_array

def convert_liberty_list(liberties, board_size=9):
    new_liberties = np.zeros((board_size, board_size))
    for i in range(board_size):
        for j in range(board_size):
            new_liberties[i, j] = liberties[i, j][0]
    return new_liberties


class go_board:
    # Class for storing go board
    # Pieces are stored in a 2d array where (0,0) is top left and (18,0) is bottom left:
    #   1 for a black piece
    #   0 for non-occupied position
    #   -1 for a white piece.
    # Notice a flipped board also is storred for quick turn shifts

    def __init__(self, board_size=9, copy=False):
        # Check if copy of object is to be performed
        if (copy == False):
            self.group_counter = 1
            self.board_size = board_size
            self.reset_board()
            self.ko = None
            self.koed_place = None
            self.color_2_number = {"black": 1, "white": -1}
            self.komi = 6.5
            self.game_ended = False
            self.last_move = None
            self.allies = np.frompyfunc(lambda: [0], 0, 1)(np.empty((self.board_size, self.board_size), dtype=object))
            # self.compare_board = np.frompyfunc(lambda: [0], 0, 1)(np.empty((self.board_size,self.board_size), dtype=object))
            self.legal_board_black = np.ones((self.board_size, self.board_size))
            self.legal_board_white = np.ones((self.board_size, self.board_size))

            self.zero_liberties = deque([])
            self.zero_lib_board = np.zeros((self.board_size, self.board_size), dtype="int8")

            self.neighbours = np.empty((self.board_size, self.board_size), dtype=object)
            for i in range(self.board_size):
                for j in range(self.board_size):
                    self.neighbours[i, j] = self.get_neighbours_fast((i, j))

    # Resets board to start


    def reset_board(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype="int8")
        self.board_flipped = np.zeros((self.board_size, self.board_size), dtype="int8")

        self.feature_board_black = np.zeros((17, self.board_size, self.board_size), dtype=bool)
        self.feature_board_black[16, :, :] = 1
        self.feature_board_white = np.zeros((17, self.board_size, self.board_size), dtype=bool)

        # Now construct liberties board
        self.liberties = np.empty((self.board_size, self.board_size), dtype="object")
        for i in range(1, self.board_size - 1):
            for j in range(1, self.board_size - 1):
                self.liberties[i, j] = [4]

        for i in range(self.board_size):
            self.liberties[0, i] = [3]
            self.liberties[-1, i] = [3]
            self.liberties[i, 0] = [3]
            self.liberties[i, -1] = [3]
        self.liberties[0, 0] = [2]
        self.liberties[-1, 0] = [2]
        self.liberties[0, -1] = [2]
        self.liberties[-1, -1] = [2]
        self.groups = np.zeros((self.board_size, self.board_size), dtype="int16")

    def copy_game(self):
        ### Metod for deep copying of object
        # gc.disable()
        # Intialize new object
        copy_object = go_board(board_size=self.board_size, copy=True)

        # Copy values
        copy_object.group_counter = self.group_counter
        copy_object.board_size = self.board_size
        copy_object.ko = self.ko
        copy_object.komi = self.komi
        copy_object.koed_place = self.koed_place
        copy_object.color_2_number = self.color_2_number
        copy_object.game_ended = self.game_ended
        copy_object.last_move = self.last_move

        # Copy value arrays
        copy_object.legal_board_black = np.copy(self.legal_board_black)
        copy_object.legal_board_white = np.copy(self.legal_board_white)
        copy_object.board = np.copy(self.board)
        copy_object.board_flipped = np.copy(self.board_flipped)
        copy_object.groups = np.copy(self.groups)
        copy_object.zero_lib_board = np.copy(self.zero_lib_board)
        copy_object.neighbours = np.copy(self.neighbours)
        copy_object.feature_board_black = np.copy(self.feature_board_black)
        copy_object.feature_board_white = np.copy(self.feature_board_white)

        # Copy list
        copy_object.zero_liberties = self.zero_liberties.copy()

        # Copy array of lists
        test = cPickle.dumps([self.allies, self.liberties], protocol=cPickle.HIGHEST_PROTOCOL, fix_imports=False)
        copy_object.allies, copy_object.liberties = cPickle.loads(test, fix_imports=False)

        return copy_object

    def get_board(self, color):
        if (color == "black"):
            return self.board
        return self.board_flipped

    def get_state(self, color):
        if (color == "black"):
            return self.feature_board_black
        return self.feature_board_white

    def get_neighbours_fast(self, position):
        # Get list of neighbours, with guaranteed no edge encountered
        #
        small_board = self.board_size - 1
        if (0 < position[0] < small_board) & (0 < position[1] < small_board):
            return [(position[0] + 1, position[1]), (position[0] - 1, position[1]),
                    (position[0], position[1] + 1), (position[0], position[1] - 1)]
        neighbours = deque([])
        if (-1 < position[0] + 1 < self.board_size):
            neighbours.append((position[0] + 1, position[1]))
        if (0 < position[0] < self.board_size + 1):
            neighbours.append((position[0] - 1, position[1]))
        if (-1 < position[1] + 1 < self.board_size):
            neighbours.append((position[0], position[1] + 1))
        if (0 < position[1] < self.board_size + 1):
            neighbours.append((position[0], position[1] - 1))
        return neighbours

    def capture(self, allies, placed, color_number):
        # Capture a group
        captured = 0
        for ally in allies:
            self.board[ally], self.board_flipped[ally] = 0, 0  # update board
            self.legal_board_black[ally] = 1
            self.legal_board_white[ally] = 1
            self.groups[ally] = 0  # Update groups
            self.liberties[ally] = [0]
            self.allies[ally] = deque([ally])
            # Save elements captured for liberty countings
            captured += 1

        for ally in allies:
            self.mass_increase_liberties(ally)

        if (captured == 1):
            self.ko = placed
            # Update legal board for ko
            self.zero_liberties.append(allies[0])
            self.zero_lib_board[allies[0]] = 1
            # Check if ko can even be captured
            if (self.liberties[self.ko][0] == 1):
                # Check if Ko now belongs to a group
                num_ally = 0
                neighbours = self.neighbours[self.ko]
                for neighbour in neighbours:
                    num_ally += (self.board[neighbour] == color_number)
                if num_ally == 0:
                    # find neutral space to prohibit placement
                    if (color_number == 1):
                        self.koed_place = (-1, allies[0])
                    else:
                        self.koed_place = (1, allies[0])

    def get_liberty_and_color(self, index):
        # Return the liberty and color of position
        return (self.liberties[index][0], self.board[index])

    def mass_reduce_liberties(self, position):
        # Increase the liberty by one for each neighbour
        neighbours = self.neighbours[position]
        duplicate_groups = deque([])
        for neighbour in neighbours:
            group = self.groups[neighbour]
            if (group in duplicate_groups):
                if group != 0:
                    self.liberties[neighbour][0] += 1
            duplicate_groups.append(group)
            self.liberties[neighbour][0] -= 1

    def mass_increase_liberties(self, position):
        # Increase the liberty by one for each neighbour
        neighbours = self.neighbours[position]
        duplicate_groups = deque([])
        for neighbour in neighbours:
            group = self.groups[neighbour]
            if (group in duplicate_groups) & (group != 0):
                self.liberties[neighbour][0] -= 1
            duplicate_groups.append(group)
            self.liberties[neighbour][0] += 1

    def check_capture(self, index, played_color_number, played):
        # Check if group should be captured, and update list of neutral
        #  positions with 0 liberty
        if self.liberties[index][0] < 1:
            if self.board[index] != played_color_number:
                if (self.board[index] == 0):
                    self.zero_liberties.append(index)
                    self.zero_lib_board[index] = 1
                else:
                    self.capture(self.allies[index], played, played_color_number)

    def is_neutral(self, position):
        return self.board[position] == 0

    def mass_check_capture(self, position, color):
        # Check if neighbouring pieces should be captured
        color_number = self.color_2_number[color]
        neighbours = self.neighbours[position]
        for neighbour in neighbours:
            self.check_capture(neighbour, color_number, position)

    def change_group(self, old_group_list, new_group_id, liberty_list):
        # Update the old groups
        new_group = deque([])
        for group in old_group_list:
            new_group += group
        for ally in new_group:
            self.groups[ally] = new_group_id
            self.liberties[ally] = liberty_list
            self.allies[ally] = new_group

    def place_and_group(self, position, color):
        # Place and update groups for piece
        color_number = self.color_2_number[color]
        # Calculate liberty from old groups and new piece, and get group ids for ally groups
        position_liberty = self.liberties[position]
        neighbours = self.neighbours[position]
        neutrals = deque([])
        allies = deque([])
        allies_group = Counter()
        for neighbour in neighbours:
            if self.board[neighbour]==0:
                neutrals.append(neighbour)

            elif self.board[neighbour] == color_number:
                allies.append(neighbour)
                allies_group[self.groups[neighbour]] += 1

        len_group = len(allies_group)
        if len_group == 0:
            # No allied group was found
            self.place_piece(position, color_number, position_liberty, self.group_counter, allies)
            self.group_counter += 1
            return

        elif len_group == 1:
            # Exactly one ally group was found
            old_group_id = self.groups[allies[0]]
            shared_neutral = 0

            # Get neutral zones' neighbours
            for neutral in neutrals:
                neutral_neighbours = self.neighbours[neutral]
                for neutral_neighbour in neutral_neighbours:
                    if self.groups[neutral_neighbour] == old_group_id:
                        shared_neutral += 1
                        break
            # Update liberties based on neighbouring neutral zones
            new_liberty = position_liberty[0] - shared_neutral + self.liberties[allies[0]][0]

            # Update group members
            group_members = self.allies[allies[0]]
            group_members.append(position)
            # Update allies
            self.allies[position] = group_members
            # Update groupds
            self.groups[position] = old_group_id
            self.liberties[position] = self.liberties[allies[0]]
            self.liberties[allies[0]][0] = new_liberty

            # Update board
            self.board[position] = (color_number == 1) - (color_number == -1)
            self.board_flipped[position] = (color_number == -1) - (color_number == 1)

            # Update go state
            if color_number == 1:
                self.feature_board_black[((0,) + position)] = 1
                self.feature_board_white[((1,) + position)] = 1
            else:
                self.feature_board_white[((0,) + position)] = 1
                self.feature_board_black[((1,) + position)] = 1

            self.legal_board_black[position] = 0
            self.legal_board_white[position] = 0
            return

        elif len(allies_group) > 1:
            # Exactly one ally group was found
            self.board[position] = (color_number == 1) - (color_number == -1)
            self.board_flipped[position] = (color_number == -1) - (color_number == 1)

            # Update go state
            if color_number == 1:
                self.feature_board_black[((0,) + position)] = 1
                self.feature_board_white[((1,) + position)] = 1
            else:
                self.feature_board_white[((0,) + position)] = 1
                self.feature_board_black[((1,) + position)] = 1

            self.legal_board_black[position] = 0
            self.legal_board_white[position] = 0
            group_members = deque([position])
            # Get allie list for each group
            group_check = deque([])
            for ally in allies:
                if self.groups[ally] not in group_check:
                    group_check.append(self.groups[ally])
                    group_members += self.allies[ally]

            # Calculate liberties
            # neighbours = deque(map(self.get_neighbours_fast,group_members))
            # neighbours = list(itertools.chain.from_iterable(neighbours))
            neighbours = self.neighbours[tuple(zip(*group_members))]
            neighbours = list(itertools.chain.from_iterable(neighbours))
            neighbours = Counter(neighbours)
            new_liberty = [sum(list(map(self.is_neutral, neighbours)))]
            for group_member in group_members:
                self.allies[group_member] = group_members
                self.groups[group_member] = self.group_counter
                self.liberties[group_member] = new_liberty

            self.group_counter += 1
            return

    def place_piece(self, move, color_number, liberty, group_id, allies):
        # Place piece and update group IDs
        self.board[move] = (color_number == 1) - (color_number == -1)
        self.board_flipped[move] = (color_number == -1) - (color_number == 1)
        self.legal_board_black[move] = 0
        self.legal_board_white[move] = 0
        # Update go state
        if color_number == 1:
            self.feature_board_black[((0,) + move)] = 1
            self.feature_board_white[((1,) + move)] = 1
        else:
            self.feature_board_white[((0,) + move)] = 1
            self.feature_board_black[((1,) + move)] = 1

        self.groups[move] = group_id
        self.liberties[move] = liberty
        allies.append(move)
        self.allies[move] = allies

    def can_capture_black(self, position, color_number):
        # Check if piece can capture to avoid suicide. This takes Ko into account.
        # Move is illigal if 0 is returned, move is legal if 1 is returned
        neighbours = self.neighbours[position]

        liberty_list = map(self.get_liberty_and_color, neighbours)
        # liberty_list = list(itertools.chain.from_iterable(liberty_list))
        for liberty, color_number in liberty_list:
            if liberty == 1:
                if color_number != 1:
                    return 1
            elif color_number == 1:
                return 1
        return 0

    def can_capture_white(self, position, color_number):
        # Check if piece can capture to avoid suicide. This takes Ko into account.
        # Move is illigal if 0 is returned, move is legal if 1 is returned
        neighbours = self.neighbours[position]

        liberty_list = map(self.get_liberty_and_color, neighbours)
        # liberty_list = list(itertools.chain.from_iterable(liberty_list))
        for liberty, color_number in liberty_list:
            if liberty == 1:
                if color_number == 1:
                    return 1
            elif color_number != 1:
                return 1
        return 0

    def move(self, move, color):
        # Runs a move through
        # Reset Ko
        self.ko = None
        self.koed_place = None

        # Check for move was pass
        if (move == "pass"):
            # Game ended
            self.game_ended = (self.last_move == "pass")
            self.last_move = move

            # Update go state
            self.feature_board_black[2:16, :, :] = self.feature_board_black[0:14, :, :]
            self.feature_board_white[2:16, :, :] = self.feature_board_white[0:14, :, :]
            return

        self.last_move = move

        # TROUBLESHOOTING LINE
        """
        legal_board = self.get_legal_board(color)
        if (legal_board[move]!=1):
            print("illigal move by color", color)
            print("board was:", self.board)
            print("move was", move)
        """

        # Check if any zero liberties should be removed
        if self.zero_lib_board[move] == 1:
            self.zero_liberties.remove(move)
            self.zero_lib_board[move] = 0
        # Update go state
        self.feature_board_black[2:16, :, :] = self.feature_board_black[0:14, :, :]
        self.feature_board_white[2:16, :, :] = self.feature_board_white[0:14, :, :]

        # First reduce neighours liberties
        self.mass_reduce_liberties(move)
        # Check for capture and capture
        self.mass_check_capture(move, color)
        # Update place piece and update groups
        self.place_and_group(move, color)

    def get_legal_board(self, color):
        # returns an array containing legal moves
        # 1 for legal move, zero for illigal move
        color_number = self.color_2_number[color]

        # Check black legal moves
        if color_number == 1:
            capture_status = list(
                map(lambda position: self.can_capture_black(position, color_number), self.zero_liberties))
            for position, capture in zip(self.zero_liberties, capture_status):
                self.legal_board_black[position] = capture

            # Check for ko
            if (self.ko != None):
                if (self.koed_place != None):
                    if (self.koed_place[0] == 1):
                        self.legal_board_black[self.koed_place[1]] = 0
            return self.legal_board_black

        # Check white legal moves
        else:
            capture_status = list(
                map(lambda position: self.can_capture_white(position, color_number), self.zero_liberties))
            for position, capture in zip(self.zero_liberties, capture_status):
                self.legal_board_white[position] = capture
            # Check for ko
            if (self.ko != None):
                if (self.koed_place != None):
                    if (self.koed_place[0] == -1):
                        self.legal_board_white[self.koed_place[1]] = 0
            return self.legal_board_white

    def map_area(self, point_board, board_size, starting_position, area, black_enc, white_enc):

        def get_neighbours(board_size, position):
            # Get list of neighbours, with guaranteed no edge encountered
            potential_neighbours = [(position[0] + 1, position[1]),
                                    (position[0] - 1, position[1]),
                                    (position[0], position[1] + 1),
                                    (position[0], position[1] - 1)]
            neighbours = deque([])
            for potential_neighbour in potential_neighbours:
                if (-1 < potential_neighbour[0] < board_size) & (-1 < potential_neighbour[1] < board_size):
                    neighbours.append(potential_neighbour)
            return neighbours

        # recursive function to map neutral unexplored area
        # It checks if a black and / or white piece is found and returns the explored area

        if (point_board[starting_position] == 3):
            point_board[starting_position] = 0
            area.append(starting_position)
            neighbours = get_neighbours(board_size, starting_position)
            for neighbour in neighbours:
                point_board, area, black_enc, white_enc = self.map_area(point_board, board_size, neighbour, area,
                                                                        black_enc, white_enc)
        elif (point_board[starting_position] == 1):
            black_enc = 1
        elif (point_board[starting_position] == -1):
            white_enc = 1
        return point_board, area, black_enc, white_enc

    def count_points(self):
        # Count points using area scoring
        # A positive score means black won, a negativ means white won, neutral is not possible with non-integer komi
        # TODO may implement mapping
        point_board = self.board

        # set neutral positions to value 3, to mark as unexplored
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (point_board[i, j] == 0):
                    point_board[i, j] = 3
        # Now run area scoring for unexplored neutral areas and convert them
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (point_board[i, j] == 3):
                    area = deque([])
                    black_enc = 0
                    white_enc = 0
                    point_board, area, black_enc, white_enc = self.map_area(point_board, self.board_size, (i, j), area,
                                                                            black_enc, white_enc)
                    if ((black_enc == 1) & (white_enc == 0)):
                        for area_explored in area:
                            point_board[area_explored] = 1
                    elif ((black_enc == 0) & (white_enc == 1)):
                        for area_explored in area:
                            point_board[area_explored] = -1
        return np.sum(point_board) - self.komi