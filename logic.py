import gym
from gym import spaces
import numpy as np

class Logic:
    
    def __init__(self, width=4, height=4, num_min=1, num_max=20, num_goal=100, random_seed=0):
        self.width = width
        self.height = height
        self.num_min = num_min
        self.num_max = num_max
        self.num_goal = num_goal
        self.board = self.new_board(width=width, height=height)
        self.score = 0
        self.is_recording = False
        self.history_file_name = None
        self.file_obj = None
        
    def new_board(self, width=4, height=4):
        # returns a 2d matrix filled with random starting nums
        new_board = []
        for i in range(height):
            new_row = []
            for j in range(width):
                new_row.append(self.new_num())
            new_board.append(new_row)
        return new_board
    
    def new_num(self):
        return np.random.randint(self.num_min, self.num_max)

    def is_valid_move(self, from_row, from_col, to_row, to_col):
        # Checks for valid moves
        try:
            inside_bounds = from_row >= 0 and from_col >= 0 and to_row >= 0 and to_col >= 0 and from_row < self.height and to_row < self.width and from_col < self.width
            not_at_goal = self.board[from_row][from_col] < self.num_goal and self.board[to_row][to_col] < self.num_goal
            inside_3x3 = int(abs(from_row - to_row)) <= 1 and int(abs(from_col - to_col)) <= 1
            not_diagonal = int(abs(from_row - to_row)) + int(abs(from_col - to_col)) != 2
            sum_under_goal = self.board[from_row][from_col] + self.board[to_row][to_col] <= self.num_goal
            is_diff_tile = not ((from_row == to_row) and (from_col == to_col))
        except IndexError:
            return False
        
        return inside_bounds and not_at_goal and inside_3x3 and not_diagonal and sum_under_goal and is_diff_tile
    
    def try_move_num(self, from_row, from_col, to_row, to_col):
        # moves the number to the target r,c , returns True if it was successful, false if otherwise
        if not self.is_valid_move(from_row, from_col, to_row, to_col):
            return False
        
        self.board[to_row][to_col] += self.board[from_row][from_col]
        self.board[from_row][from_col] = None

        # add score:
        # if the total is equal to goal, add 2500 to the score
        #   after looking at actual gameplay, this number is decayed based on how long since either the game started,
        #   or the last time you made a sum equal to the number goal (egg in the minigame)
        # if the total isn't at the goal, add the newly created number to the score
        #   accurate to how this is done in the minigame
        SCORE_GOAL_REWARD = 2500
        if(self.board[to_row][to_col] == self.num_goal):
            self.score += SCORE_GOAL_REWARD
        else:
            self.score += self.board[to_row][to_col]

        self.apply_gravity_and_num()

        if(self.is_recording):
            self.write_curr_state_to_file()

        return True
    
    # def move_num_with_reward(self, from_row, from_col, to_row, to_col):
    #     if not self.is_valid_move(from_row, from_col, to_row, to_col):
    #         raise ValueError("you CANNOT put an invalid move into this function.")
        
    #     self.board[to_row][to_col] += self.board[from_row][from_col]
    #     self.board[from_row][from_col] = None

    #     # add score
    #     SCORE_GOAL_MULTIPLIER = 10
    #     if(self.board[to_row][to_col] == self.num_goal):
    #         self.score += SCORE_GOAL_MULTIPLIER * self.num_goal
    #     else:
    #         self.score += self.board[to_row][to_col]

    #     self.apply_gravity_and_num()

        
    def apply_gravity_and_num(self):
        # moves all the tiles above the empty space down 1, then add a new num to the top of the board
        empty_r = 0
        empty_c = 0
        found_empty = False
        for r, row in enumerate(self.board):
            for c, val in enumerate(row):
                if(val is None):
                    empty_r = r
                    empty_c = c
                    found_empty = True
        
        # should never run, hopefully
        if not found_empty:
            raise ValueError("something went wrong :(")
        
        # move all tiles above the empty space down 1
        for i in range(empty_r):
            self.board[empty_r - i][empty_c] = self.board[empty_r - i - 1][empty_c]
        
        # add a new_num to the top of the board
        self.board[0][empty_c] = self.new_num()

    def __str__(self):
        to_return = ""
        for row in self.board:
            for val in row:
                to_return += format(val, f'0{len(str(self.num_goal))}d') + " " # returns a string with the number as a 3 length integer. ex: 12 -> 012, 
            to_return += "\n"

        return to_return
    
    def valid_moves_available(self):
        # checks to see if there is any valid move left to be played (False -> game is over)
        return (self.count_tiles_with_valid_move() > 0)

    def tile_has_valid_move(self, row_i, col_i):
        # checks if a tile at a given r,c index as a move available
        return self.is_valid_move(row_i, col_i, row_i - 1, col_i) or self.is_valid_move(row_i,col_i, row_i + 1, col_i) or self.is_valid_move(row_i,col_i, row_i, col_i - 1) or self.is_valid_move(row_i,col_i, row_i, col_i + 1)
    
    def count_tiles_with_valid_move(self):
        # returns the amount of remaining tiles that have at least one move available
        count = 0
        for row_i, row in enumerate(self.board):
            for col_i, val in enumerate(row):
                if(self.tile_has_valid_move(row_i, col_i)):
                    count += 1
        return count
    
    def get_valid_moves_as_tuples(self):
        # returns a list of valid moves as a tuple of (from_row_i, from_col_i, to_row_i, to_col_i)
        valid_moves = []
        for from_row_i, from_row in enumerate(self.board):
            for from_col_i, val in enumerate(from_row):
                for to_row_i, to_row in enumerate(self.board):
                    for to_col_i, val in enumerate(to_row):
                        if(self.is_valid_move(from_row_i, from_col_i, to_row_i, to_col_i)):
                            valid_move = (from_row_i, from_col_i, to_row_i, to_col_i)
                            valid_moves.append(valid_move)
        return valid_moves
    
    def reset(self, seed=None, record=False, file_name=None):
        # reset the board
        self.board = self.new_board()
        self.score = 0

        # close file
        if(self.is_recording):
            self.is_recording = False
            self.close_history_file()

        if(record):
            self.is_recording = True
            self.history_file_name = file_name
            self.open_file_obj()
            self.write_curr_state_to_file()


    def get_nums_by_row(self):
        # get the board as a list
        to_return = []
        for row in self.board:
            for val in row:
                to_return.append(val)
        return to_return
    
    # helper function for logic_env
    # prioritizes having the 100's near the bottom if possible
    # big bonus for having a full row
    # DOES NOT SCALE WITH DIFFERENT SIZES!!!
    def get_current_reward(self):
        HEIGHT_OFFSET = 1
        GOAL_REWARD = 100
        ROW_REWARD = 1000

        reward = 0
        for row_i, row in enumerate(self.board):
            full_row = True
            for num in row:
                if(num == self.num_goal):
                    # has less rewards for physically higher 100's
                    reward += (row_i + HEIGHT_OFFSET) * GOAL_REWARD
                else:
                    full_row = False
            if(full_row):
                reward += ROW_REWARD
        
        return reward
        
    def game_ended(self):
        return self.count_tiles_with_valid_move() == 0
    
    def write_curr_state_to_file(self):
        if(self.file_obj is None):
            raise ValueError("no file obj")
        self.file_obj.write(str(self) + "\n")
        return
    
    def open_file_obj(self):
        self.file_obj = open(f".\game_outputs\{self.history_file_name}.txt", "w")

    def close_history_file(self):
        if(self.file_obj is None):
            raise ValueError("no file obj")
        
        self.file_obj.close()
