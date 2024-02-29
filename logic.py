import random

class Logic:

    def __init__(self, width=4, height=4, num_min=1, num_max=20, num_goal=100):
        self.width = width
        self.height = height
        self.num_min = num_min
        self.num_max = num_max
        self.num_goal = num_goal
        self.board = self.new_board(width=width, height=height)
        self.score = 0
        
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
        return random.randint(self.num_min, self.num_max)

    def is_valid_move(self, from_row, from_col, to_row, to_col):
        # Checks for valid moves
        return not (from_row < 0 or from_col < 0 or to_row < 0 or to_col < 0 or # cannot go out of bounds
           from_row >= self.height or to_row >= self.height or from_col >= self.width or to_col >= self.width or # cannot go out of bounds
           self.board[from_row][from_col] >= self.num_goal or self.board[to_row][to_col] >= self.num_goal or # cannot already have reached the goal
           abs(from_row - to_row) != 1 or abs(from_col - to_col) != 1 or # has to be adjacent tiles
           self.board[from_row][from_col] + self.board[to_row][to_col] > self.num_goal) # sum has to be under or equal to the goal
    
    def try_move_num(self, from_row, from_col, to_row, to_col):
        # moves the number to the target r,c , returns True if it was successful, false if otherwise
        if not self.is_valid_move(from_row, from_col, to_row, to_col):
            return False
        
        self.board[to_row][to_col] += self.board[from_row][from_col]
        self.board[from_row][from_col] = None

        # add score
        SCORE_GOAL_MULTIPLIER = 10
        if(self.board[to_row][to_col] == self.num_goal):
            score += SCORE_GOAL_MULTIPLIER * self.num_goal
        else:
            score += self.board[to_row][to_col]
        
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
        