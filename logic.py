import random

class Logic:

    def __init__(self, width=4, height=4, num_min=1, num_max=20, num_goal=100):
        self.width = width
        self.height = height
        self.num_min = num_min
        self.num_max = num_max
        self.num_goal = num_goal
        
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