from logic import Logic
from random_input_ml import RandomInputModel

class ModelHandler:

    def __init__(self, model_choice):
        self.display_score = False
        self.display_moves = False
        self.display_turn = False
        self.model_choice = model_choice
        self.logic = self.make_board()

    def make_board_with_input(self):
        # initializes board according to the user's config
        if(input("Display mid-game score? (Y/N): ").upper() == "Y"):
            self.display_score = True
        if(input("Display mid-game moves? (Y/N): ").upper() == "Y"):
            self.display_moves = True
        if(input("Display mid-game turn? (Y/N): ").upper() == "Y"):
            self.display_turn = True

        if(input("Would you like the bot to run with default configs? (Y/N): ").upper() == "Y"):
            return Logic()
        else:
            width = int(input("Width: "))
            height = int(input("Height: "))
            num_min = int(input("num_min: "))
            num_max = int(input("num_max: "))
            num_goal = int(input("num_goal: "))
            return Logic(width ,height, num_min, num_max, num_goal)
        
    def make_board(self):
        return Logic()
    
    def reset(self, seed=None):
        self.logic.reset(seed)
        
    def run(self):
        # returns the final score of the model
        
        turn = 0
        while True:
            if(self.display_turn):
                print("") 
            turn += 1
            # display info about the current status of the model
            if(self.display_turn):
                print(f"Turn {turn}:")
            if(self.display_score):
                print(f"Current Score - {self.logic.score}")
            if(self.display_moves):
                print(f"Current Moves - {self.logic.count_tiles_with_valid_move()}")
                print(self.logic)
            
            # make a move based on the model
            move = None
            if(self.model_choice == "random"):
                move = RandomInputModel.get_move(self.logic)

            # end if there are no valid moves left
            if(move == None):
                break

            is_success = self.logic.try_move_num(move[0], move[1], move[2], move[3])

            if not is_success:
                raise ValueError("something went wrong :(")

        return self.logic.score

