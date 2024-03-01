from logic import Logic

class TextUserHandler:

    def __init__(self):
        self.logic = self.make_board_with_input()
        self.run()

    def make_board_with_input(self):
        # initializes board according to the user's config
        if(input("Would you like to play with default configurations? (Y/N): ").upper() == "Y"):
            return Logic()
        else:
            width = int(input("Width: "))
            height = int(input("Height: "))
            num_min = int(input("num_min: "))
            num_max = int(input("num_max: "))
            num_goal = int(input("num_goal: "))
            return Logic(width ,height, num_min, num_max, num_goal)
        
    def run(self):
        # starts running game logic
        while self.logic.valid_moves_available():
            print("")
            print(f"Current Score: {self.logic.score}")
            print(self.logic)
            while True:
                try:
                    print("")
                    from_input = input("Enter a tile to move. (r,c):").split(",")
                    from_input = [int(num) for num in from_input]
                    if(len(from_input) != 2):
                        raise ValueError()
                    from_r, from_c = from_input[0], from_input[1]
                    
                    to_input = input("Enter a tile to merge into. (r,c):").split(",")
                    to_input = [int(num) for num in to_input]
                    if(len(to_input) != 2):
                        raise ValueError()
                    to_r, to_c = to_input[0], to_input[1]

                    self.logic.try_move_num(from_r, from_c, to_r, to_c)
                    break
                except ValueError:
                    print()
                    print("Please enter valid input.")

        print(f"No available moves left!\nYour final score was {self.logic.score}.")

    def convert_index_to_r_c(self, index):
        row_i = index // self.logic.width
        col_i = index % self.logic.width
        
        return row_i, col_i