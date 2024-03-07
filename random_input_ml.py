import random

class RandomInputModel:

    def get_move(logic_obj):
        # returns a 4 length list of the model's chosen move
        width = len(logic_obj.board)
        height = len(logic_obj.board)
        
        while True:
            from_r_input = random.randint(0, height - 1)
            from_c_input = random.randint(0, width - 1)
            to_r_input = random.randint(0, height - 1)
            to_c_input = random.randint(0, width - 1)

            if(logic_obj.is_valid_move(from_r_input, from_c_input, to_r_input, to_c_input)):
                move = [from_r_input, from_c_input, to_r_input, to_c_input]
                return move