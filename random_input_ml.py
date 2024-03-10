import random

class RandomInputModel:

    def get_move(logic_obj):
        # returns a 4 length list of the model's chosen move
        width = len(logic_obj.board)
        height = len(logic_obj.board)
        
        valid_moves = logic_obj.get_valid_moves_as_tuples()

        # return None if there are no valid moves left
        if(len(valid_moves) == 0):
            return None
        
        return random.choice(valid_moves)