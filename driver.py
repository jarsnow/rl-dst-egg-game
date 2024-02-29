from logic import Logic

def main():
    board = Logic()
    print(board)
    board.try_move_num(0,0,1,1)
    print(board)

main()