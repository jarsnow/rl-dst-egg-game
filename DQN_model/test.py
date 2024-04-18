# from collections import namedtuple

from logic_env import LogicEnv
from DQN_system import Agent

def main():
    #print_moves()
    agent = Agent()
    agent.run()

def print_moves():
    for i in range(64):
        env = LogicEnv()
        print("all possible moves:")
        print(f"Move: {env.get_playable_move_from_input(i)}, is valid?: {env.logic.is_valid_move(*env.get_playable_move_from_input(i))}")

if __name__ == "__main__":
    main()