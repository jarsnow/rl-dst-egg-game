# from collections import namedtuple
from logic_env import LogicEnv
from model_handler import ModelHandler

def main():
    #print_moves()
    #play_random(100)
    play_agent()
    #play_one_random_check_reward_state()

def print_moves():
    for i in range(64):
        env = LogicEnv()
        print("all possible moves:")
        print(f"Move: {env.get_playable_move_from_input(i)}, is valid?: {env.logic.is_valid_move(*env.get_playable_move_from_input(i))}")

def play_random(num):
    
    mh = ModelHandler("random")
    scores = []
    for i in range(num):
        final_score = mh.run()
        print(f"score of inter {i}: {final_score}")
        scores.append(final_score)
        mh.reset()
    
    print(f"avg over {num}: {sum(scores) / len(scores)}")

def play_one_random_check_reward_state():
    mh = ModelHandler("random")
    mh.run()

def play_agent():
    from DQN_system import Agent
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()