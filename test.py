from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

test_1 = Transition("state_obj", "action_input", "next_state_obj", 10)
print(test_1)