from collections import namedtuple

Outline = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

test_1 = Outline("logic_obj1", ("new", [1,0,0,0,0,0,0,0,0,1,0,0,0,0], "logic_obj2", "reward_num"))