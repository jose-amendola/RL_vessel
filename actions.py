import numpy as np
from itertools import product

agentActions = 4
preyActions = 4
NORTH, SOUTH, WEST, EAST, NOOP = range(5)

# rudder_lvls = np.arange(-1.0, 1.01, 0.2)
# thruster_lvls = np.arange(-0.6, 0.61, 0.2)
rudder_lvls = [-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
thruster_lvls = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
action_combinations = list(map(list, product(rudder_lvls, thruster_lvls)))
possible_actions = range(len(action_combinations))
action_dict = dict()
for i in possible_actions:
    action_dict[str(i)] = action_combinations[i]

def map_from_action(action_number):
    comb = action_combinations[action_number]
    return comb[0], comb[1]


def all_agent_actions():
    return possible_actions