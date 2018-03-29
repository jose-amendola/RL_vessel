import numpy as np
from itertools import product

spaces = {'simple_action_space': {
        'rudder_lvls':[-0.4, 0.0, 0.4],
        'thruster_lvls':[-0.4, 0.0, 0.4]
        },
        'large_action_space':{
            'rudder_lvls':[-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'thruster_lvls':[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        }
}


class BaseAction(object):
    def __init__(self, space_name):
        action_space = spaces[space_name]
        self.rudder_lvls = action_space['rudder_lvls']
        self.thruster_lvls = action_space['thruster_lvls']
        self.action_combinations = list(map(list, product(self.rudder_lvls, self.thruster_lvls)))
        self.possible_actions = range(len(self.action_combinations))

    def map_from_action(self, action_number):
        comb = self.action_combinations[action_number]
        return comb[0], comb[1]

    def all_agent_actions(self):
        return self.possible_actions