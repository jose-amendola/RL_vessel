import numpy as np
from itertools import product

spaces = {'simple_action_space': {
        'rudder_lvls':[-0.5, 0.0, 0.5],
        'thruster_lvls':[-0.5, 0.0, 0.5]
        },
        'cte_rotation': {
        'rudder_lvls':[-0.5, 0.0, 0.5],
        'thruster_lvls':[-0.5]
        },
        'stable': {
            'rudder_lvls': [-0.5, 0.0, 0.5],
            'thruster_lvls': [0.2]
        },
        'smooth': {
            'rudder_lvls': [-0.2, 0.0, 0.2],
            'thruster_lvls': [0.2]
        },
        'complete_angle': {
            'rudder_lvls': [-0.5, -0.2, 0.0, 0.2, 0.5],
            'thruster_lvls': [0.2]
        },
        'large_action_space':{
            'rudder_lvls':[-0.5, 0.0, 0.5],
            'thruster_lvls':[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        },
        'only_rudder_action_space':{
            'rudder_lvls':[-0.4, 0.0, 0.4],
            'thruster_lvls':[0.0]
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