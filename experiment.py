# -*- coding: utf-8 -*-
"""
Created on May, 25th, 08:29 2017.

@author: Felipe Leno
Source for running a test experiment.

You have to complete the sources:
environment.py
actions.py


You might also want to change parameters in:
qlearning.py
tilecoding.py


"""

import argparse
import sys
import scipy.io as io
import random
import os
import qlearning
import environment
import datetime

variables_file = "experiment_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')


def get_args():
    """Arguments for the experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training_steps',type=int, default=10000)
    parser.add_argument('-e', '--evaluation_steps', type=int, default=5)
    return parser.parse_args()


    
def build_objects():
    """Builds the agent (Q-learning) object and environment object
            
    """
    return qlearning.QLearning(), environment.Environment()
    

def main():
    parameter = get_args()
    agent, env = build_objects()
    env.set_up()

    # At first, the agent is exploring
    agent.exploring = True
    #Executes the number of training steps specified in the -t parameter
    for step in range(parameter.training_steps):
        #The first step is to define the current state
        state = env.get_state()
        #The agent selects the action according to the state
        action = agent.select_action(state)
        #The state transition is processed
        statePrime, action, reward = env.step(action)
        #The agent Q-update is performed
        agent.observe_reward(state, action, statePrime, reward)
        print("***Training step "+str(step+1)+" Completed")
    #Now that the training has finished, the agent can use his policy without updating it
    agent.exploring = False
    a = dict()
    a['QTable'] = agent.expose_QTable()
    # io.savemat(variables_file, a)
    # Executes the number of evaluation steps specified in the -e parameter
    for step in range(parameter.evaluation_steps):
        #Mostly the same as training, but without observing the rewards
        #The first step is to define the current state
        state = env.get_state()
        #The agent selects the action according to the state
        action = agent.select_action(state)
        #The state transition is processed
        env.step(action)
        print("***Evaluation step " + str(step+1) + " Completed")
    
    

if __name__ == '__main__':
    main()
