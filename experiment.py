import argparse
import sys
import scipy.io as io
import blabla
import os
import qlearning
import environment
import datetime
import json
import actions

variables_file = "experiment_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
json_dict = dict()


def get_args():
    """Arguments for the experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training_steps',type=int, default=10000)
    parser.add_argument('-e', '--evaluation_steps', type=int, default=5)
    return parser.parse_args()


def build_objects():
    buoys = list()
    steps_between_actions = 10
    vessel_id = '36'
    rudder_id = '0'
    thruster_id = '0'
    scenario = 'default'
    return qlearning.QLearning(), environment.Environment(buoys, steps_between_actions, vessel_id, rudder_id, thruster_id, scenario)


def main():
    with open('data.txt', 'w') as outfile:
        #TODO Use context for dumping json to file
        parameter = get_args()
        agent, env = build_objects()
        json_dict['actions'] = actions.action_combinations
        env.set_up()
        episode_transitions_list = list()
        # At first, the agent is exploring
        agent.exploring = True
        #Executes the number of training steps specified in the -t parameter
        #TODO Include episode loop
        for step in range(parameter.training_steps):
            transition = dict()
            #The first step is to define the current state
            state = env.get_state()
            transition[state.__str__()] = state
            #The agent selects the action according to the state
            action = agent.select_action(state)
            #The state transition is processed
            statePrime, action, reward = env.step(action)
            transition[action.__str__()] = action
            transition[statePrime.__str__()] = statePrime
            transition[reward.__str__()] = reward
            #The agent Q-update is performed
            #TODO Get signal for final states in order to reset episode
            agent.observe_reward(state, action, statePrime, reward)
            print("***Training step "+str(step+1)+" Completed")
            episode_transitions_list.append(transition)
        #Now that the training has finished, the agent can use his policy without updating it
        agent.exploring = False
        a = dict()
        a['QTable'] = agent.expose_QTable()
        json.dump(json_dict, variables_file)
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
