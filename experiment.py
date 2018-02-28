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

variables_file = "experiment_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.json'
json_dict = dict()
episodes = 5000


def get_args():
    """Arguments for the experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training_steps',type=int, default=1000)
    parser.add_argument('-e', '--evaluation_steps', type=int, default=5)
    return parser.parse_args()


def build_objects():
    steps_between_actions = 20
    funnel_start = (14000, 7000)
    N01 = (11724.8015, 5582.9127)
    N03 =  (9191.6506, 4967.8532)
    N05 = (6897.7712, 4417.3295)
    N07 = (5539.2417, 4089.4744)
    Final = (5812.6136, 3768.5637)
    N06 = (6955.9288, 4227.1846)
    N04 = (9235.8653, 4772.7884)
    N02 = (11770.3259, 5378.4429)
    funnel_end = (14000, 4000)
    #TODO Organize JSON file
    #TODO Option for disable viewer
    #TODO Option for reading JSON and plotting it
    #TODO Fix viewer
    buoys = (funnel_start, N01, N03, N05, N07, Final, N06, N04, N02, funnel_end)

    vessel_id = '36'
    rudder_id = '0'
    thruster_id = '0'
    scenario = 'default'
    goal = ((N07[0]+Final[0])/2, (N07[1]+Final[1])/2)
    return qlearning.QLearning(), environment.Environment(buoys, steps_between_actions, vessel_id,
                                                          rudder_id, thruster_id, scenario, goal)


def main():
    with open(variables_file, 'w') as outfile:
        #TODO Use context for dumping json to file
        parameter = get_args()
        agent, env = build_objects()
        json_dict['actions'] = actions.action_combinations
        env.set_up()
        # At first, the agent is exploring
        agent.exploring = True
        #Executes the number of training steps specified in the -t parameter
        for episode in range(episodes):
            episode_transitions_list = list()
            for step in range(parameter.training_steps):
                transition = dict()
                #The first step is to define the current state
                state = env.get_state()
                transition[u'state'] = state
                #The agent selects the action according to the state
                action = agent.select_action(state)
                #The state transition is processed
                statePrime, action, reward = env.step(action)
                transition[u'action'] = action
                transition[u'statePrime'] = statePrime
                transition[u'reward'] = reward
                #The agent Q-update is performed
                #TODO Get signal for final states in order to reset episode
                final_flag = env.is_final()
                agent.observe_reward(state, action, statePrime, reward, final_flag)
                print("***Training step "+str(step+1)+" Completed")
                episode_transitions_list.append(transition)
                if final_flag:
                    continue
            json_dict['Episode'+str(episode)] = episode_transitions_list
            json_str = json.dumps(json_dict)
            outfile.write(json_str)
        #Now that the training has finished, the agent can use his policy without updating it
        agent.exploring = False
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
