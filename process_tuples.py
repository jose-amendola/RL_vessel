from simulation_settings import *
import matplotlib.pyplot as plt
import pickle
import os
import reward
import learner

if __name__ == '__main__':

    dir_name = './dyna/samples'
    files = os.listdir(dir_name)
    transitions = list()
    batch_list = list()
    for file in files:
        with open(os.path.join(dir_name, file), 'rb') as infile:
            print('Loading file:',file)
            try:
                while True:
                    transitions = pickle.load(infile)
            except EOFError as e:
                pass
        print('Number of transitions added : ', len(transitions))
        batch_list = batch_list + transitions
    # states = [tuple[0] for tuple in batch_list]
    # states_p = [tuple[2] for tuple in batch_list]
    # pos_x = [state[0] for state in states]
    # pos_y = [state[1] for state in states]
    # pos_x_p = [state[0] for state in states_p]
    # pos_y_p = [state[1] for state in states_p]
    # plt.xlim(0, 12000)
    # plt.ylim(2000, 8000)
    # ax = plt.axes()

    replace_reward = reward.RewardMapper(plot_flag=False, r_mode_='cte')
    replace_reward.set_boundary_points(buoys)
    replace_reward.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    # new_list = replace_reward.replace_reward(batch_list)

    batch_learner = learner.Learner(r_m_=replace_reward, nn_=True)
    batch_learner.add_tuples(batch_list)
    batch_learner.set_up_agent()
    batch_learner.fqi_step(50)



    # ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
    # plt.draw()
    # plt.pause(2)
    # for i,st in enumerate(states):
    #     ax.arrow(pos_x[i] ,pos_y[i], (pos_x_p[i] - pos_x[i]), (pos_y_p[i] - pos_y[i]))


    # x_pos = [state[0] for state in states if state[0] < 12000]
    # y_pos = [state[1] for state in states if state[0] < 12000]
    # plt.plot(x_pos, y_pos,'g^')


