from simulation_settings import *
import reward
import learner
import environment
from viewer import Viewer
import utils
import csv
import matplotlib.pyplot as plt
import tensorflow as tf


reward_mapping = reward.RewardMapper('quadratic', _g_helper=geom_helper)


def collect_trajectories():
    with tf.device('/cpu:0'):
        reward_mapping.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
        viewer = Viewer()
        viewer.plot_boundary(buoys)
        viewer.plot_goal(goal, 1000)

        agents = ['agents/agent_20180705153048Sequential_r____disc_0.8it11.h5']
        starting_points = [
            [11000, 5240, -100.5, 3, 0, 0],
            [11000, 5320, -104.5, 3, 0, 0],
            [11000, 5320, -105.5, 3, 0, 0],
            [11000, 5300, -103.5, 3, 0, 0],
            [11000, 5300, -102.5, 3, 0, 0],
            [11000, 5300, -101.5, 3, 0, 0]]

        for agent_obj in agents:
            viewer
            agent = learner.Learner(load_saved_regression=agent_obj, nn_=True)
            ret_tuples = list()
            results = list()
            num_steps = list()
            env = environment.Environment(rw_mapper=reward_mapping)
            env.set_up()
            for start_pos in starting_points:
                final_flag = 0
                transitions_list = list()
                compact_state_list = list()
                total_steps = 0
                env.set_single_start_pos_mode(start_pos)
                env.move_to_next_start()
                steps_inside = 0
                for step in range(evaluation_steps):
                    state = env.get_state()
                    print('Value for yaw_p :', state[5])
                    viewer.plot_position(state[0], state[1], state[2])
                    state_r = utils.convert_to_simple_state(state, geom_helper)
                    compact_state_list.append(state_r)
                    print('Value for yaw_p :', state_r[3])
                    action = agent.select_action(state_r)
                    state_prime, reward = env.step(action[0], action[1])
                    transition = (state, (action[0], action[1]), state_prime, reward)
                    if abs(state_r[2]) < 50:
                        steps_inside += 1
                    final_flag = env.is_final()
                    print("***Evaluation step " + str(step + 1) + " Completed")
                    transitions_list.append(transition)
                    ret_tuples += transitions_list
                    total_steps = step
                    if final_flag != 0:
                        break
                results.append(final_flag)
                num_steps.append(total_steps)
                with open(agent_obj + '_' + str(start_pos[1]) + '_' + str(start_pos[2]) + str(final_flag) + '.csv', 'wt') as out:
                    csv_out = csv.writer(out)
                    csv_out.writerow(['x', 'y', 'heading', 'rudder_lvl', 'balance'])
                    for tr, compact_state in zip(transitions_list, compact_state_list):
                        pos = (tr[0][0], tr[0][1], tr[0][2], tr[1][0], compact_state[2])
                        csv_out.writerow(pos)


def print_stats():
    trajectory_files = ['data/agent_20180705153048Sequential_r____disc_0.8it3.h5_5280_-103.5-1.csv']
    average_rudder_list = list()
    average_balance_list = list()
    for traj_file in trajectory_files:
        with open(traj_file, 'r') as file:
            reader = csv.reader(file,  delimiter=',')
            print(traj_file)
            for row in reader:
                if row:
                    if row[1] not in ['x', 'y', 'heading', 'rudder_lvl', 'balance']:
                        average_rudder_list.append(abs(float(row[3])))
                        average_balance_list.append(abs(float(row[4])))
        print('Average rudder magnitude: ', np.average(average_rudder_list))
        print('Average balance magnitude: ', np.average(average_balance_list))
        print('Variance of balance magnitude: ', np.var(average_balance_list))


def plot_loss():
    file = 'data/agent_20180705153048Sequential_r____disc_0.8it3.h5_5280_-103.5-1.csv'
    loss_list = list()
    with open(file, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if row:
                if row[1] != 'loss':
                    loss_list.append(float(row[1]))
    plt.plot(loss_list)
    plt.show()


if __name__ == "__main__":
    collect_trajectories()
    # plot_loss()
    # print_stats()