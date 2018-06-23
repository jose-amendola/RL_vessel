import qlearning
import environment
from simulation_settings import *
from viewer import Viewer
import pickle
import learner
import utils
import argparse
import csv
import random


def load_pickle_file(file_to_load):
    with open(file_to_load, 'rb') as infile:
        var_list = pickle.load(infile)
        episodes_list = list()
        while True:
            try:
                ep = pickle.load(infile)
                episodes_list.append(ep)
            except EOFError as e:
                break
    return var_list, episodes_list

def load_agent(file_to_load):
    agent_obj = None
    with open(file_to_load, 'rb') as infile:
        agent_obj = pickle.load(infile)
    return agent_obj

def replay_trajectory(episodes):
    view = Viewer()
    view.plot_boundary(buoys)
    view.plot_goal(goal, goal_factor)
    for episode in episodes:
        transitions_list = episode['transitions_list']
        for transition in transitions_list:
            state = transition[0]
            view.plot_position(state[0], state[1], state[2])
    view.freeze_screen()

def sample_transitions(start_state=0, end_state=-1) -> object:
    #TODO Implement
    action_space_name = 'complete_angle'
    action_space = actions.BaseAction(action_space_name)
    env = environment.Environment(buoys, steps_between_actions, vessel_id, rudder_id, thruster_id, scenario, goal,
                                  goal_heading_e_ccw, goal_vel_lon)
    env.set_up()
    # env.set_sampling_mode(start_state, end_state)
    env.set_single_start_pos_mode([9000, 4819.10098, -103.5, 3, 0, 0])
    # env.set_single_start_pos_mode([13000, 5777.706, -103.5, 3, 0, 0])
    # env.starts_from_file_mode('samples/starting_points_global_coord20180512195730')
    transitions_list = list()
    for episode in range(5000000):
        if episode == 1:
            env.start_bifurcation_mode()
        print('### NEW STARTING STATE ###',episode)
        #TODO fix onconsistency in order of methods from Environment
        env.move_to_next_start()
        final_flag = 0
        for action in action_space.action_combinations:
            # env.set_up()
            env.reset_to_start()
            for i in range(500000):
                rand_act = random.choice(action_space.action_combinations)
                if random.random() < 0.2:
                    act = rand_act
                else:
                    act = action
                state = env.get_state()
                angle = act[0]
                rot = act[1]
                if episode == 0:
                    angle = 0.0
                state_prime, rw = env.step(angle, rot)
                print('Reward:', rw)
                final_flag = env.is_final()
                # New format
                transition = (state, (angle, rot), state_prime, rw, final_flag)
                if i % 50 == 0 and episode == 0:
                    env.add_states_to_start_list(state_prime)
                transitions_list.append(transition)
                if final_flag != 0 or state[0] < 7000:
                    break
            with open(sample_file + 'action_' + action_space_name + '_s' + str(start) + '_' + str(episode),
                      'wb') as outfile:
                pickle.dump(transitions_list, outfile)
                transitions_list = list()


def main():
    action_space_name = 'cte_rotation'
    action_space = actions.BaseAction(action_space_name)
    agent = qlearning.QLearning(q_file, epsilon=0.1, action_space=action_space, gamma=1.0)
    env = environment.Environment(buoys, steps_between_actions, vessel_id, rudder_id, thruster_id, scenario, goal,
                                  goal_heading_e_ccw, goal_vel_lon)
    # with open(variables_file, 'wb') as outfile:
    pickle_vars = dict()
    pickle_vars['action_space'] = action_space_name
    env.set_up()
    env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
    # env.set_single_start_pos_mode([6000, 4000, -103.5, 3, 0, 0])
    agent.exploring = True
    # pickle.dump(pickle_vars, outfile)
    for episode in range(500):
        print('###STARTING EPISODE ', episode)
        transitions_list = list()
        final_flag = 0
        env.move_to_next_start()
        env.reset_to_start()
        for step in range(5000):
            state = env.get_state()
            print('Yaw:', state[2])
            angle, rot = agent.select_action(state)
            state_prime, reward = env.step(angle, rot)
            # state_rime, reward = env.step(0, 0)
            print('Reward:', reward)
            transition = (state, (angle, rot), state_prime, reward)
            final_flag = env.is_final()
            agent.observe_reward(state, angle, rot, state_prime, reward, final_flag)
            print("***Training step "+str(step+1)+" Completed")
            transitions_list.append(transition)
            if final_flag != 0:
                break
                # state_rime, reward = env.step(0, 0)
                # state_rime, reward = env.step(0, 0)
        with open('qlearning_' + 'action_' + action_space_name + '_' + str(episode),
                  'wb') as outfile:
            pickle.dump(transitions_list, outfile)
            transitions_list = list()
    with open(learner_file, 'wb') as outfile:
        pickle.dump(agent, outfile)


def evaluate_agent(ag_obj):
    agent = learner.Learner(load_saved_regression=ag_obj, nn_=True)
    env = environment.Environment(buoys, 20, vessel_id, rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw,
                                  goal_vel_lon, _increment=0.5)
    env.set_up()

    starting_points = [
                        [11000, 5250, -101, 3, 0, 0],
                        [11000, 5300, -104, 3, 0, 0],
                        [11000, 5280, -103.5, 3, 0, 0],
                        [11000, 5320, -103.5, 3, 0, 0],
                        [11000, 5320, -103.5, 3, 0, 0]]

    ret_tuples = list()
    # env.set_single_start_pos_mode([11000, 5380.10098, -103, 3, 0, 0])
    # env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
    # env.set_single_start_pos_mode([12000, 5500, -90, 3, 0, 0])
    # env.set_single_start_pos_mode([6600, 4200, -102, 3, 0, 0])
    # env.starts_from_file_mode('starting_points_global_coord')
    # env.move_to_next_start()
    results = list()
    num_steps = list()
    steps_inside_goal_region = list()
    for start_pos in starting_points:
        final_flag = 0
        transitions_list = list()
        total_steps = 0
        env.set_single_start_pos_mode(start_pos)
        env.move_to_next_start()
        steps_inside = 0
        for step in range(evaluation_steps):
            state = env.get_state(state_mode='simple_state')
            state_r = env.convert_to_simple_state(state)
            action = agent.select_action(state_r)
            state_prime, reward = env.step(action[0], action[1])
            transition = (state, (action[0], action[1]), state_prime, reward)
            if abs(state_r[2]) < 50:
                steps_inside+=1
            final_flag = env.is_final()
            print("***Evaluation step " + str(step + 1) + " Completed")
            transitions_list.append(transition)
            ret_tuples = ret_tuples + transitions_list
            total_steps = step
            if final_flag != 0:
                break
        results.append(final_flag)
        num_steps.append(total_steps)
        steps_inside_goal_region.append(steps_inside)
        with open('trajectory_'+agent.learner.__class__.__name__+'it'+str(total_steps)+'end'+str(final_flag), 'wb') as outfile:
            pickle.dump(transitions_list, outfile)
        with open('trajectory_'+agent.learner.__class__.__name__+'it'+str(total_steps)+'end'+str(final_flag)+'.csv', 'wt') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['x', 'y', 'heading', 'rudder_lvl'])
            for tr in transitions_list:
                pos = (tr[0][0], tr[0][1], tr[0][2], tr[1][0])
                csv_out.writerow(pos)
    with open('results'+agent.learner.__class__.__name__+datetime.datetime.now().strftime('%Y%m%d%H%M%S'), 'wb') as outfile:
            pickle.dump(num_steps, outfile)
            pickle.dump(results, outfile)
            pickle.dump(steps_inside_goal_region, outfile)
    return ret_tuples


def run_episodes(agent):
    env = environment.Environment(buoys, 20, vessel_id, rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw,
                                  goal_vel_lon, _increment=0.5)
    env.set_up()

    starting_points = [
                       [11000, 5360, -106, 3, 0, 0],
                       [11000, 5240,-100, 3, 0, 0],
                       [11000, 5360, -106, 2.5, 0, 0],
                        [11000, 5240,-100, 2.5, 0, 0],
                        [11000, 5360, -106, 2.0, 0, 0],
                        [11000, 5240,-100, 2.0, 0, 0],
                       [11000, 5360, -106, 1.5, 0, 0],
                        [11000, 5240, -100, 1.5, 0, 0],
                       [11000, 5360, -106, 1.0, 0, 0],
                        [11000, 5240,-100, 1.0, 0, 0],
                       [11000, 5360, -106, 0.5, 0, 0],
                        [11000, 5240,-100, 0.5, 0, 0]]
    ret_tuples = list()
    # env.set_single_start_pos_mode([11000, 5380.10098, -103, 3, 0, 0])
    # env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
    # env.set_single_start_pos_mode([12000, 5500, -90, 3, 0, 0])
    # env.set_single_start_pos_mode([6600, 4200, -102, 3, 0, 0])
    # env.starts_from_file_mode('starting_points_global_coord')
    # env.move_to_next_start()
    results = list()
    num_steps = list()
    for start_pos in starting_points:
        final_flag = 0
        transitions_list = list()
        total_steps = 0
        env.set_single_start_pos_mode(start_pos)
        env.move_to_next_start()
        for step in range(50):
            state = env.get_state()
            state_r = utils.convert_to_simple_state(state)
            action = None
            if random.random() < 0.2:
                action = random.choice(action_space.action_combinations)
            else:
                action = agent.select_action(state_r)
            state_prime, reward = env.step(action[0], action[1])
            final_flag = env.is_final()
            transition = (state, (action[0], action[1]), state_prime, reward, final_flag)
            print("***Evaluation step " + str(step + 1) + " Completed")
            transitions_list.append(transition)
            ret_tuples += transitions_list
            total_steps = step
            if final_flag != 0:
                break
        results.append(final_flag)
        num_steps.append(total_steps)
        with open('trajectory_'+agent.learner.__class__.__name__+'it'+str(total_steps)+'end'+str(final_flag), 'wb') as outfile:
            pickle.dump(transitions_list, outfile)
        with open('trajectory_'+agent.learner.__class__.__name__+'it'+str(total_steps)+'end'+str(final_flag)+'.csv', 'wt') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['x', 'y', 'heading', 'rudder_lvl'])
            for tr in transitions_list:
                pos = (tr[0][0], tr[0][1], tr[0][2], tr[1][0])
                csv_out.writerow(pos)
    with open('results'+agent.learner.__class__.__name__, 'wb') as outfile:
            pickle.dump(num_steps, outfile)
            pickle.dump(results, outfile)
    return ret_tuples

if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('s', type=str, help='Start')
    parser.add_argument('e', type=str, help='End')
    parser.add_argument('--mode', type=str, help='End')

    args = parser.parse_args()
    start = 0
    end = -1
    if args.s:
        start = args.s
    if args.e:
        end = args.e

    # main()
    sample_transitions(start, end)
    # ag = load_agent('agents/agent_20180519195648DecisionTreeRegressor_r_rule_disc_0 .0it1')
    # evaluate_agent(ag)
    # evaluate_agent('agents/agent_20180524154344Sequential_r_linear_with_rudder_punish_disc_0.8it5.h5')


