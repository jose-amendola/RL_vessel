from simulation_settings import *
import matplotlib.pyplot as plt
import pickle
import os
import reward
import learner
from viewer import Viewer
import trace
from planar import Vec2
from planar.line import Line, LineSegment
import utils
import random
import experiment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# def plot_simple_state_tuples(tuples):
#     state = [tuple[0] for tuple in tuples]


def get_success_trajectories(tuples):
    trajs = list()
    tmp = list()
    for tup in tuples:
        if tup[4] == 0:
            tmp.append(tup)
        elif tup[4] == 1:
            tmp.append(tup)
            trajs = trajs + tmp
            tmp = list()
        elif tup[4] == -1:
            tmp = list()
    return trajs

def convert_state_space(state,rw_mapper):
    v_lon, v_drift, n_used = utils.global_to_local(state[3],state[4],state[2])
    bl = rw_mapper.get_shore_balance(state[0],state[1])
    misalign = state[2] + 103.5 #guidance angle
    return (v_lon, misalign, bl)

def plot_sequence(tuples):
    view = Viewer()
    view.plot_boundary(buoys)
    view.plot_goal(goal, goal_factor)
    for tuple in tuples:
        state = tuple[0]
        print('Position: ', state[0], state[1], state[2])
        print('Velocities: ', state[3], state[4], state[5])
        print('Action: ',tuple[1])
        view.plot_position(state[0], state[1], state[2])
    view.freeze_screen()

def mirror_point(point_a, point_b, point_to_mirror):
    vec_a = Vec2(point_a[0], point_a[1])
    vec_b = Vec2(point_b[0] - point_a[0], point_b[1] - point_a[1])
    line = Line(vec_a,vec_b)
    org_vec = Vec2(point_to_mirror[0],point_to_mirror[1])
    reflect_vec = line.reflect(org_vec)
    return reflect_vec.x, reflect_vec.y

def mirror_velocity(vel_x, vel_y, heading_n_cw, new_heading):
    v_lon, v_drift, not_used = utils.global_to_local(vel_x, vel_y, heading_n_cw)
    new_v_x, new_v_y, not_used = utils.local_to_global(v_lon, - v_drift, 90-new_heading)
    return new_v_x, new_v_y

def reflect_state_across_line(point_a, point_b, state):
    line_angle = - 103.5
    x = tuple[0][0]
    y = tuple[0][1]
    new_x, new_y = mirror_point(point_a, point_b,(state[0], state[1]))
    yaw = tuple[0][2]
    v_x = tuple[0][3]
    v_y = tuple[0][4]
    v_yaw = tuple[0][5]
    new_yaw = line_angle - (yaw - line_angle)
    new_v_x, new_v_y = mirror_velocity(state[3], state[4], yaw, new_yaw)
    new_v_yaw = - v_yaw
    return (new_x, new_y, new_yaw, new_v_x, new_v_y, new_v_yaw)


def reflect_tuple_on_line(point_a, point_b, tuple):
    reflected_state = reflect_state_across_line(point_a, point_b, tuple[0])
    reflected_state_p = reflect_state_across_line(point_a, point_b, tuple[2])
    reflected_action = (-tuple[1][0], tuple[1][1])
    return (reflected_state, reflected_action, reflected_state_p, tuple[3], tuple[4])

def get_strictly_simetric_set(org_tuples, mapper, point_a, point_b):
    tuples = [tup for tup in org_tuples if tup[0][2]+103.5 < 20 and tup[0][0] < 11500]
    print('Number of tuples to be considered:', len(tuples))
    tuples_with_reflection = list()
    for tuple in tuples:
        reflect_tuple = reflect_tuple_on_line(point_a, point_b, tuple)
        if mapper.is_inbound_coordinate(reflect_tuple[0][0], reflect_tuple[0][1]):
            tuples_with_reflection.append(tuple)
            tuples_with_reflection.append(reflect_tuple)
    print('Number of tuples after reflection:', len(tuples_with_reflection))


if __name__ == '__main__':
    # dir_name = './samples'
    # files = os.listdir(dir_name)
    # transitions = list()
    # batch_list = list()
    # for file in files:
    #     with open(os.path.join(dir_name, file), 'rb') as infile:
    #         print('Loading file:',file)
    #         try:
    #             while True:
    #                 #TODO rever
    #                 transitions = pickle.load(infile)
    #         except EOFError as e:
    #             pass
    #     print('Number of transitions added : ', len(transitions))
    #     batch_list = batch_list + transitions
    # tuples = [tuple for tuple in batch_list if (tuple[1][0] < 0 and tuple[0][2] < -80 and tuple[0][3]< 0)]
    # transitions = list()
    # with open('samples/samples_bundle', 'rb') as infile:
    #     transitions = pickle.load(infile)
    # plot_sequence(transitions)
    # trajectories = list()
    # tmp = list()
    # for tuple in tuples:
    #     tmp.append(tuple)
    #     if tuple[4] != 0 :
    #         trajectories.append(tmp)
    #         tmp = list()
    #
    # # selected_trajectories = [traj for traj in trajectories if traj[0][0][0] > 7000 and traj[0][0][2] < -3.0]
    # model_trajectory = [traj for traj in trajectories if traj[0][0][0] > 9000 and
    #                                                     traj[-1][4] == 1]
    # resulting_list = list()
    # for traj in selected_trajectories:
    #     resulting_list = resulting_list + traj
    # for i in range(10):
    #     for m_t in model_trajectory:
    #         resulting_list = resulting_list + m_t


    # resulting_list = list()
    # for traj in model_trajectory:
    #     resulting_list = resulting_list + traj


    # long_trajectories = [traj for traj in trajectories if len(traj) >10]



    # with open(os.path.join(dir_name + '/' + 'samples20180429212517action_cte_rotation_s0_0'), 'rb') as infile:
    #     try:
    #         while True:
    #             transitions = pickle.load(infile)
    #     except EOFError as e:
    #         pass
    # print('Number of transitions added : ', len(transitions))
    # tuples = transitions
    # #correct
    # correct_tuples = list()
    # for tuple in tuples:
    #     correct_tuple = (tuple[0],(0.0,-0.5),tuple[2],tuple[3],tuple[4])
    #     correct_tuples.append(correct_tuple)

    replace_reward = reward.RewardMapper(plot_flag=False, r_mode_='step')
    replace_reward.set_boundary_points(buoys)
    replace_reward.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    point_a, point_b = replace_reward.get_guidance_line()
    replace_reward.set_shore_lines(upper_shore, lower_shore)
    #
    tuples = list()
    bundle_name = 'samples/samples_bundle_complete_action_b'
    with open(bundle_name,'rb') as file:
        tuples = pickle.load(file)
    # tuples = [tup for tup in tuples if tup[0][2]+103.5 < 20 and 7000 < tup[0][0] < 11500 and tup[0][3]<0]
    # print('Number of tuples to be considered:', len(tuples))
    random.shuffle(tuples)
    reduct_batch = tuples[:50000]
    tuples_with_reflection = list()
    for tuple in reduct_batch:

        reflect_tuple = reflect_tuple_on_line(point_a, point_b, tuple)
        if replace_reward.is_inbound_coordinate(reflect_tuple[0][0], reflect_tuple[0][1]):
            tuples_with_reflection.append(tuple)
            tuples_with_reflection.append(reflect_tuple)
    print('Number of tuples after reflection:', len(tuples_with_reflection))
    with open(bundle_name+'_filter_reflected_sim',
              'wb') as outfile:
        pickle.dump(tuples_with_reflection, outfile)
    # org_tuples = list()
    # with open('samples/samples_bundle_two_angles', 'rb') as file:
    #     org_tuples = pickle.load(file)

    # new_angle_tup = [tup for tup in tuples if tup[0][4] > -0.2]
    # selected_tuples = random.sample(org_tuples, 10000)
    # plot_sequence(selected_tuples)

    batch_learner = learner.Learner(r_m_=replace_reward, nn_=True)
    new_list = batch_learner.replace_reward(tuples_with_reflection)

    simple_state_tuples = list()

    for tuple in new_list:
        new_state = convert_state_space(tuple[0],replace_reward)
        new_state_p = convert_state_space(tuple[2], replace_reward)
        new_tuple = (new_state, tuple[1], new_state_p, tuple[3], tuple[4])
        simple_state_tuples.append(new_tuple)

    final = [tpl for tpl in simple_state_tuples if tpl[0][0] > 0]


    batch_learner.add_tuples(final)
    batch_learner.set_up_agent()
    batch_learner.fqi_step(5)

    for i in range(100):
        additional_tuples = experiment.run_episodes(batch_learner)
        converted_new_tuples = list()
        for tup in additional_tuples:
            new_state = convert_state_space(tup[0], replace_reward)
            new_state_p = convert_state_space(tup[2], replace_reward)
            new_tuple = (new_state, tup[1], new_state_p, tup[3], tup[4])
            converted_new_tuples.append(new_tuple)
            #repeat it so it gets more weight in learning
        for i in range(10):
            batch_learner.add_tuples(converted_new_tuples)
        batch_learner.set_up_agent()
        batch_learner.fqi_step(1)
    print('Finished')
