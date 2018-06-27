from geometry_helper import is_inbound_coordinate
from simulation_settings import *
import pickle
import os
import reward
import learner
from viewer import Viewer
from planar import Vec2
from planar.line import Line, LineSegment
import utils
import random
import experiment


def replace_reward(transition_list, mp):
    new_list = list()
    first_state = transition_list[0][0]
    mp.initialize_ship(first_state[0], first_state[1], first_state[2], first_state[3],
                       first_state[4], first_state[5])
    for transition in transition_list:
        resulting_state = transition[2]
        action_selected = transition[1]
        mp.update_ship(resulting_state[0], resulting_state[1], resulting_state[2], resulting_state[3],
                       resulting_state[4], resulting_state[5], action_selected[0], action_selected[1])
        new_reward = mp.get_reward()
        ret = 0
        if geom_helper.ship_collided():
            ret = -1
        elif geom_helper.reached_goal():
            ret = 1
        print("Final step:", ret)
        tmp = list(transition)
        tmp[3] = new_reward
        tmp[4] = ret
        transition = tuple(tmp)
        new_list.append(transition)
    print(new_list[-1])
    return new_list


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


def plot_sequence(tuples):
    view = Viewer()
    view.plot_boundary(buoys)
    view.plot_goal(goal, goal_factor)
    for tuple in tuples:
        state = tuple[0]
        print('Position: ', state[0], state[1], state[2])
        print('Velocities: ', state[3], state[4], state[5])
        print('Action: ', tuple[1])
        view.plot_position(state[0], state[1], state[2])
    view.freeze_screen()


def mirror_point(point_a, point_b, point_to_mirror):
    vec_a = Vec2(point_a[0], point_a[1])
    vec_b = Vec2(point_b[0] - point_a[0], point_b[1] - point_a[1])
    line = Line(vec_a, vec_b)
    org_vec = Vec2(point_to_mirror[0], point_to_mirror[1])
    reflect_vec = line.reflect(org_vec)
    return reflect_vec.x, reflect_vec.y


def mirror_velocity(vel_x, vel_y, heading_n_cw, new_heading):
    v_lon, v_drift, not_used = utils.global_to_local(vel_x, vel_y, heading_n_cw)
    new_v_x, new_v_y, not_used = utils.local_to_global(v_lon, - v_drift, 90 - new_heading)
    return new_v_x, new_v_y


def reflect_state_across_line(point_a, point_b, state):
    line_angle = - 103.5
    x = tuple[0][0]
    y = tuple[0][1]
    new_x, new_y = mirror_point(point_a, point_b, (state[0], state[1]))
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
    return reflected_state, reflected_action, reflected_state_p, tuple[3], tuple[4]


def get_strictly_simetric_set(org_tuples, mapper, point_a, point_b):
    tuples = [tup for tup in org_tuples if tup[0][2] + 103.5 < 20 and tup[0][0] < 11500]
    print('Number of tuples to be considered:', len(tuples))
    tuples_with_reflection = list()
    for tuple in tuples:
        reflect_tuple = reflect_tuple_on_line(point_a, point_b, tuple)
        if is_inbound_coordinate(mapper.boundary, reflect_tuple[0][0], reflect_tuple[0][1]):
            tuples_with_reflection.append(tuple)
            tuples_with_reflection.append(reflect_tuple)
    print('Number of tuples after reflection:', len(tuples_with_reflection))


if __name__ == '__main__':
    rew = reward.RewardMapper(r_mode_='quadratic')
    rew.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    #
    tuples = list()
    bundle_name = 'samples/samples_bundle_new'
    with open(bundle_name, 'rb') as file:
        tuples = pickle.load(file)

    filtered = [tpl for tpl in tuples if tpl[0][3] < 0]

    reduct_batch = random.sample(filtered, 1000)
    new_list = replace_reward(reduct_batch, rew)
    simple_state_tuples = list()

    for tuple in new_list:
        new_state = utils.convert_to_simple_state(tuple[0])
        new_state_p = utils.convert_to_simple_state(tuple[2])
        new_tuple = (new_state, tuple[1], new_state_p, tuple[3], tuple[4])
        simple_state_tuples.append(new_tuple)

    batch_learner = learner.Learner(nn_=True)
    batch_learner.add_tuples(simple_state_tuples)
    batch_learner.set_up_agent()
    batch_learner.fqi_step(5)

    for i in range(500):
        additional_tuples = experiment.run_episodes(batch_learner)
        os.chdir('..')
        converted_new_tuples = list()
        for tup in additional_tuples:
            new_state = utils.convert_to_simple_state(tup[0])
            new_state_p = utils.convert_to_simple_state(tup[2])
            new_tuple = (new_state, tup[1], new_state_p, tup[3], tup[4])
            converted_new_tuples.append(new_tuple)
            # repeat it so it gets more weight in learning
        sampling = random.sample(converted_new_tuples, 100)
        batch_learner.add_tuples(converted_new_tuples)
        batch_learner.set_up_agent()
        batch_learner.fqi_step(5)
    print('Finished')
