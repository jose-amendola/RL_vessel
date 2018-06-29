import numpy as np
import pickle
import os


def local_to_global(x_local, y_local, heading_e_ccw):
    heading_n_cw = 90 - heading_e_ccw
    theta = np.radians(-heading_e_ccw)
    c, s = np.cos(theta), np.sin(theta)
    m = np.matrix(((c, -s), (s, c)))
    local_array = np.matrix([[x_local, y_local]]).T
    transformed = m * local_array
    #TODO Check why y signal goes inverted
    return transformed.item(0), -transformed.item(1), heading_n_cw


def global_to_local(x_global, y_global, heading_n_cw):
    heading_e_ccw = 90 - heading_n_cw
    theta = np.radians(heading_e_ccw)
    c, s = np.cos(theta), np.sin(theta)
    m = np.matrix(((c, -s), (s, c)))
    global_array = np.matrix([[x_global, y_global]]).T
    transformed = m * global_array
    return transformed.item(0), transformed.item(1), heading_e_ccw


def channel_angle_e_ccw(point_a, point_b):
    line = np.array(point_b) - np.array(point_a)
    support = np.array((point_a[0]+10,point_a[1])) - np.array(point_a)
    c = np.dot(line, support) / np.linalg.norm(line) / np.linalg.norm(support)  # -> cosine of the angle
    angle = np.arccos(np.clip(c, -1, 1))
    return 360 - np.rad2deg(angle)


def merge_sample_files_into_one(dir_name):
    files = os.listdir(dir_name)
    batch_list = list()
    for file in files:
        with open(os.path.join(dir_name, file), 'rb') as infile:
            if file.startswith('samples'):
                trajectory = list()
                print('Loading file:', file)
                try:
                    while True:
                        transitions = pickle.load(infile)
                except EOFError as e:
                    pass
                trajectory = trajectory + transitions
        print('Number of transitions added : ', len(transitions))
        batch_list = batch_list + transitions
    with open(dir_name + '/samples_bundle', 'wb') as bundle_file:
        pickle.dump(batch_list, bundle_file)


def convert_to_simple_state(state, g_helper):
    v_lon, v_drift, n_used = global_to_local(state[3], state[4], state[2])
    bl = g_helper.get_shore_balance(state[0], state[1])
    misalign = state[2] + 103.5
    theta_p = state[5]
    return v_lon, misalign, bl, theta_p



if __name__ == "__main__":
    merge_sample_files_into_one('C:\\Users\\jose_amendola\\RL_vessel\\samples')
    # print(global_to_local(-2.91453, -0.6997167, -103.5))
    # print(global_to_local(-1,-1,-135))
    # print(local_to_global(1.41, 0, 225))
    # N03 = (9191.6506, 4967.8532)
    # N05 = (6897.7712, 4417.3295)
    # print(channel_angle_e_ccw(N03,N05))
    # print(local_to_global(1.5, 0, 194))