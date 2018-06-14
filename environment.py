# -*- coding: utf-8 -*-
import blabla
import threading
import math
import reward
import buzz_python
import itertools
import utils
import random
import time
import numpy as np
import subprocess
import os
from simulation_settings import *
import pickle
import random



class Environment(buzz_python.session_subscriber):
    def __init__(self, _buoys_list, _step, _vessel_id, _rudder_id, _thr_id, _scn, _goal, _g_heading, _g_vel_l,
                 _increment=0.1):
        super(Environment, self).__init__()
        self.step_increment = _increment
        self.buoys = _buoys_list
        self.goal = _goal
        self.g_heading = _g_heading
        self.g_vel_l = _g_vel_l
        self.steps_between_actions = _step
        # self.vessel_id = '102'
        self.vessel_id = _vessel_id
        self.rudder_id = _rudder_id
        self.thruster_id = _thr_id
        # self.mongo_addr = 'mongodb://10.1.1.92:27017'
        # self.dbname = 'test'
        self.simulation_id = 'sim'
        # self.scenario = 'Santos_Container_L349B45'
        self.scenario = _scn
        self.control_id = '555'
        self.chat_address = '127.0.0.1'
        self.simulation = []
        self.dyna_ctrl_id = '407'
        self.allow_advance_ev = threading.Event()
        self.dyna_ready_ev = threading.Event()
        self.own = []
        self.vessel = []
        self.rudder = []
        self.thruster = []
        self.max_angle = 0
        self.max_rot = 0
        self.reward_mapper = reward.RewardMapper(r_mode_='step')
        self.init_state = list()
        self._final_flag = False
        self.initial_states_sequence = None
        self.reward_mapper.set_boundary_points(self.buoys)
        self.reward_mapper.set_goal(self.goal, self.g_heading, self.g_vel_l)
        self.reward_mapper.get_guidance_line()
        self.reward_mapper.set_shore_lines(upper_shore, lower_shore)
        os.chdir('./dyna')
        self.dyna_proc = None
        self.accumulated_starts = list()


    def get_sample_states(self):
        #TODO implement
        x = np.linspace(5000, 13000, 16)
        y = np.linspace(3000, 8000, 100)
        # theta = np.linspace(-90, -120, 4)
        # theta = np.append(theta, -103)
        vel_decay = 4/13000
        theta = -103
        # vlon = np.linspace(1.5, 3.0, 4)
        g = np.meshgrid(x, y, theta, vlon)
        tmp = np.vstack(map(np.ravel, g))
        combinations = np.transpose(tmp)
        states = list()
        for comb in combinations:
            if self.reward_mapper.is_inbound_coordinate(comb[0], comb[1]):
                states.append((comb[0], comb[1], comb[2], comb[3], 0, 0))
        return states

    def create_variants_to_start(self, local_coord_start):
        start_variants  = list()
        x = np.linspace(-20, 20, 5)
        y = np.linspace(-20, 20, 5)
        theta = np.linspace(-5, +5, 3)
        vlon = np.linspace(-0.2, +0.2, 3)
        vdrift = np.linspace(-0.1, +0.1, 3)
        theta_p = np.linspace(-0.1, +0.1, 3)
        g = np.meshgrid(x, y, theta, vlon, vdrift, theta_p)
        tmp = np.vstack(map(np.ravel, g))
        shifts = np.transpose(tmp)
        for shift in shifts:
            if self.reward_mapper.is_inbound_coordinate(shift[0]+local_coord_start[0], shift[1]+local_coord_start[1]):
                gl_vars = [(org+shift) for org, shift in zip(shift, local_coord_start)]
                start_variants.append(tuple(gl_vars))
        return start_variants

    def add_states_to_start_list(self, global_coord_state):
        print("Adding states to start list.")
        v_lon, v_drift, not_used = utils.global_to_local(global_coord_state[3],global_coord_state[4], global_coord_state[2])
        new_start_state = (global_coord_state[0], global_coord_state[1], global_coord_state[2],
                           v_lon, v_drift, global_coord_state[5])
        variants_list = self.create_variants_to_start(new_start_state)
        variants_list.append(new_start_state)
        self.accumulated_starts.append(new_start_state)
        self.accumulated_starts = self.accumulated_starts + variants_list

    def get_initial_states(self):
        positions_dict = self.reward_mapper.generate_inner_positions()
        init_angle = -100
        states_list = list()
        init_vel_l = 6
        for position in positions_dict:
            for count in range(10):
                state = (position[0], position[1],
                         init_angle*random.triangular(0.8, 1.2),
                         positions_dict[position]/4000*init_vel_l*random.triangular(0.8, 1.2),
                         0, 0)
                states_list.append(state)
        return states_list

    def is_final(self):
        ret = 0
        if self.reward_mapper.reached_goal():
            ret = 1
        elif self.reward_mapper.collided():
            ret = -1
        print("Final step:", ret)
        return ret

    def on_state_changed(self, state):
        if state == buzz_python.STANDBY:
            print("Dyna ready")
            self.dyna_ready_ev.set()

    def on_time_advanced(self, time):
        step = self.simulation.get_current_time_step()


    def on_time_advance_requested(self, process):
        if process.get_id() == self.dyna_ctrl_id:
            self.allow_advance_ev.set()

    def set_up(self):
        self.dyna_proc = subprocess.Popen(['Dyna_reset_prop.exe ','--pid', '407', '-f', 'suape-local.json', '-c', '127.0.0.1'])
        # ds = buzz_python.create_bson_data_source(self.mongo_addr, self.dbname)
        ds = buzz_python.create_bson_data_source('suape-local.json')
        ser = buzz_python.create_bson_serializer(ds)
        self.simulation = buzz_python.create_simco_simulation(self.simulation_id, self.control_id, ser)
        self.simulation.connect(self.chat_address)
        self.init(self.simulation)
        scn = ser.deserialize_scenario(self.scenario)
        self.simulation.add(scn)
        factory = self.simulation.get_element_factory()
        r_c = factory.build_runtime_component(self.dyna_ctrl_id)
        self.simulation.add(r_c)
        self.own = self.simulation.get_runtime_component(self.control_id)

        self.simulation.build()
        self.simulation.set_current_time_increment(self.step_increment)
        self.start()
        self.vessel = self.simulation.get_vessel(self.vessel_id)

        # self.reset_state(self.init_state[0], self.init_state[1], self.init_state[2],
        #                      self.init_state[3], self.init_state[4], self.init_state[5])
        # self.reward_mapper.update_ship(-200, -200, 10,, 0, 0
        self.simulation.advance_time()
        self.advance()
        # self.initial_states_sequence = itertools.cycle(self.get_initial_states())

        # self.init_state = next(self.initial_states_sequence)
        self.advance()
        self.get_propulsion()

    def start(self):
        self.dyna_ready_ev.wait()
        self.simulation.start()
        self.dyna_ready_ev.clear()

    def get_propulsion(self):
        thr_tag = buzz_python.thruster_tag()
        self.simulation.is_publisher(thr_tag, thr_tag.SMH_DEMANDED_ROTATION)

        rdr_tag = buzz_python.rudder_tag()
        self.simulation.is_publisher(rdr_tag, rdr_tag.SMH_DEMANDED_ANGLE)

        thr_ev_taken_tag = buzz_python.thruster_control_taken_event_tag()
        self.simulation.is_publisher(thr_ev_taken_tag, True)

        rdr_ev_taken_tag = buzz_python.rudder_control_taken_event_tag()
        self.simulation.is_publisher(rdr_ev_taken_tag, True)

        self.thruster = self.vessel.get_thruster(self.thruster_id)
        ctrl_ev = buzz_python.create_thruster_control_taken_event(self.thruster)
        ctrl_ev.set_controlled_fields(thr_tag.SMH_DEMANDED_ROTATION)
        ctrl_ev.set_controller(self.own)
        self.simulation.publish_event(ctrl_ev)

        self.rudder = self.vessel.get_rudder(self.rudder_id)
        ctrl_ev_r = buzz_python.create_rudder_control_taken_event(self.rudder)
        ctrl_ev_r.set_controlled_fields(rdr_tag.SMH_DEMANDED_ANGLE)
        ctrl_ev_r.set_controller(self.own)
        self.simulation.publish_event(ctrl_ev_r)
        self.simulation.update(self.thruster)
        self.max_rot = self.thruster.get_maximum_rotation()
        self.simulation.update(self.rudder)
        self.max_angle = self.rudder.get_maximum_angle()

    def get_state(self, state_mode=None):
        self.simulation.sync(self.vessel)
        lin_pos_vec = self.vessel.get_linear_position()
        ang_pos_vec = self.vessel.get_angular_position()
        lin_vel_vec = self.vessel.get_linear_velocity()
        ang_vel_vec = self.vessel.get_angular_velocity()
        x = lin_pos_vec[0]
        y = lin_pos_vec[1]
        theta = ang_pos_vec[2]
        xp = lin_vel_vec[0]
        yp = lin_vel_vec[1]
        thetap = ang_vel_vec[2]
        return x, y, theta, xp, yp, thetap

    def convert_to_simple_state(self, state):
        v_lon, v_drift, n_used = utils.global_to_local(state[3], state[4], state[2])
        bl = self.reward_mapper.get_shore_balance(state[0], state[1])
        misalign = state[2] + 103.5
        return (v_lon, misalign, bl)


    def advance(self):
        self.allow_advance_ev.wait()
        self.simulation.advance_time()
        self.allow_advance_ev.clear()

    def step(self, angle_level, rot_level):
        """**Implement Here***
            The state transition. The agent executed the action in the parameter
            :param rot_level:
            :param angle_level:
        """

        print('Rotation level: ', rot_level)
        print('Angle level: ', angle_level)
        self.thruster.set_demanded_rotation(rot_level*self.max_rot)
        self.simulation.update(self.thruster)
        self.rudder.set_demanded_angle(angle_level*self.max_angle)
        self.simulation.update(self.rudder)

        cycle = 0
        for cycle in range(self.steps_between_actions):
            self.advance()
        statePrime = self.get_state() #Get next State
        print('statePrime: ', statePrime)
        self.reward_mapper.update_ship(statePrime[0], statePrime[1], statePrime[2], statePrime[3], statePrime[4],
                                       statePrime[5], angle_level, rot_level)
        rw = self.reward_mapper.get_reward()
        print(self.reward_mapper.collided())
        if self.reward_mapper.collided():
            print("Collided!!!")
            # self.init_state = next(self.initial_states_sequence)
            # self.reset_state_localcoord(self.init_state[0], self.init_state[1], self.init_state[2], self.init_state[3],
            #                             self.init_state[4], self.init_state[5])
            # statePrime = self.get_state()  # Get next State
        return statePrime, rw

    def start_bifurcation_mode(self):
        random.shuffle(self.accumulated_starts)
        self.initial_states_sequence = itertools.cycle(self.accumulated_starts)
        print('Total of {} starting points generated.'.format(len(self.accumulated_starts)))
        with open('samples/starting_points_global_coord'+timestamp,'wb') as starts_file:
            pickle.dump(self.accumulated_starts, starts_file)

    def move_to_next_start(self):
        self.init_state = next(self.initial_states_sequence)

    def starts_from_file_mode(self, file_name):
        start_list = list()
        with open(file_name, 'rb') as infile:
            print('Loading file:', file_name)
            try:
                while True:
                    start_list = pickle.load(infile)
            except EOFError as e:
                pass
        print('Number of transitions added : ', len(start_list))
        random.shuffle(start_list)
        self.initial_states_sequence = itertools.cycle(start_list)

    def set_single_start_pos_mode(self, init_state=None):
        if not init_state:
            org_state = self.get_state()
            vel_l, vel_drift, theta = utils.global_to_local(org_state[3],org_state[4],org_state[2])
            self.init_state = (org_state[0], org_state[1], org_state[2], vel_l, vel_drift, 0)
        else:
            self.init_state = (init_state[0], init_state[1], init_state[2], init_state[3], init_state[4], init_state[5])
        self.reset_state_localcoord(self.init_state[0], self.init_state[1], self.init_state[2], self.init_state[3],
                                    self.init_state[4], self.init_state[5])
        dummy_list = list()
        dummy_list.append(self.init_state)
        self.initial_states_sequence = itertools.cycle(dummy_list)

    def set_sampling_mode(self, start=0, end=-1):
        states = self.get_sample_states()
        print('#####TOTAL STARTING STATES AVAILABLE:{}'.format(len(states)))
        start = int(start)
        end = int(end)
        if len(states) > start:
            if len(states) > end != -1:
                states = states[start:end]
            else:
                states = states[start:]
        self.initial_states_sequence = itertools.cycle(states)

    def reset_to_start(self):
        self.reset_state_localcoord(self.init_state[0], self.init_state[1], self.init_state[2], self.init_state[3],
                                    self.init_state[4], self.init_state[5])

    def reset_state_localcoord(self, x, y, theta, vel_lon, vel_drift, vel_theta):
       #Apparently Dyna ADV is using theta n_cw
        self.vessel.set_linear_position([x, y, 0.00])
        self.vessel.set_linear_velocity([vel_lon, vel_drift, 0.00])
        self.vessel.set_angular_position([0.00, 0.00, theta])
        self.vessel.set_angular_velocity([0.00, 0.00, vel_theta])
        self.reward_mapper.initialize_ship(x, y, theta, vel_lon, vel_drift, vel_theta)
        self.simulation.sync(self.vessel)

    def finish(self):
        if self.simulation:
            self.simulation.stop()
            self.dyna_proc.terminate()

    def __del__(self):
        if self.simulation:
            self.simulation.stop()
            self.simulation.disconnect()


if __name__ == "__main__":
    test = Environment()
    test.set_up()
    for i in range(100):
        ret = test.step(0, 0)
        print(ret)