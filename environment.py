# -*- coding: utf-8 -*-
import blabla
import threading
import math
import actions
import reward
import buzz_python


class Environment(buzz_python.session_subscriber):
    def __init__(self):
        super(Environment, self).__init__()
        self.buoys = list()
        self.steps_between_actions = 10
        # self.vessel_id = '102'
        self.vessel_id = '36'
        self.rudder_id = '0'
        self.thruster_id = '0'
        self.mongo_addr = 'mongodb://10.1.1.92:27017'
        self.dbname = 'test'
        self.simulation_id = 'sim'
        # self.scenario = 'Santos_Container_L349B45'
        self.scenario = 'default'
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
        self.reward_mapper = reward.RewardMapper(True)
        self.init_pos = (-200, -300, 5)
        self.init_vel = (6, 0 ,0)

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
        self.simulation.set_current_time_increment(0.1)
        self.start()
        self.vessel = self.simulation.get_vessel(self.vessel_id)
        self.reset_state(self.init_pos[0], self.init_vel[0], self.init_pos[1], self.init_vel[1], self.init_pos[2],
                                                                                                self.init_vel[2])
        self.reward_mapper.update_ship_position(-200,-200,10)
        self.simulation.advance_time()
        self.advance()
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

    def get_state(self):
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

    def advance(self):
        self.allow_advance_ev.wait()
        self.simulation.advance_time()
        self.allow_advance_ev.clear()

    def step(self, action):
        """**Implement Here***
            The state transition. The agent executed the action in the parameter
        """
        cycle = 0
        for cycle in range(self.steps_between_actions):
            self.advance()
        print(action)
        rot_level, angle_level = actions.map_from_action(action)
        self.thruster.set_demanded_rotation(rot_level*self.max_rot)
        self.simulation.update(self.thruster)
        self.rudder.set_demanded_angle(angle_level*self.max_angle)
        self.simulation.update(self.rudder)
        statePrime = self.get_state() #Get next State

        self.reward_mapper.update_ship_position(statePrime[0],statePrime[1],statePrime[2])
        reward = self.reward_mapper.get_reward()
        print(self.reward_mapper.collided())
        if self.reward_mapper.collided():
            print("Collided!!!")
            self.reset_state(self.init_pos[0], self.init_vel[0], self.init_pos[1], self.init_vel[1], self.init_pos[2], self.init_vel[2])
            statePrime = self.get_state()  # Get next State
        return statePrime, action, reward

    def reset_state(self, x, vel_x, y, vel_y, theta, vel_theta):
        self.vessel.set_linear_position([x, y, 0.00])
        self.vessel.set_linear_velocity([vel_x, vel_y, 0.00])
        self.vessel.set_angular_position([0.00, 0.00, theta])
        self.vessel.set_angular_velocity([0.00, 0.00, vel_theta])
        self.simulation.sync(self.vessel)

    def __del__(self):
        if self.simulation:
            self.simulation.stop()
            self.simulation.disconnect()


if __name__ == "__main__":
    test = Environment()
    test.set_up()
    for i in range(100):
        ret = test.step(i)
        print(ret)