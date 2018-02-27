import numpy as np
from itertools import product
from shapely.geometry import Polygon, LineString
from shapely import affinity
from viewer import Viewer
from math import sin, cos, radians


class RewardMapper(object):
    def __init__(self, plot_flag=True):
        self.boundary = None
        self.goal_rec = None
        self.ship_polygon = None
        self.ship = None
        self.plot_flag = plot_flag
        self.view = Viewer()
        self.set_ship_geometry(((0,0),(10,10),(0,20)))
        # self.set_boundary_points(((-300,-400),(-300,-200),(-100,0), (20,200), (30,500), (30,700), (120,1000), \
        #                            (200, 1000), (140,700), (140,500), (90,200), (100,0), (-100,-200), (-100,-400)))
        # self.set_goal((150, 900))

    def generate_inner_positions(self):
        #TODO implement
        pass

    def set_boundary_points(self, points):
        self.boundary = Polygon(points)
        if self.plot_flag:
            self.view.plot_boundary(points)

    def set_ship_geometry(self, points):
        self.ship_polygon = Polygon(points)

    def set_goal(self, points):
        self.goal_rec = Polygon(((points[0] - 20, points[1] - 20), (points[0] - 20, points[1] + 20), (points[0] + 20, points[1] + 20),
                            (points[0] + 20, points[1] - 20)))
        if self.plot_flag:
            self.view.plot_goal(points)

    def update_ship_position(self, x, y, heading):
        self.ship = affinity.translate(self.ship_polygon, x, y)
        self.ship = affinity.rotate(self.ship, heading, 'center')
        if self.plot_flag:
            self.view.plot_position(x, y, heading)

    def get_shortest_distance_from_boundary(self):
        a = self.ship.distance(self.boundary)
        return a

    def collided(self):
        collided = (not self.boundary.contains(self.ship))
        return collided

    def reached_goal(self):
        #TODO incorporate velocity and heading
        ret = self.goal_rec.contains(self.ship)
        return ret

    def get_reward(self):
        reward = -0.1
        if self.collided():
            reward = -1
        elif self.reached_goal():
            reward = 1
        return reward

if __name__ == "__main__":
    reward_map = RewardMapper(True)
    reward_map.update_ship_position(20, 20, 30)
    print(reward_map.collided())
    # reward_map.update_ship_position(200, 200, 50)
    # reward_map.set_goal((100, 1000))
    for i in range(500):
        reward_map.update_ship_position(i, i, i)
        ret = reward_map.collided()
        if ret:
            print(ret)

#TODO Find a way to save Q table into file
#TODO Save configurations and Q table
