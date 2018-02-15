import numpy as np
from itertools import product
import pylab
from shapely.geometry import Polygon, LineString
from shapely import affinity

# ship = Polygon(((0,0),(0,15),(2,20),(4,15),(4, 0)))
# ship = affinity.scale(ship, 2, 3)

# ship = affinity.rotate(ship, 30, 'center')
# # ship = affinity.translate(ship, 10, 10)
# goal = (150,900)
# goal_rec = Polygon(((goal[0]-20, goal[1]-20),(goal[0]-20, goal[1]+20),(goal[0]+20, goal[1]+20),(goal[0]+20, goal[1]-20)))
# # pylab.ion()
# pylab.xlim([-1000,1000])
# pylab.ylim([-1000,1000])
# boundary_points = [(-300,-200),(2,0), (20,200), (30,500), (30,700), (120,1000)]
# decoupled = list(zip(*boundary_points))
# boundary = LineString(boundary_points)
#
# boundary_points_b = [(-100,-200),(100,0), (90,200), (140,500), (140,700), (200,1000)]
# decoupled_b = list(zip(*boundary_points_b))
# boundary_b = LineString(boundary_points_b)
#
# pylab.plot(decoupled[0], decoupled[1], color='#666666', aa=True, lw=1.0)
# pylab.plot(decoupled_b[0], decoupled_b[1], color='#666666', aa=True, lw=1.0)
#
# a = np.asarray(goal_rec.exterior)
# pylab.fill(a[:,0], a[:,1], 'r')
#
# a = np.asarray(ship.exterior)
# pylab.fill(a[:,0], a[:,1], 'c')
#
# pylab.draw()
#
# print(ship.distance(boundary))
# print(ship.crosses(boundary))
# pylab.plot(decoupled[0], decoupled[1], color='#666666', aa=True, lw=1.0)
# a = np.asarray(ship.exterior)
# pylab.fill(a[:,0], a[:,1], 'c')
#
# ship = affinity.translate(ship, 10, 10)
# print(ship.distance(boundary))
# print(ship.crosses(boundary))
# pylab.draw()
#
# pylab.pause(1)
# a = np.asarray(ship.exterior)
# pylab.fill(a[:,0], a[:,1], 'c')
#
# pylab.draw()
# pylab.pause(1)
#
# pylab.show()
#
# def plot_scenario():
#     pass
# def plot_vessel():
#     pass
#
# def get_distance_from_boundaries():
#     a = ship.distance(boundary)
#     b = ship.distance(boundary_b)
#     return a,b
#
# def get_distance_from_goal():
#     return ship.distance(goal_rec)
#
# def collided():
#     return ship.crosses(boundary) || ship.crosses(boundary_b)

class RewardMapper(object):
    def __init__(self, plot_flag_=True):
        pylab.xlim([-1000, 1000])
        pylab.ylim([-1000, 1000])
        pylab.ion()
        self.plot_flag = plot_flag_
        self.ship = None
        self.set_ship_geometry(((0,0),(0,15),(2,20),(4,15),(4, 0)))
        self.goal_rec = None
        self.set_goal((150, 900))
        self.boundary = None
        self.boundary_b = None
        self.set_boundary_points([(-300,-200),(2,0), (20,200), (30,500), (30,700), (120,1000)],
                                 [(-100,-200),(100,0), (90,200), (140,500), (140,700), (200,1000)])

    def set_boundary_points(self, list_a, list_b):
        self.boundary = LineString(list_a)
        self.boundary_b = LineString(list_b)
        if self.plot_flag:
            decoupled = list(zip(*list_a))
            decoupled_b = list(zip(*list_b))
            pylab.plot(decoupled[0], decoupled[1], color='#666666', aa=True, lw=1.0)
            pylab.plot(decoupled_b[0], decoupled_b[1], color='#666666', aa=True, lw=1.0)
            pylab.draw()


    def set_ship_geometry(self, points):
        self.ship = Polygon(points)

    def set_goal(self, points):
        self.goal_rec = Polygon(((points[0] - 20, points[1] - 20), (points[0] - 20, points[1] + 20), (points[0] + 20, points[1] + 20),
                            (points[0] + 20, points[1] - 20)))
        if self.plot_flag:
            a = np.asarray(self.goal_rec.exterior)
            pylab.fill(a[:, 0], a[:, 1], 'r')
            pylab.draw()


    def update_ship_position(self, x, y, heading):
        self.ship = affinity.translate(self.ship, x, y)
        self.ship = affinity.rotate(self.ship, heading, 'center')
        if self.plot_flag:
            a = np.asarray(self.ship.exterior)
            pylab.fill(a[:, 0], a[:, 1], 'c')
            pylab.draw()

            pylab.show()

    def get_distance_from_boundaries(self):
        a = self.ship.distance(self.boundary)
        b = self.ship.distance(self.boundary_b)
        return a, b

    def collided(self):
        return self.ship.crosses(self.boundary) or self.ship.crosses(self.boundary_b)

    def reached_goal(self):
        return self.ship.crosses(self.goal_rec)

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
    reward_map.update_ship_position(200, 200, 50)

#TODO Find a way to save Q table into file
#TODO Save configurations and Q table
