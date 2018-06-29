from shapely.geometry import Polygon, LineString, Point
from shapely import affinity
import numpy as np



class GeometryHelper(object):
    def __init__(self):
        self.goal_rec = None
        self.goal_margin = 100
        self.ship_polygon = None
        self.boundary = None
        self.guid_line = None
        self.upper_shore = None
        self.lower_shore = None
        self.ship = None
        self.points = None

    def set_ship_geometry(self, points):
        self.ship_polygon = Polygon(points)

    def set_goal_rec(self, center_x, center_y):
        self.goal_rec = Polygon(((center_x - self.goal_margin, center_y - self.goal_margin),
                                 (center_x - self.goal_margin, center_y + self.goal_margin),
                                 (center_x + self.goal_margin, center_y + self.goal_margin),
                                 (center_x + self.goal_margin, center_y - self.goal_margin)))

    def set_polygon_position(self, x, y, heading):
        self.ship = affinity.translate(self.ship_polygon, x, y)
        self.ship = affinity.rotate(self.ship, heading, 'center')

    def set_boundary_points(self, points):
        self.boundary = Polygon(points)

    def set_shore_lines(self, upper_points, lower_points):
        self.upper_shore = LineString(upper_points)
        self.lower_shore = LineString(lower_points)

    def get_shore_balance(self, x, y):
        ship_point = Point((x, y))
        # upper means positive sign
        upper_dist = ship_point.distance(self.upper_shore)
        lower_dist = ship_point.distance(self.lower_shore)
        return upper_dist - lower_dist

    def get_shortest_distance_from_boundary(self):
        a = self.ship.distance(self.boundary)
        return a

    def set_guidance_line(self):
        y_temp_a = get_middle_y(self.boundary, 8000)
        y_temp_b = get_middle_y(self.boundary, 9000)
        x_a = 3600
        x_b = 14000
        m, b = np.polyfit([8000, 9000], [y_temp_a,y_temp_b], 1)
        y_a = m*x_a+b
        y_b = m*x_b+b
        self.points = [(x_a, y_a), (x_b, y_b)]
        self.guid_line = LineString([(x_a, y_a), (x_b, y_b)])

    def get_simmetry_points(self):
        return self.points

    def ship_collided(self):
        contains = self.boundary.contains(self.ship)
        return not contains

    def reached_goal(self):
        return self.goal_rec.contains(self.ship)


def get_middle_y(boundary, x):
    line = LineString([(x, 0), (x, 15000)])
    intersect = boundary.intersection(line)
    return (intersect.bounds[1] + intersect.bounds[3]) / 2


def generate_inner_positions(goal_point, boundary):
    points_dict = dict()
    for line_x in range(int(goal_point[0] + 5000), int(goal_point[0] + 6000), 500):
        line = LineString([(line_x, 0), (line_x, 15000)])
        intersect = boundary.intersection(line)
        if intersect.geom_type == 'LineString':
            middle_point = (line_x, (intersect.bounds[1] + intersect.bounds[3]) / 2)
            upper_point = (line_x, (middle_point[1] + intersect.bounds[3]) / 2)
            lower_point = (line_x, (middle_point[1] + intersect.bounds[1]) / 2)
            dist = Point(goal_point).distance(Point(middle_point))
            points_dict[middle_point] = dist
            points_dict[upper_point] = dist
            points_dict[lower_point] = dist
    return points_dict


def is_inbound_coordinate(boundary, x, y):
    return boundary.buffer(-20).contains(Point(x, y))
