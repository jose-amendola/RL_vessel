import turtle
import asyncio

class Viewer(object):
    def __init__(self):
        turtle.speed(0)
        turtle.mode('logo')
        turtle.setworldcoordinates(-1000,-1000,1200,1200)
        turtle.degrees()
        turtle.penup()
        # turtle.tracer(0, 0)s

    def plot_position(self, x, y , theta):
        converted_angle = 180 - theta
        turtle.fillcolor('gray')
        turtle.setpos(x, y)
        turtle.stamp()
        turtle.setheading(converted_angle)

    def plot_goal(self,point):
        turtle.speed(0)
        turtle.setpos(point[0],point[1])
        turtle.pendown()
        turtle.fillcolor('red')
        turtle.begin_fill()
        turtle.circle(20)
        turtle.end_fill()
        turtle.penup()


    def plot_boundary(self, points_list):
        pass
        turtle.speed(0)
        turtle.setpos(points_list[0][0], points_list[0][1])
        turtle.pendown()
        turtle.fillcolor('blue')
        turtle.begin_fill()
        for point in points_list:
            turtle.setpos(point[0], point[1])
        turtle.end_fill()
        turtle.penup()

