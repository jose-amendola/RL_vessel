import turtle

class Viewer(object):
    def __init__(self):

        scr = turtle.Screen()
        turtle.mode('standard')
        xsize, ysize = scr.screensize()
        turtle.setworldcoordinates(0, 0, xsize, ysize)

        turtle.hideturtle()
        turtle.speed('fastest')
        turtle.tracer(0, 0)
        turtle.penup()


    def plot_position(self, x, y , theta):
        turtle.setpos(x, y)
        turtle.ship.setheading(theta)

    def plot_goal(self):
        pass

    def plot_boundary(self):
        pass

