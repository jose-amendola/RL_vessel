import turtle
import tkinter


class Viewer(object):
    def __init__(self):
        turtle.speed(0)
        turtle.mode('logo')
        turtle.setworldcoordinates(5000, 2000, 14000, 7000)
        turtle.degrees()
        turtle.penup()
        self.step_count = 0
        self.steps_for_stamp = 10
        # turtle.tracer(0, 0)s

    def plot_position(self, x, y , theta):
        converted_angle = 180 - theta
        turtle.fillcolor('gray')
        turtle.setpos(x, y)
        # if self.step_count == self.steps_for_stamp:
        #     turtle.stamp()
        #     self.step_count = 0
        # else:
        #     self.step_count += 1
        turtle.stamp()
        turtle.setheading(converted_angle)
        # turtle.pendown()

    def plot_goal(self,point, factor):
        turtle.speed(0)
        turtle.setpos(point[0] - factor, point[1] - factor)
        turtle.pendown()
        turtle.fillcolor('red')
        turtle.begin_fill()
        turtle.setpos(point[0] - factor, point[1] + factor)
        turtle.setpos(point[0] + factor, point[1] + factor)
        turtle.setpos(point[0] + factor, point[1] - factor)
        turtle.end_fill()
        turtle.penup()


    def plot_boundary(self, points_list):
        turtle.speed(0)
        turtle.setpos(points_list[0][0], points_list[0][1])
        turtle.pendown()
        turtle.fillcolor('blue')
        turtle.begin_fill()
        for point in points_list:
            turtle.setpos(point[0], point[1])
        turtle.end_fill()
        turtle.penup()

    def freeze_screen(self):
        tkinter.mainloop()

    # def __del__(self):
        # canvasvg.saveall("image.svg", turtle.getcanvas())


