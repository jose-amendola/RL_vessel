import turtle
import math
import tkinter


class Viewer(object):
    def __init__(self):
        turtle.speed(0)
        turtle.mode('logo')
        #turtle.shapesize(stretch_len=24.4745, stretch_wid=2.25)
        turtle.setworldcoordinates(-1000, -1000, 1000, 1000)
        turtle.setup()
        turtle.screensize(4000, 1000, 'white')
        #cv = screen.getcanvas()
        #turtle.screensize(30000, 1000, 'white')
        self.l_vessel = 50 # Metade do comprimento da embarcacao
        w_vessel = 5 # Metade da largura da embarcacao
        turtle.register_shape('vessel',((0,self.l_vessel),(w_vessel,self.l_vessel/2),(w_vessel,-self.l_vessel),(-w_vessel,-self.l_vessel),(-w_vessel,self.l_vessel/2)))
        turtle.register_shape('rudder',((-1,0),(1,0),(1,-10),(-1,-10)))
        self.vessel = turtle.Turtle()
        self.vessel.shape('vessel')
        self.vessel.fillcolor('red')
        self.vessel.penup()
        self.rudder=turtle.Turtle()
        self.rudder.shape('rudder')
        self.rudder.fillcolor('green')
        self.rudder.penup()
        self.step_count = 0
        self.steps_for_stamp = 30
        turtle.degrees()
        # turtle.tracer(0, 0)s

    def plot_position(self, x, y , theta, rud_angle):
        converted_angle = 90 - theta
        #turtle.fillcolor('green')
        self.vessel.setpos(x, y)
        self.vessel.setheading(converted_angle)
        self.vessel.pendown()
        self.rudder.setpos(x-2*self.l_vessel*math.cos(math.pi*converted_angle/180), y-2*self.l_vessel*math.sin(math.pi*converted_angle/180))
        self.rudder.setheading(converted_angle+rud_angle)
        # if self.step_count == self.steps_for_stamp:
        #     turtle.stamp()
        #     self.step_count = 0
        # else:
        #     self.step_count += 1
        #turtle.stamp()

    def plot_guidance_line(self, point_a, point_b):
        self.vessel.setpos(point_a[0], point_a[1])
        self.vessel.pendown()
        self.vessel.setpos(point_b[0], point_b[1])
        self.vessel.penup()


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
    
    def end_of_episode(self):
        self.vessel.penup()

    # def __del__(self):
        # canvasvg.saveall("image.svg", turtle.getcanvas())

if __name__=='__main__':
    viewer = Viewer()
    viewer.plot_guidance_line((0,0),(500,0))
