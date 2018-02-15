import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

class InteractiveCircle(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('equal')

        self.circ = Circle((0.05, 0.05), 0.001)
        self.ax.add_artist(self.circ)
        self.ax.set_title('Click to move the circle')

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is None:
            return
        self.circ.center = event.xdata, event.ydata
        start_time = time.time()
        self.fig.canvas.draw()
        print("%f" %(time.time() - start_time))

    def show(self):
        plt.show()


InteractiveCircle().show()