import time
import numpy as np
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from collections import deque
from bokeh.io import push_notebook
from bokeh.models.glyphs import Rect, Line, Ellipse
from bokeh.models.renderers import GlyphRenderer
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Label
from bokeh.palettes import viridis
from bokeh.plotting import figure


def none_func():
    return None


class PendelWagenSystem:
    def __init__(self, car_width=0.1, car_height=0.05, rod_length=0.5, pendulum_size=0.05):
        self.car_width = car_width
        self.car_height = car_height

        self.rod_length = rod_length
        self.pendulum_size = pendulum_size

        self.pendulum = GlyphRenderer(data_source=ColumnDataSource(dict(x=[0], y=[0])),
                                      glyph=Ellipse(x='x', y='y', width=self.pendulum_size, height=self.pendulum_size))

        self.car = GlyphRenderer(data_source=ColumnDataSource(dict(x=[0], y=[0])),
                                 glyph=Rect(x='x', y='y', width=self.car_width, height=self.car_height))

        self.rod = GlyphRenderer(data_source=ColumnDataSource(dict(x=[0,0], y=[0,0])),
                                 glyph=Line(x='x', y='y', line_width=2))

        self.move = GlyphRenderer(data_source=ColumnDataSource(dict(x=deque([0,1], maxlen=200), y=deque([0,1], maxlen=200))),
                                  glyph=Ellipse(x='x', y='y', width=0.008, height=0.008,
                                                fill_alpha=0.25, line_alpha=0.0, fill_color="#cc0000"))

        self.ground = GlyphRenderer(data_source=ColumnDataSource(dict(x=[-100, 100], y=[-car_height/2, -car_height/2])),
                                    glyph=Line(x='x', y='y', line_width=1, line_alpha=0.5))

    def draw(self, state):
        phi, q = state[[0, 1]]
        pendulum_x = -self.rod_length * np.sin(phi) + q
        pendulum_y = self.rod_length * np.cos(phi)
        self.pendulum.data_source.data['x'] = [pendulum_x]
        self.pendulum.data_source.data['y'] = [pendulum_y]
        self.car.data_source.data['x'] = [q]
        self.rod.data_source.data['x'] = [q, pendulum_x]
        self.rod.data_source.data['y'] = [0, pendulum_y]

        new_move_data = dict(x=self.move.data_source.data['x'] + deque([pendulum_x]),
                             y=self.move.data_source.data['y'] + deque([pendulum_y]))

        self.move.data_source.data.update(new_move_data)

    @property
    def glyphs(self):
        return [self.car, self.rod,self.pendulum, self.move, self.ground]


class MehrMassenSystem:
    def __init__(self, body_width=0.5, body_height=0.25, num_bodies=2, body_distance=1.0):
        self.body_width = body_width
        self.body_height = body_height
        self.num_bodies = num_bodies
        self.rod_colors = viridis(51)
        self.body_distance = body_distance

        self.bodies = []
        self.rods = []

        for _ in range(num_bodies):
            self.bodies.append(GlyphRenderer(data_source=ColumnDataSource(dict(x=[0], y=[0])),
                                             glyph=Rect(x='x', y='y', width=self.body_height, height=self.body_width)))

        for _ in range(num_bodies-1):
            self.rods.append(GlyphRenderer(data_source=ColumnDataSource(dict(x=[0, 0], y=[0, 0])),
                                           glyph=Line(x='x', y='y', line_width=2, line_color=self.rod_colors[0])))

        self.ground = GlyphRenderer(data_source=ColumnDataSource(dict(x=[-100, 100], y=[-self.body_height/2, -self.body_height/2])),
                                    glyph=Line(x='x', y='y', line_width=1, line_alpha=0.5))

    def draw(self, state):
        #body_state = np.array(list(zip(*np.split(np.array(state), len(self.bodies)))))
        
        #body_state = np.add.accumulate(state[0:self.num_bodies])
        body_state = state[0:self.num_bodies]

        for i, body in enumerate(self.bodies):
            body.data_source.data['x'] = [body_state[i] + i*self.body_distance] 

        for i, rod in enumerate(self.rods):
            rod.data_source.data['x'] = [self.bodies[i].data_source.data['x'][0], self.bodies[i+1].data_source.data['x'][0]]
            rod.data_source.data['y'] = [0, 0]
            rod.glyph.line_color = self.rod_colors[int(np.clip(state[i+1]*25, a_min=0, a_max=50))]

    @property
    def glyphs(self):
        return [*self.bodies, self.ground, *self.rods]


class Animation:
    """
    Provides animation capabilities.

    Given a callable function that draws an image of the system state and simulation data
    this class provides a method to created an animated representation of the system.
    """

    def __init__(self, sim_data, image=PendelWagenSystem(), speed_factor=1, fig=None, state=1, x_range=None, wait=0.01):

        self.wait = wait
        
        self.set_sim_data(sim_data)
        self.set_image(image)

        self.figure = fig if fig else self.create_fig(state)
        
        self.k = int(self.x.shape[0] / (np.max(self.t)/self.wait))

        self.x = self.x[0:-1:self.k]
        self.t = self.t[0:-1:self.k]

        self.speed_factor = speed_factor

        self.time = Label(x=25, y=self.figure.plot_height - 50, x_units='screen', y_units='screen')
        self.figure.add_layout(self.time)
        self.figure.renderers += self.image.glyphs

    def animate(self, target, **kwargs):
        self.speed_factor = kwargs.get('speed_factor', self.speed_factor)
		
        speed_string = f'  ({self.speed_factor}x)' if not self.speed_factor == 1 else ""
        for i, t in enumerate(self.t):
            self.time.text = f't = {np.around(t, decimals=2):.3f}' + speed_string
            self.image.draw(self.x[i, :])
            
            if 'callback' in kwargs:
                kwargs['callback'](i, t, self.x[i, :])
            
            #if self.fps_sim_data > self.fps:
            #    time.sleep(1/(self.fps_sim_data - self.fps))

            time.sleep(self.wait / self.speed_factor)

            push_notebook(handle=target)
    
    def set_sim_data(self, sim_data):
        self.sim_data = sim_data
        self.x, self.t = sim_data
        
        self.k = int(self.x.shape[0] / (np.max(self.t)/self.wait))

        self.x = self.x[0:-1:self.k]
        self.t = self.t[0:-1:self.k]
        
    def set_image(self, image):
        self.image = image
    
    def create_fig(self, state):
        # adapt plot ranges

        h = 0
        w = 1000

        while h == 0 or h > 800:

            w -= 100

            x_range = [np.min(self.x[:, state]) - 0.1, np.max(self.x[:, state]) + 0.1] if not self.x_range else self.x_range
            y_range = [-0.5 - 0.1, 0.5 + 0.1]

            h = int((np.diff(y_range)[0]/np.diff(x_range)[0])*w)

        return figure(x_range=x_range, y_range=y_range, width=w, height=h)

    def show(self, t):
        pass

    def set_limits(self):
        pass

    def set_labels(self):
        pass


def plot_sim_states(res, size=(12, 12), rad2deg=(True, False, True, False),
                    labels=(r"$\varphi$ in °", "$q$ in m",
                            r"$\dot{\varphi}$ in $\frac{°}{s}$", r"$\dot{q}$ in $\frac{m}{s}$")):

    states = []

    if rad2deg is not None:
        for i, x in enumerate(res[0].T):
            if rad2deg[i]:
                states.append(np.rad2deg(x))
            else:
                states.append(x)
    else:
        states = res[0].T.copy()

    tt = res[1]
    plt.figure(figsize=size)
    num_rows = len(res[0].T) // 2
    for i, X in enumerate(states):

        plt.subplot(num_rows, 2, i+1)
        plt.plot(tt, X)
        try:
            plt.ylabel(labels[i])
        except Exception:
            pass
        plt.xlabel("t in s")
        plt.grid()

    plt.tight_layout()


def plot_sample_distribution_scatter(S, fig=None, labels=("x", "y"), size=10, **kwargs):
    nullfmt = NullFormatter()  # no labels
    fig = plt.figure(figsize=(size, size)) if not fig else fig

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axes = [axScatter, axHistx, axHisty]

    # no labels for histograms
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    scatter = axScatter.scatter(*S.T, **kwargs, zorder=100)
    axScatter.set_xlabel(labels[0])
    axScatter.set_ylabel(labels[1])
    axScatter.grid(True, linestyle='--')

    axHistx.hist(S[:, 0], 50, ec='white', fc='C7', zorder=100)
    axHisty.hist(S[:, 1], 50, orientation='horizontal', ec='white', fc='C7', zorder=100)
    axHistx.xaxis.grid(True, linestyle='--')
    axHisty.yaxis.grid(True, linestyle='--')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    return scatter, fig, axes

def plot_phi_error(X, X_test, ax=None, title=None, y_lim=None, outliers=False, reference=None, fontsize=10):

    if ax is None:
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111)

    AE = np.abs(X - X_test)[:, [0, 2]]
    MAE = np.mean(AE, axis=0)
    RMSE = np.sqrt(np.mean((X - X_test)**2, axis=0))[[0, 2]]

    if outliers:
        ax.boxplot(np.rad2deg(AE))
    else:
        ax.boxplot(np.rad2deg(AE), 0, '')

    ax.plot([1, 2], np.rad2deg(MAE), 'x', markersize=10, label='MAE')

    if reference:
        ax.plot([1, 2], np.rad2deg(reference), 'x', markersize=10, label='MAE_ref', color='r')

    ax.set_xticklabels([r'$\varphi$', r'$\dot{\varphi}$'])
    ax.set_ylabel(r"Fehler in $\circ$ bzw. $\frac{\circ}{s}$")
    ax.legend()
    ax.grid()

    if y_lim:
        ax.set_ylim(y_lim)

    if title:
        ae_string = "\n" + r"$AE_{\varphi, max}=\,$" + f"{np.max(np.rad2deg(AE[:, 0])):.2f}   " +\
                    r"$AE_{\dot{\varphi}, max}=\,$" + f"{np.max(np.rad2deg(AE[:, 1])):.2f}"
        mae_string = "\n" + r"$MAE_{\varphi}=\,$" + f"{np.rad2deg(MAE[0]):.2f}   " + r"$MAE_{\dot{\varphi}}=\,$" + f"{np.rad2deg(MAE[1]):.2f}"
        rmse_string = "\n" + r"$RMSE_{\varphi}=\,$" + f"{np.rad2deg(RMSE[0]):.2f}   " + r"$RMSE_{\dot{\varphi}}=\,$" + f"{np.rad2deg(RMSE[1]):.2f}"
        ax.set_title(title + ae_string + mae_string + rmse_string, fontsize=fontsize)