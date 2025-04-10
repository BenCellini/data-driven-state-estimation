
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import figure_functions as ff


class TrajectoryAnimator:
    def __init__(self, x, y, heading, color=None, colormap='inferno_r', colornorm=None, future_color='gray',
                 n_skip=0, arrow_size=None, sliding_window=0, fig_kwargs=None):
        """
        Initialize trajectory plot.

        :param list | np.ndarray x: x-position time-series of the trajectory
        :param list | np.ndarray y: y-position time-series of the trajectory
        :param list | np.ndarray heading: heading time-series of the trajectory
        :param list | np.ndarray color: color time-series
        """

        # Set trajectory data
        self.x = x[0::n_skip]
        self.y = y[0::n_skip]
        self.heading = heading[0::n_skip]
        self.n = self.x.shape[0]
        self.sliding_window = sliding_window

        # Set colors
        self.color = color[0::n_skip]
        self.colormap = colormap
        self.colornorm = colornorm
        self.future_color = mpl.colors.to_rgb(future_color)
        self.future_colormap = mpl.colors.ListedColormap(np.vstack([self.future_color, self.future_color]))

        # Set axes limits
        self.x_range = self.x.max() - self.x.min()
        self.y_range = self.y.max() - self.y.min()

        space = 0.05
        self.xlim = (self.x.min() - space * self.x_range, self.x.max() + space * self.x_range)
        self.ylim = (self.y.min() - space * self.y_range, self.y.max() + space * self.y_range)

        if arrow_size is None:
            scale = np.max([self.x_range, self.y_range])
            self.arrow_size = scale * 0.05
        else:
            self.arrow_size = arrow_size

        # Set figure
        fig_kwargs_default = dict(dpi=300,
                                  figsize=(10, 10)
                                  )

        self.fig_kwargs = fig_kwargs_default.copy()
        if fig_kwargs is not None:
            self.fig_kwargs.update(fig_kwargs)

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, **self.fig_kwargs)

        # Create animation
        self.animation = FuncAnimation(self.fig, self.update, frames=range(0, self.n + 1), blit=False)

    def update(self, frame):
        """ Update & plot.
        """

        # Clear axis
        self.ax.clear()

        # Plot trajectory history
        if frame > 0:
            x = self.x[0:frame]
            y = self.y[0:frame]
            heading = self.heading[0:frame]
            color = self.color[0:frame]

            ff.plot_trajectory(x, y, heading, color=color,
                               ax=self.ax,
                               nskip=0,
                               size_radius=self.arrow_size,
                               colormap=self.colormap,
                               colornorm=self.colornorm
                               )

        # Plot trajectory future
        if frame < self.n:
            x = self.x[frame:]
            y = self.y[frame:]
            heading = self.heading[frame:]
            color = self.color[frame:]

            ff.plot_trajectory(x, y, heading, color=color,
                               ax=self.ax,
                               nskip=0,
                               size_radius=self.arrow_size,
                               colormap=self.future_colormap,
                               colornorm=self.colornorm
                               )


        # Draw sliding window
        if self.sliding_window > 0:
            if frame > 0:
                self.ax.plot(self.x[frame:frame + self.sliding_window],
                             self.y[frame:frame + self.sliding_window],
                             linewidth=50.0 * self.arrow_size, alpha=0.2, color='purple')

        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)

        self.ax.set_axis_off()

        # self.ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        # self.ax.tick_params(axis='both', which='major', labelsize=6, bottom=False, labelbottom=False, left=False,
        #               labelleft=False)
        # for s in ['top', 'bottom', 'left', 'right']:
        #     self.ax.spines[s].set_visible(False)
