import proplot as pplt
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

'''
File for helping with some plotting functions
'''


# Colors to assign to auxiliary tasks (they will be assigned in order)
colors = pplt.Cycle('default').by_key()['color']
hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
rgb_colors = np.array([hex_to_rgb(color) for color in colors])/255

def rgb_to_hex(rgb, scaleup=True):
    if scaleup:
        r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    else:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

hex_colors = [rgb_to_hex(color) for color in rgb_colors]




def create_lasso_selector_plot(data):
    '''
    Given 2D data of shape (N, 2), create a plot where lasso can select chunks of points
    and return which indexes belonged to each cluster
    
    returns:
        cluster_idxs (list of indices 1 per cluster), selector (object)

    Note in Jupyter notebook, must run 
        %matplotlib tk
    first to make the plot in a pop out
    '''
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    
    subplot_kw = dict(xlim=(xmin, xmax), ylim=(ymin, ymax), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            # print("Selected points:")
            # print(selector.xys[selector.ind])

            selector.onenter(selector.ind)
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")
    plt.show()
    
    return selector.cluster_idxs, selector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_on=0.1, alpha_off=0.01):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        
        self.colored_clusters = 1
        
        self.cluster_idxs = []

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_off
        self.fc[self.ind, -1] = self.alpha_on
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def onenter(self, idxs):
        self.cluster_idxs.append(idxs)
        new_color = np.concatenate([rgb_colors[self.colored_clusters], [self.alpha_on]])
        self.fc[idxs] = new_color
        self.fc[:, -1] = self.alpha_on
        self.colored_clusters += 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

        
    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


