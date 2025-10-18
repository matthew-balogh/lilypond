import matplotlib.pylab as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from lilypond.basin import Basin

class LegacyPond:
    def __init__(self, basin: Basin, verb=False):
        self.basin = basin
        self.verb = verb

        if self.verb: print("LegacyPond has been initialized.")
    
    def visualize_hitmap(self, cmap="binary", title="Hitmap", ax=None):
        if ax is None:
            ax = plt.gca()
        
        ax.set_title(title)

        im = ax.imshow(self.basin.hitmap_, origin="lower", cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.2)
        cax.tick_params(labelsize=11)
        cbar = plt.colorbar(im, cax=cax)

    def visualize_distance_map(self, cmap="Spectral_r", title="Distance map", ax=None):
        if ax is None:
            ax = plt.gca()
        
        ax.set_title(title)

        im = ax.imshow(self.basin.distmap_, origin="lower", cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.2)
        cax.tick_params(labelsize=11)
        cbar = plt.colorbar(im, cax=cax)
