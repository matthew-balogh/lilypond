import matplotlib.pylab as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from lilypond.pond import Pond

class LegacyPond:
    def __init__(self, pond: Pond, verb=False):
        self.pond = pond
        self.verb = verb

        if self.verb: print("LegacyPond is initialized.")
    
    def visualize_hitmap(self, cmap="binary", ax=None, title="Hitmap"):
        if ax is None:
            ax = plt.gca()
        
        ax.set_title(title)

        im = ax.imshow(self.pond.hitmap_, origin="lower", cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.2)
        cax.tick_params(labelsize=11)
        cbar = plt.colorbar(im, cax=cax)

    def visualize_distance_map(self, cmap="Spectral_r", ax=None, title="Distance map"):
        if ax is None:
            ax = plt.gca()
        
        ax.set_title(title)

        im = ax.imshow(self.pond.distmap_, origin="lower", cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.2)
        cax.tick_params(labelsize=11)
        cbar = plt.colorbar(im, cax=cax)
