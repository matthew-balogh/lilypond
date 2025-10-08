import matplotlib.pylab as plt
import numpy as np

from minisom import MiniSom
from typing import Literal
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Pond:
    def __init__(self, som: MiniSom, data, som_distance_scaling: Literal["sum", "mean"]="mean", verb=False):
        self.som = som
        self.data = data
        self.som_distance_scaling = som_distance_scaling
        self.verb = verb

        if self.verb: print("Pond is created.")

    def settle(self):
        if hasattr(self, "settled_"):
            del self.settled_
        if hasattr(self, "raided_"):
            del self.raided_
        if hasattr(self, "flooded_"):
            del self.flooded_

        hitmap = self.som.activation_response(self.data).astype(int)
        distmap = self.som.distance_map(scaling=self.som_distance_scaling)
        lattice_shape = hitmap.shape

        self.hitmap_ = hitmap
        self.distmap_ = distmap
        self.lattice_shape_ = lattice_shape
        self.rows_, self.cols_ = lattice_shape

        self.cmapWaterBlue_ = ListedColormap(['royalblue'])

        self.settled_ = True
        if self.verb: print("Pond has settled.")

        return self;

    def flood(self, below_activations=1, underwater_opacity=.4):
        assert (underwater_opacity >= 0.0) and (underwater_opacity <= 1.0), "The `underwater_opacity` must be within [0.0, 1.0]."
        assert below_activations >= 0, "The `below_activations` must be a positive number."

        self.flood_below_activations_ = below_activations
        self.underwater_opacity_ = underwater_opacity

        self.flooded_ = True
        if self.verb: print(f"Pads with less than {below_activations} activations have been flooded.")

        return self

    def raid(self, data_abnormal):
        raidwinmap = self.som.win_map(data_abnormal)
        self.raidwinmap_ = raidwinmap

        self.raided_ = True
        if self.verb: print("Pond has been raided.")

        return self

    def visualize(self, ax=None, title=None,
                  petal_color="white", petal_magnifier=3, petal_width=1, petal_min_gap_fraction=.25, hide_petals=False,
                  pad_min_gap_fraction=.25):
        assert hasattr(self, "settled_") and self.settled_ is True, "The pond has not settled yet. Call `settle()` first."

        if ax is None:
            ax = plt.gca()
        
        if title is not None:
            ax.set_title(title)

        # water layer
        ## display static blue background
        backgroundImg = ax.imshow(np.zeros(self.lattice_shape_), origin="lower", cmap=self.cmapWaterBlue_)
        extent = backgroundImg.get_extent()
        x_min, x_max, _, _ = extent
        pixel_width = (x_max - x_min) / self.rows_
        ax.figure.canvas.draw()
        points_data = np.array([[0, 0], [pixel_width, 0]])
        points_display = ax.transData.transform(points_data)
        pixel_width_points = points_display[1, 0] - points_display[0, 0]

        # pad layer

        ## display pads sized according to their neighboring distance
        inverse_normalized_values = 1 - (self.distmap_ / self.distmap_.max()) 

        max_diameter_fraction = 1 - (2 * pad_min_gap_fraction)
        min_marker_size = (pixel_width_points * 0.2) ** 2
        max_marker_size = (pixel_width_points * max_diameter_fraction) ** 2
        marker_sizes = min_marker_size + inverse_normalized_values * (max_marker_size - min_marker_size)
        x_coords = np.repeat(np.arange(self.cols_), self.rows_)
        y_coords = np.tile(np.arange(self.rows_), self.cols_)

        hitmapShape = self.hitmap_.shape
        
        if not hasattr(self, 'flooded_') or self.flooded_ is not True:
            ax.scatter(x_coords, y_coords, color="mediumseagreen", s=marker_sizes.T, alpha=1, marker="8")

            # petals
            if not hide_petals:
                for i, j in np.ndindex(hitmapShape):
                    self.__place_petals(j, i, self.hitmap_[i, j],
                                        petal_magnifier, petal_min_gap_fraction,
                                        pixel_width, petal_width, petal_color, 1, ax)


        elif hasattr(self, 'flood_below_activations_'):
            hit_mask = self.hitmap_.T.flatten() >= self.flood_below_activations_

            marker_sizes_filt = marker_sizes.T.flatten()[hit_mask]
            ax.scatter(x_coords[hit_mask], y_coords[hit_mask], color="mediumseagreen", s=marker_sizes_filt.T, alpha=1, marker="8")

            hit_mask_2d = hit_mask.reshape(self.hitmap_.T.shape).T

            # petals
            if not hide_petals:
                for i, j in np.ndindex(hitmapShape):
                    if hit_mask_2d[i, j]:
                        self.__place_petals(j, i, self.hitmap_[i, j],
                                            petal_magnifier, petal_min_gap_fraction,
                                            pixel_width, petal_width, petal_color, 1, ax)

            marker_sizes_filt = marker_sizes.T.flatten()[~hit_mask]
            ax.scatter(x_coords[~hit_mask], y_coords[~hit_mask], color="mediumseagreen", s=marker_sizes_filt.T, alpha=self.underwater_opacity_, marker="8")

            hit_mask_2d = ~hit_mask.reshape(self.hitmap_.T.shape).T

            # petals
            if not hide_petals:
                for i, j in np.ndindex(hitmapShape):
                    if hit_mask_2d[i, j]:
                        self.__place_petals(j, i, self.hitmap_[i, j],
                                            petal_magnifier, petal_min_gap_fraction,
                                            pixel_width, petal_width, petal_color, self.underwater_opacity_, ax)
        
        if hasattr(self, "raided_") and self.raidwinmap_ is not None:
            for (x, y), points in self.raidwinmap_.items():
                ax.scatter(x, y, color="red", s=len(points)*15, marker="^", zorder=10)

        if self.verb: print(f"Pond is visualized.")

    def legacy(self):
        if not hasattr(self, "legacyPond_") or self.legacyPond_ is None:
            self.legacyPond_ = LegacyPond(self, self.verb)
        
        return self.legacyPond_

    def __place_petals(self, cx, cy, hit_num, petal_magnifier, petal_min_gap_fraction, pixel_width, linewidth, color, opacity, ax):
        if hit_num == 0:
            return
        
        hit_num = int(hit_num * petal_magnifier)
        
        max_diameter_fraction = 1 - 2 * petal_min_gap_fraction
        length = (pixel_width / 2) * max_diameter_fraction
        
        angles = np.linspace(0, 360, hit_num, endpoint=False)
        for angle in angles:
            rad = np.radians(angle)
            x1, y1 = cx, cy
            x2 = cx + length * np.cos(rad)
            y2 = cy + length * np.sin(rad)
            
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=opacity, solid_capstyle='round', zorder=3,)
        
        center_dot = plt.Circle((cx, cy), length * 0.1, color="#FFC800", zorder=6, alpha=opacity)
        ax.add_patch(center_dot)


class LegacyPond:
    def __init__(self, pond: Pond, verb=False):
        self.pond = pond
        self.verb = verb

        if self.verb: print("Legacy pond is created.")
    
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
