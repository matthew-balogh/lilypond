import numpy as np
import matplotlib.pylab as plt

from typing import Literal
from matplotlib.colors import LinearSegmentedColormap

from lilypond.basin import Basin

class Pond:

    def __init__(self, basin: Basin, verb=False):
        self.basin = basin
        self.verb = verb

        if self.verb: print("Pond has been initialized.")

    def style_pad(self, gap=.25, marker="8", coloring:Literal["constant", "gradient"]="constant"):
        self.pad_gap_ = gap
        self.pad_marker_ = marker
        self.pad_coloring_ = coloring

        self.pad_styled_ = True
        return self
    
    def style_petal(self, color="white", magnifier=3, width=1, gap=.25, hide=False):
        self.petal_color_ = color
        self.petal_magnifier_ = magnifier
        self.petal_width_ = width
        self.petal_gap_ = gap
        self.hide_petals_ = hide

        self.petal_styled_ = True
        return self
    
    def style_flood(self, underwater_opacity=.4):
        self.underwater_opacity_ = underwater_opacity
        
        self.flood_styled_ = True
        return self
    
    def style_attract(self, marker="3", size_base=15, color="black", opacity=.9):
        self.attract_marker_marker_ = marker
        self.attract_marker_size_base_ = size_base
        self.attract_marker_color_ = color
        self.attract_marker_opacity_ = opacity

        self.attract_styled_ = True
        return self

    def style_raid(self, marker="^", size_base=15, color="black", opacity=.9):
        self.raid_marker_marker_ = marker
        self.raid_marker_size_base_ = size_base
        self.raid_marker_color_ = color
        self.raid_marker_opacity_ = opacity

        self.raid_styled_ = True
        return self
            
    def flood(self, below_activations=1):
        assert below_activations >= 0, "The `below_activations` must be a non-negative number."

        self.flood_below_activations_ = below_activations

        self.flooded_ = True
        if self.verb: print(f"Pads with less than {below_activations} activations have been flooded.")

        return self

    def attract(self, data):
        self.attract_winmap_ = self.basin.som.win_map(data)

        self.attracted_ = True
        if self.verb: print("Pond has attracted.")

        return self
    
    def raid(self, data):
        self.raid_winmap_ = self.basin.som.win_map(data)

        self.raided_ = True
        if self.verb: print("Pond has been raided.")

        return self
    
    def clean_attract_raid(self):
        if hasattr(self, "attract_winmap_"): del self.attract_winmap_
        if hasattr(self, "raid_winmap_"): del self.raid_winmap_
        if hasattr(self, "attracted_"): del self.attracted_
        if hasattr(self, "raided_"): del self.raided_

        # TODO: clean styling?

        if self.verb: print("Pond has been cleaned from attraction and raid.")

        return self

    def observe(self, title=None, ax=None):
        if ax is None:
            ax = plt.gca()
        
        if title is not None:
            ax.set_title(title)
        
        # ensure style
        self.__style()

        # ensure flood
        if not hasattr(self, "flooded_") or self.flooded_ is False:
            self.flood()

        # water layer (display static blue background)
        backgroundImg = ax.imshow(np.zeros(self.basin.lattice_shape_), origin="lower", cmap=self.basin.cmapWaterBlue_)
        pixel_width, pixel_width_points = self.__calc_pixel_width(backgroundImg, ax)

        # pad layer

        distmap = self.basin.distmap_
        hitmap = self.basin.hitmap_

        marker_sizes = self.__calc_marker_sizes(distmap, pixel_width_points)
        x_coords = np.repeat(np.arange(self.basin.cols_), self.basin.rows_)
        y_coords = np.tile(np.arange(self.basin.rows_), self.basin.cols_)

        flood_mask = hitmap.T.flatten() >= self.flood_below_activations_

        pad_scatter_kwargs = {"marker": self.pad_marker_}

        if self.pad_coloring_ == "gradient":
            pad_colors_cmap = LinearSegmentedColormap.from_list("PondGreens", [
                (0.05, 0.15, 0.05),   # dark
                (0.15, 0.35, 0.15),   # natural
                (0.25, 0.55, 0.25),   # medium seagree
                (0.35, 0.7, 0.35),    # lighter
                (0.45, 0.85, 0.45)    # subtle soft
            ], N=256)
            pad_colors = distmap.T.flatten()
            pad_colors_norm = plt.Normalize(vmin=distmap.min(), vmax=distmap.max())
            
            pad_scatter_kwargs.update({
                "cmap": pad_colors_cmap,
                "norm": pad_colors_norm
            })
        else:
            pad_scatter_kwargs["color"] = "mediumseagreen"

        ## unflooded pads
        mask = flood_mask.copy()
        marker_sizes_filt = marker_sizes.T.flatten()[mask]

        if self.pad_coloring_ == "gradient":
            pad_scatter_kwargs.update({
                "c": pad_colors[mask]
            })
        
        ax.scatter(x_coords[mask], y_coords[mask], s=marker_sizes_filt, alpha=1, **pad_scatter_kwargs)

        ### respective petals
        mask_2d = mask.reshape(hitmap.T.shape).T
        if not self.hide_petals_:
            for i, j in np.ndindex(hitmap.shape):
                if mask_2d[i, j]:
                    self.__place_petals(j, i, hitmap[i, j], pixel_width, ax)

        ## flooded pads
        mask = ~flood_mask.copy()
        marker_sizes_filt = marker_sizes.T.flatten()[mask]

        if self.pad_coloring_ == "gradient":
            pad_scatter_kwargs.update({
                "c": pad_colors[mask]
            })

        ax.scatter(x_coords[mask], y_coords[mask], s=marker_sizes_filt, alpha=self.underwater_opacity_, **pad_scatter_kwargs)

        ### respective petals
        mask_2d = mask.reshape(hitmap.T.shape).T
        if not self.hide_petals_:
            for i, j in np.ndindex(hitmap.shape):
                if mask_2d[i, j]:
                    self.__place_petals(j, i, hitmap[i, j], pixel_width, ax, opacity=self.underwater_opacity_)

        # attract layer
        if hasattr(self, "attracted_"):
            for (x, y), points in self.attract_winmap_.items():
                ax.scatter(y, x,
                           color=self.attract_marker_color_, s=self.attract_marker_size_base_ * len(points), marker=self.attract_marker_marker_,
                           alpha=self.attract_marker_opacity_, zorder=10)

        # raid layer
        if hasattr(self, "raided_"):
            for (x, y), points in self.raid_winmap_.items():
                ax.scatter(y, x,
                           color=self.raid_marker_color_, s=self.raid_marker_size_base_ * len(points), marker=self.raid_marker_marker_,
                           alpha=self.raid_marker_opacity_, zorder=11)

        plt.show()

        if self.verb: print(f"Pond is visualized.")

    def aerial(self):
        from lilypond.aerial import Aerial
        return Aerial(self, self.verb)
    
    def __calc_pixel_width(self, backgroundImg, ax):
        backgroundImgXMin, backgroundImgXMax, _, _ = backgroundImg.get_extent()
        pixel_width = (backgroundImgXMax - backgroundImgXMin) / self.basin.rows_
        ax.figure.canvas.draw()
        points_data = np.array([[0, 0], [pixel_width, 0]])
        points_display = ax.transData.transform(points_data)
        pixel_width_points = points_display[1, 0] - points_display[0, 0]
        return pixel_width, pixel_width_points
    
    def __calc_marker_sizes(self, distmap, pixel_width_points):
        inverse_normalized_distances = 1 - (distmap / distmap.max())
        max_diameter_fraction = 1 - (2 * self.pad_gap_)
        min_marker_size = (pixel_width_points * 0.2) ** 2
        max_marker_size = (pixel_width_points * max_diameter_fraction) ** 2
        marker_sizes = min_marker_size + inverse_normalized_distances * (max_marker_size - min_marker_size)
        return marker_sizes

    def __place_petals(self, cx, cy, hit_num, pixel_width, ax, opacity=1):
        if hit_num == 0:
            return

        hit_num = int(hit_num * self.petal_magnifier_)
        max_diameter_fraction = 1 - 2 * self.petal_gap_
        length = (pixel_width / 2) * max_diameter_fraction
        angles = np.linspace(0, 360, hit_num, endpoint=False)

        for angle in angles:
            rad = np.radians(angle)
            x1, y1 = cx, cy
            x2 = cx + length * np.cos(rad)
            y2 = cy + length * np.sin(rad)

            ax.plot([x1, x2], [y1, y2], color=self.petal_color_, linewidth=self.petal_width_, alpha=opacity, solid_capstyle='round', zorder=3)

        center_dot = plt.Circle((cx, cy), length * 0.1, color="#FFC800", zorder=6, alpha=opacity)
        ax.add_patch(center_dot)

    def __style(self):
        if not hasattr(self, "pad_styled_") or self.pad_styled_ is False:
            self.style_pad()
        if not hasattr(self, "petal_styled_") or self.petal_styled_ is False:
            self.style_petal()
        if not hasattr(self, "flood_styled_") or self.flood_styled_ is False:
            self.style_flood()
        if not hasattr(self, "attract_styled_") or self.attract_styled_ is False:
            self.style_attract()
        if not hasattr(self, "raid_styled_") or self.raid_styled_ is False:
            self.style_raid()
