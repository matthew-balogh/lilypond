import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches

from typing import Literal, Optional
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import KBinsDiscretizer

from lilypond.basin import Basin

class Pond:

    def __init__(self, basin: Basin, verb=False):
        self.basin = basin
        self.verb = verb

        if self.verb: print("Pond has been initialized.")

    def style_pad(self, gap=.25, marker="8"):
        self.pad_gap_ = gap
        self.pad_marker_ = marker

        self.pad_styled_ = True
        return self
    
    def style_petal(self, color="white", magnifier=3, width=1, size_base=None, gap=.25, hide=False):
        self.petal_color_ = color
        self.petal_magnifier_ = magnifier
        self.petal_width_ = width
        self.petal_size_base_ = size_base
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
    
    def set_coloring_strategy(self, strategy:Literal["uniform", "distance_map", "component_map"]="uniform", component_idx=None):
        self.pad_coloring_strategy_ = strategy
        self.pad_coloring_component_idx_ = component_idx

        self.pad_coloring_strategy_set_ = True
        return self
    
    def aggregate_petals(self, patch_size=(2, 2), method:Literal["sum", "mean"]="sum"):
        self.petal_agg_patch_size_ = patch_size
        self.petal_agg_method_ = method
        
        self.petal_aggregated_ = True
        return self
    
    def discretize_petals(self, n_bins=10):
        self.petal_disc_n_bins_ = n_bins

        self.petal_discretized_ = True
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

    def observe(self, return_fig=False, title=None, ax=None):
        if ax is None:
            ax = plt.gca()
        
        if title is not None:
            ax.set_title(title)
        
        # ensure style
        self.__style()

        # ensure coloring strategy
        if not hasattr(self, "pad_coloring_strategy_set_") or self.pad_coloring_strategy_set_ is False:
            self.set_coloring_strategy()

        # ensure flood
        if not hasattr(self, "flooded_") or self.flooded_ is False:
            self.flood()

        # water layer (display static blue background)
        backgroundImg = ax.imshow(np.zeros(self.basin.lattice_shape_), origin="lower", cmap=self.basin.cmapWaterBlue_)
        pixel_width, pixel_width_points = self.__calc_pixel_width(backgroundImg, ax)

        # pad layer

        distmap = self.basin.distmap_
        hitmap = self.basin.hitmap_
        hitmap_petal = hitmap.copy()
        petals_aggregated = hasattr(self, "petal_aggregated_") and self.petal_aggregated_
        petals_discretized = hasattr(self, "petal_discretized_") and self.petal_discretized_

        if petals_aggregated:
            hitmap_petal = self.__aggregate_hitmap(hitmap, patch_size=self.petal_agg_patch_size_, method=self.petal_agg_method_)

        hitmap_petal_transformed = hitmap_petal.copy()

        if petals_discretized:
            mask_positive = hitmap_petal > 0
            hit_discretizer = KBinsDiscretizer(n_bins=self.petal_disc_n_bins_, strategy="uniform", encode="ordinal", random_state=self.basin.random_seed)
            hitmap_petal_transformed[mask_positive] = hit_discretizer.fit_transform(hitmap_petal[mask_positive].reshape(-1, 1)).ravel()
            hitmap_petal_transformed[mask_positive] += 1
            hitmap_petal_transformed = hitmap_petal_transformed.reshape(hitmap_petal.shape)

        marker_sizes = self.__calc_marker_sizes(distmap, pixel_width_points)
        x_coords = np.repeat(np.arange(self.basin.cols_), self.basin.rows_)
        y_coords = np.tile(np.arange(self.basin.rows_), self.basin.cols_)

        flood_mask_pad = hitmap.T.flatten() >= self.flood_below_activations_
        flood_mask_petal = hitmap_petal.T.flatten() >= self.flood_below_activations_

        pad_scatter_kwargs = {"marker": self.pad_marker_}

        if petals_aggregated:
            bh, bw = self.petal_agg_patch_size_
            if (bh + bw) > 2:
                for (i, j) in np.ndindex(hitmap_petal.shape):
                    rect = patches.Rectangle((j * bh, i * bw), bw-1, bh-1, linewidth=2, edgecolor='grey', facecolor='none', zorder=0, alpha=.6)
                    ax.add_patch(rect)

        if self.pad_coloring_strategy_ != "uniform":
            print(self.pad_coloring_strategy_)
            if self.pad_coloring_strategy_ == "distance_map":
                pad_colors_cmap = LinearSegmentedColormap.from_list("PondGreens", [
                    (0.05, 0.15, 0.05),   # dark
                    (0.15, 0.35, 0.15),   # natural
                    (0.25, 0.55, 0.25),   # medium seagree
                    (0.35, 0.7, 0.35),    # lighter
                    (0.45, 0.85, 0.45)    # subtle soft
                ], N=256)
                pad_colors = distmap.T.flatten()
                pad_colors_norm = plt.Normalize(vmin=distmap.min(), vmax=distmap.max())
                
            elif self.pad_coloring_strategy_ == "component_map":
                assert self.pad_coloring_component_idx_ is not None, "The component idx must be set when the coloring strategy is set to `component_map`."
                assert self.pad_coloring_component_idx_ >= 0, "The component idx must be a positive number."
                assert self.pad_coloring_component_idx_ < self.basin.component_size_, f"The component idx must be smaller than {self.basin.component_size_}."
                pad_colors_cmap = plt.get_cmap("BrBG")
                node_weights_fi = self.basin.node_weights_[:, :, self.pad_coloring_component_idx_]
                pad_colors = node_weights_fi.T.flatten()
                pad_colors_norm = plt.Normalize(vmin=node_weights_fi.min(), vmax=node_weights_fi.max())

            pad_scatter_kwargs.update({
                "cmap": pad_colors_cmap,
                "norm": pad_colors_norm
            })
        else:
            pad_scatter_kwargs["color"] = "mediumseagreen"

        ## unflooded pads
        mask = flood_mask_pad.copy()
        marker_sizes_filt = marker_sizes.T.flatten()[mask]

        if self.pad_coloring_strategy_ != "uniform":
            pad_scatter_kwargs.update({
                "c": pad_colors[mask]
            })
        
        ax.scatter(x_coords[mask], y_coords[mask], s=marker_sizes_filt, alpha=1, **pad_scatter_kwargs)

        ### respective petals
        mask_petal = flood_mask_petal.copy()
        mask_2d = mask_petal.reshape(hitmap_petal.T.shape).T
        if not self.hide_petals_:
            for i, j in np.ndindex(hitmap_petal.shape):
                if mask_2d[i, j]:
                    if petals_aggregated:
                        bh, bw = self.petal_agg_patch_size_
                        cx = np.mean([j * bh, (j + 1) * bh - 1])
                        cy = np.mean([i * bw, (i + 1) * bw - 1])
                    else:
                        cx, cy = j, i

                    self.__place_petals(cx, cy, hitmap_petal_transformed[i, j], pixel_width, ax)

        ## flooded pads
        mask = ~flood_mask_pad.copy()
        marker_sizes_filt = marker_sizes.T.flatten()[mask]

        if self.pad_coloring_strategy_ != "uniform":
            pad_scatter_kwargs.update({
                "c": pad_colors[mask]
            })

        ax.scatter(x_coords[mask], y_coords[mask], s=marker_sizes_filt, alpha=self.underwater_opacity_, **pad_scatter_kwargs)

        ### respective petals
        mask_petal = ~flood_mask_petal.copy()
        mask_2d = mask_petal.reshape(hitmap_petal.T.shape).T
        if not self.hide_petals_:
            for i, j in np.ndindex(hitmap_petal.shape):
                if mask_2d[i, j]:
                    if petals_aggregated:
                        bh, bw = self.petal_agg_patch_size_
                        cx = np.mean([j * bh, (j + 1) * bh - 1])
                        cy = np.mean([i * bw, (i + 1) * bw - 1])
                    else:
                        cx, cy = j, i

                    self.__place_petals(cx, cy, hitmap_petal_transformed[i, j], pixel_width, ax, opacity=self.underwater_opacity_)

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

        if return_fig:
            if self.verb: print(f"Pond figure is retrieved.")
            return ax.figure
        else:
            if self.verb: print(f"Pond is visualized.")
            plt.show()

    def aerial(self, subsample: Optional[float] = None):
        from lilypond.aerial import Aerial
        return Aerial(self, subsample, self.verb)
    

    def __aggregate_hitmap(self, hitmap, patch_size=(5,5), method:Literal["sum", "mean"]="sum"):
        matrix = hitmap.copy()

        h, w = matrix.shape
        bh, bw = patch_size
        
        if h % bh != 0 or w % bw != 0:
            raise ValueError(f"Matrix shape {matrix.shape} is not divisible by block shape {patch_size}")
        
        H, W = h // bh, w // bw
        matrix = matrix[:H*bh, :W*bw]  # crop to fit exact blocks
        blocks = matrix.reshape(H, bh, W, bw).swapaxes(1,2).reshape(-1, bh, bw)

        if method == "sum":
            agg = np.array([b.sum() for b in blocks]).reshape(int(h / bh), int(w / bw))
        elif method == "mean":
            agg = np.array([b.mean() for b in blocks]).reshape(int(h / bh), int(w / bw))
        else: raise NotImplementedError(f"Aggregation method \"{method}\" is not supported.")

        return agg
    
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

    def __place_petals(self, cx, cy, count, pixel_width, ax, opacity=1):
        if count == 0:
            return
        
        size_base = self.petal_size_base_ if self.petal_size_base_ is not None else (pixel_width / 2)

        count = int(count * self.petal_magnifier_)
        max_diameter_fraction = 1 - 2 * self.petal_gap_
        length = size_base * max_diameter_fraction
        angles = np.linspace(0, 360, count, endpoint=False)

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
