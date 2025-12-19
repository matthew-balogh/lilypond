import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches

from typing import Literal, Optional, Union
from numpy.typing import ArrayLike

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
    
    def style_rhizome(self, linewidth=1, marker_start="o", marker_end="3", opacity=.9, zorder=1):
        self.rhizome_linewidth_ = linewidth
        self.rhizome_marker_start_ = marker_start
        self.rhizome_marker_end_ = marker_end
        self.rhizome_opacity_ = opacity
        self.rhizome_zorder_ = zorder

        self.rhizome_styled_ = True
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

    def attract(self, data,
                apply_style:Literal["normal", "abnormal", None]=None,
                marker="3", size_base=15, color=None,
                cmap="coolwarm", cmap_values:Union[Literal["bmu-distance"], ArrayLike]="bmu-distance", cmap_vmin=None, cmap_vmax=None, cmap_label=None,
                opacity=.9, zorder=10, label=None,
                subsample_ratio: Optional[float] = None):

        if (subsample_ratio is not None) and (not (0 < subsample_ratio < 1)):
            raise ValueError("Parameter `subsample` must be in the range (0, 1).")

        if not hasattr(self, "attract_winmaps_"):
            self.attract_winmaps_ = []
        if not hasattr(self, "attract_winmaps_idx_"):
            self.attract_winmaps_idx_ = []
        if not hasattr(self, "attract_markers_"):
            self.attract_markers_ = []
        if not hasattr(self, "attract_cmap_values_"):
            self.attract_cmap_values_ = []
        if not hasattr(self, "attract_cmap_labels_"):
            self.attract_cmap_labels_ = []
        if not hasattr(self, "attract_cmap_vlimits_"):
            self.attract_cmap_vlimits_ = []

        attr_wm = self.basin.som.win_map(data)
        attr_wm_idx = self.basin.som.win_map(data, return_indices=True)

        if subsample_ratio is not None:
            for (((x, y), points), ((_, _), points_idx)) in zip(attr_wm.items(), attr_wm_idx.items()):
                points = np.array(points)
                points_idx = np.array(points_idx)

                num_cluster = min(len(points), 3)
                km = KMeans(n_clusters=num_cluster, n_init=10, random_state=self.basin.random_seed)
                km.fit(points)

                labels = km.labels_
                unique_labels = np.unique(labels)

                sampled_points = []
                sampled_points_idx = []

                # sample per local cluster
                rng = np.random.default_rng(self.basin.random_seed)

                for lbl in unique_labels:
                    cluster_points = points[labels == lbl]
                    cluster_points_idx = points_idx[labels == lbl]

                    k = int(subsample_ratio * len(cluster_points))
                    k = max(1, k)

                    # if cluster smaller than k, just take all
                    if len(cluster_points) <= k:
                        chosen = cluster_points
                        chosen_idx = cluster_points_idx
                    else:
                        idx = rng.choice(len(cluster_points), size=k, replace=False)
                        chosen = cluster_points[idx]
                        chosen_idx = cluster_points_idx[idx]

                    sampled_points.append(chosen)
                    sampled_points_idx.append(chosen_idx)

                points_sampled = np.vstack(sampled_points)
                points_idx_sampled = np.hstack(sampled_points_idx)

                attr_wm_idx[(x, y)] = points_idx_sampled
                attr_wm[(x, y)] = points_sampled

        self.attract_winmaps_.append(attr_wm)
        self.attract_winmaps_idx_.append(attr_wm_idx)

        attract_marker = {
            "marker": marker,
            "size_base": size_base,
            "color": color,
            "cmap": cmap,
            "opacity": opacity,
            "zorder": zorder,
            "label": label,
        }

        if apply_style == "normal":
            attract_marker.update({"color": "blue", "marker": "3"})
        elif apply_style == "abnormal":
            attract_marker.update({"color": "red", "marker": "^"})

        self.attract_markers_.append(attract_marker)

        if cmap is not None:
            if isinstance(cmap_values, str) and cmap_values == "bmu-distance":
                dist_values = np.linalg.norm(data - self.basin.som.quantization(data), axis=1)
                self.attract_cmap_values_.append(dist_values)
                self.attract_cmap_labels_.append("Distance to BMU")
            elif cmap_values is not None:
                self.attract_cmap_values_.append(np.asarray(cmap_values).copy())
                self.attract_cmap_labels_.append(cmap_label)
            
            self.attract_cmap_vlimits_.append((cmap_vmin, cmap_vmax))

        self.attracted_ = True
        if self.verb: print("Pond has attracted.")

        return self
    
    def clean_attract(self):
        if hasattr(self, "attract_winmaps_"): del self.attract_winmaps_
        if hasattr(self, "attract_winmaps_idx_"): del self.attract_winmaps_idx_
        if hasattr(self, "attract_markers_"): del self.attract_markers_
        if hasattr(self, "attract_cmap_values_"): del self.attract_cmap_values_
        if hasattr(self, "attracted_"): del self.attracted_

        # TODO: clean styling?

        if self.verb: print("Pond has been cleaned from attraction.")

        return self

    def see_rhizome(self, X=None, ax=None, mode:Literal["all", "violating"]="violating", neighborhood:Literal["moore", "von-neumann"]="moore"):
        if X is None:
            X = self.basin.data.copy()

        if ax is None:
            ax = plt.gca()

        # ensure style
        if not hasattr(self, "rhizome_styled_") or self.rhizome_styled_ is False:
            self.style_rhizome()

        b2mu_inds = np.argsort(self.basin.som._distance_from_weights(X), axis=1)[:, :2]
        b2my_xy = np.unravel_index(b2mu_inds, self.basin.som._weights.shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]

        if mode == "violating":
            dxdy = np.hstack([np.diff(b2mu_x), np.diff(b2mu_y)])
            distance = np.linalg.norm(dxdy, axis=1)

            t_neigh = 1.42 if neighborhood == "moore" else 1
            show_rhizome = distance > t_neigh

        for i in range(len(b2mu_x)):

            if mode == "violating" and not show_rhizome[i]:
                continue

            x1, y1 = b2mu_x[i, 0], b2mu_y[i, 0]
            x2, y2 = b2mu_x[i, 1], b2mu_y[i, 1]

            ax.plot([y1, y2], [x1, x2], color="black", linewidth=self.rhizome_linewidth_, alpha=self.rhizome_opacity_, zorder=self.rhizome_zorder_)
            ax.scatter(y1, x1, s=50, color="black", marker=self.rhizome_marker_start_, alpha=self.rhizome_opacity_, zorder=self.rhizome_zorder_)
            ax.scatter(y2, x2, s=100, color="black", marker=self.rhizome_marker_end_, alpha=self.rhizome_opacity_, zorder=self.rhizome_zorder_)

        self.rhyzome_added_ = True
        if self.verb: print("Rhizome has been added.")
        
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
        
        ax.scatter(x_coords[mask], y_coords[mask], s=marker_sizes_filt, alpha=1, **pad_scatter_kwargs, label="pad")

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

        ax.scatter(x_coords[mask], y_coords[mask], s=marker_sizes_filt, alpha=self.underwater_opacity_, **pad_scatter_kwargs, label="pad (flooded)")

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
            fig = ax.figure
            all_axes = fig.get_axes()
            n_axes = len(all_axes)
            ax_idx = all_axes.index(ax)

            if not hasattr(fig, "_pond_gridspec"):
                gs = fig.add_gridspec(len(self.attract_winmaps_) + 1, n_axes, height_ratios=np.hstack(([1], np.repeat(.03, len(self.attract_winmaps_)))), width_ratios=np.repeat(1, n_axes))
                fig._pond_gridspec = gs
            
            gs = fig._pond_gridspec

            for attr_i, (attr_wm, attr_wm_idx, attr_m, attr_cmap_values, attr_cmap_vlimits, attr_cmap_label) in enumerate(zip(self.attract_winmaps_, self.attract_winmaps_idx_, self.attract_markers_, self.attract_cmap_values_, self.attract_cmap_vlimits_, self.attract_cmap_labels_)):
                cmap = plt.get_cmap(attr_m["cmap"])

                cmap_vmin = attr_cmap_vlimits[0] if attr_cmap_vlimits[0] is not None else attr_cmap_values.min()
                cmap_vmax = attr_cmap_vlimits[1] if attr_cmap_vlimits[1] is not None else attr_cmap_values.max()
                norm = Normalize(vmin=cmap_vmin, vmax=cmap_vmax)

                for attr_wm_i, (((x, y), points), ((_, _), points_idx)) in enumerate(zip(attr_wm.items(), attr_wm_idx.items())):
                    attr_m_label = attr_m["label"] if attr_wm_i == 0 else "_nolegend_"
                    jitter_amount = .5

                    values = attr_cmap_values[points_idx]

                    if attr_m["color"] is not None:
                        ax.scatter(
                                y + (np.random.rand(len(points)) - .5) * jitter_amount,
                                x + (np.random.rand(len(points)) - .5) * jitter_amount,
                                color=attr_m["color"],
                                s=attr_m["size_base"], marker=attr_m["marker"],
                                alpha=attr_m["opacity"], zorder=attr_m["zorder"], label=attr_m_label)
                    else:
                        ax.scatter(
                                y + (np.random.rand(len(points)) - .5) * jitter_amount,
                                x + (np.random.rand(len(points)) - .5) * jitter_amount,
                                c=values,
                                cmap=cmap,
                                norm=norm,
                                s=attr_m["size_base"], marker=attr_m["marker"],
                                alpha=attr_m["opacity"], zorder=attr_m["zorder"], label=attr_m_label)
                    
                    
                if attr_m["color"] is None:
                    cax = fig.add_subplot(gs[attr_i + 1, ax_idx])

                    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), orientation="horizontal", cax=cax, ax=ax) \
                        .set_label(f"{attr_cmap_label} ({attr_m['label']})")
        
        """ ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            labelspacing=1.5,
            borderpad=1.5,
            ncol=3,
        ) """

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
