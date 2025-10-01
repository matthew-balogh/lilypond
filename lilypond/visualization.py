import matplotlib.pyplot as plt
import numpy as np

from minisom import MiniSom
from matplotlib.colors import ListedColormap

cmapWaterBlue = ListedColormap(['royalblue'])

def _place_flower(cx, cy, value, linewidth=3, pixel_width=1.0, gap_fraction=0.1, petal_multiplier=1, color="white", ax=plt.gca()):
    value = value * petal_multiplier

    if value == 0:
        return
    
    n = int(value)
    
    max_diameter_fraction = 1 - 2 * gap_fraction
    length = (pixel_width / 2) * max_diameter_fraction
    
    angles = np.linspace(0, 360, n, endpoint=False)
    for angle in angles:
        rad = np.radians(angle)
        x1 = cx
        y1 = cy
        x2 = cx + length * np.cos(rad)
        y2 = cy + length * np.sin(rad)
        
        ax.plot([x1, x2], [y1, y2], color=color, 
               linewidth=linewidth, alpha=0.9, 
               solid_capstyle='round', zorder=3)
    
    center_dot = plt.Circle((cx, cy), length * 0.1, color="#FFC800", zorder=6)
    ax.add_patch(center_dot)


def plot_pond(som: MiniSom,
              data,
              som_distance_scaling="mean",
              petal_multiplier=3,
              petal_color="white",
              petal_width=1,
              min_pad_gap_fraction=0.15,
              ax=None):
    
    assert (min_pad_gap_fraction < 0.5) and (min_pad_gap_fraction > 0.0), "Ensure 'min_pad_gap_fraction' is between (0, 0.5)"

    if ax is None:
        ax = plt.gca()

    # elements from SOM

    hitmap = som.activation_response(data).astype(int)
    distmap = som.distance_map(scaling=som_distance_scaling)
    lattice_shape = hitmap.shape

    # water layer

    im = ax.imshow(np.zeros(lattice_shape), origin="lower", cmap=cmapWaterBlue)

    # pad layer

    rows, cols = lattice_shape
    x_coords = np.repeat(np.arange(cols), rows)
    y_coords = np.tile(np.arange(rows), cols)

    extent = im.get_extent()
    x_min, x_max, _, _ = extent
    pixel_width = (x_max - x_min) / rows

    ax.figure.canvas.draw()
    points_data = np.array([[0, 0], [pixel_width, 0]])
    points_display = ax.transData.transform(points_data)
    pixel_width_points = points_display[1, 0] - points_display[0, 0]

    normalized_values = distmap / distmap.max() 
    inverse_normalized_values = 1 - normalized_values

    max_diameter_fraction = 1 - 2 * min_pad_gap_fraction
    min_marker_size = (pixel_width_points * 0.2) ** 2
    max_marker_size = (pixel_width_points * max_diameter_fraction) ** 2
    marker_sizes = min_marker_size + inverse_normalized_values * (max_marker_size - min_marker_size)

    ax.scatter(x_coords, y_coords, color="mediumseagreen", s=marker_sizes.T, alpha=.8, marker="8")

    # petals

    for i, j in np.ndindex(hitmap.shape):
        _place_flower(i, j, hitmap[j, i], linewidth=petal_width, pixel_width=pixel_width, gap_fraction=min_pad_gap_fraction*1.25, petal_multiplier=petal_multiplier, color=petal_color, ax=ax)
