import plotly.graph_objects as go
import numpy as np

from lilypond.pond import Pond


class RaidedPond():

    def __init__(self, pond: Pond, data_abnormal, verb=False):
        self.pond = pond
        self.data_abnormal = data_abnormal
        self.verb = verb

        if self.verb: print("Raided pond is created.")

    def visualize(self, return_fig=True):
        som = self.pond.som
        distance_map = som.distance_map()
        abnormal_win_map = som.win_map(self.data_abnormal)

        triplets = [
            (i, j, val)
            for (i, j), val_list in abnormal_win_map.items()
            for val in val_list
        ]

        bmu_x, bmu_y, v = zip(*triplets)
        bmu_dist = np.linalg.norm(self.data_abnormal - som.quantization(self.data_abnormal), axis=1)

        r, c = self.pond.lattice_shape_
        gridX, gridY = np.linspace(0, c-1, c), np.linspace(0, r-1, r)
        gridXX, gridYY = np.meshgrid(gridX, gridY)

        fig = go.Figure()

        # raid points
        fig.add_trace(go.Scatter3d(
            x=bmu_x, y=bmu_y, z=bmu_dist,
            mode='markers',
            marker=dict(size=3, color='black', opacity=1, symbol="cross")
        ))

        # raid point projections on XY-plane
        fig.add_trace(go.Scatter3d(
            x=bmu_x, y=bmu_y, z=np.zeros_like(bmu_dist),
            mode='markers',
            marker=dict(size=1, color='grey', symbol='square'),
            name='Projections'
        ))

        # projection lines
        for xi, yi, zi in zip(bmu_x, bmu_y, bmu_dist):
            fig.add_trace(go.Scatter3d(
                x=[xi, xi], y=[yi, yi], z=[0, zi],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                opacity=1,
                showlegend=False
            ))

        # pond surface
        fig.add_trace(go.Surface(
            x=gridXX, y=gridYY, z=np.zeros_like(gridXX),
            surfacecolor=np.flipud(distance_map),
            colorscale='Viridis_r',
            showscale=True,
            opacity=0.7
        ))

        if return_fig: return fig
        else: fig.show()
