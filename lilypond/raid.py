import plotly.graph_objects as go
import numpy as np

from lilypond.pond import Pond


class RaidedPond():

    def __init__(self, pond: Pond, data_abnormal, verb=False):
        self.pond = pond
        self.data_abnormal = data_abnormal
        self.verb = verb

        if self.verb: print("Raided pond is created.")

    def visualize(self):
        som = self.pond.som
        distance_map = som.distance_map()

        x, y = zip(*som.win_map(self.data_abnormal).keys())
        z = np.linalg.norm(self.data_abnormal - som.quantization(self.data_abnormal), axis=1)

        m, n = self.pond.lattice_shape_
        x_coords = np.linspace(0, n-1, n)
        y_coords = np.linspace(0, m-1, m)
        xx, yy = np.meshgrid(x_coords, y_coords)

        fig = go.Figure()

        # raid points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color='black', opacity=1, symbol="cross")
        ))

        # raid point projections on XY-plane
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=np.zeros_like(z),
            mode='markers',
            marker=dict(size=5, color='grey', symbol='circle'),
            name='Projections'
        ))

        # projection lines
        for xi, yi, zi in zip(x, y, z):
            fig.add_trace(go.Scatter3d(
                x=[xi, xi], y=[yi, yi], z=[0, zi],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                opacity=1,
                showlegend=False
            ))

        # pond surface
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=np.zeros_like(xx),
            surfacecolor=np.flipud(distance_map),
            colorscale='Viridis_r',
            showscale=True,
            opacity=0.7
        ))

        # fig.update_layout(scene=dict(zaxis=dict(range=[0, 1])))
        fig.show()

        return fig