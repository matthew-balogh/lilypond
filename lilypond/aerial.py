import numpy as np
import plotly.graph_objects as go

from lilypond.pond import Pond

class Aerial():

    def __init__(self, pond: Pond, verb=False):
        self.pond = pond
        self.verb = verb

        if self.verb: print("Aerial has been initialized.")

    def style_attract(self, marker=dict(size=3, color='black', opacity=1, symbol="circle")):
        self.attract_marker_ = marker

        self.attract_styled_ = True
        return self

    def style_raid(self, marker=dict(size=3, color='black', opacity=1, symbol="cross")):
        self.raid_marker_ = marker

        self.raid_styled_ = True
        return self

    def attract(self, data):
        if (data is None) or (len(data) == 0):
            if self.verb: print("No data is given to attract.")
            return
        
        if not hasattr(self, "data_attract_"): self.data_attract_ = data
        else: self.data_attract_ = np.vstack((self.data_attract_, data))

        self.attracted_ = True
        if self.verb: print("Attract was successful.")

        return self
    
    def raid(self, data):
        if (data is None) or (len(data) == 0):
            if self.verb: print("No data is given to raid.")
            return
        
        if not hasattr(self, "data_raid_"): self.data_raid_ = data
        else: self.data_raid_ = np.vstack((self.data_raid_, data))

        self.raided_ = True
        if self.verb: print("Raid was successful.")

        return self
    
    def observe(self, pond_cmap="Viridis_r", pond_showscale=False, pond_opacity=.75, return_fig=True):

        # ensure style
        self.__style()

        som = self.pond.basin.som

        r, c = self.pond.basin.lattice_shape_
        gridX, gridY = np.linspace(0, r-1, r), np.linspace(0, c-1, c)
        gridXX, gridYY = np.meshgrid(gridX, gridY)

        fig = go.Figure()

        # pond surface
        fig.add_trace(go.Surface(
            x=gridXX, y=gridYY, z=np.zeros_like(gridXX),
            surfacecolor=self.pond.basin.distmap_,
            colorscale=pond_cmap,
            showscale=pond_showscale,
            opacity=pond_opacity
        ))

        # Attract points (normal)
        if hasattr(self, "data_attract_") and len(self.data_attract_) > 0:

            bmu_x, bmu_y, bmu_dist = [], [], []
            winmap_normal = som.win_map(self.data_attract_)

            for (i, j), val_list in winmap_normal.items():
                for val in val_list:
                    bmu_x.append(j)
                    bmu_y.append(i)
                    bmu_dist.append(np.linalg.norm([val] - som.quantization([val])))

            ## points
            fig.add_trace(go.Scatter3d(
                x=bmu_x, y=bmu_y, z=bmu_dist,
                mode='markers',
                marker=self.attract_marker_
            ))

            ## projection lines
            for xi, yi, zi in zip(bmu_x, bmu_y, bmu_dist):
                fig.add_trace(go.Scatter3d(
                    x=[xi, xi], y=[yi, yi], z=[0, zi],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    opacity=1,
                    showlegend=False
                ))

        # Raid points (abnormal)
        if hasattr(self, "data_raid_") and len(self.data_raid_) > 0:

            bmu_x, bmu_y, bmu_dist = [], [], []
            winmap_abnormal = som.win_map(self.data_raid_)

            for (i, j), val_list in winmap_abnormal.items():
                for val in val_list:
                    bmu_x.append(j)
                    bmu_y.append(i)
                    bmu_dist.append(np.linalg.norm([val] - som.quantization([val])))

            ## points
            fig.add_trace(go.Scatter3d(
                x=bmu_x, y=bmu_y, z=bmu_dist,
                mode='markers',
                marker=self.raid_marker_
            ))

            ## projection lines
            for xi, yi, zi in zip(bmu_x, bmu_y, bmu_dist):
                fig.add_trace(go.Scatter3d(
                    x=[xi, xi], y=[yi, yi], z=[0, zi],
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    opacity=1,
                    showlegend=False
                ))

        fig.update_layout(
            scene = dict(
                zaxis = dict(range=[0, None]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            )
        )

        if return_fig: return fig
        else: fig.show()

    def __style(self):
        if not hasattr(self, "attract_styled_") or self.attract_styled_ is False:
            self.style_attract()
        if not hasattr(self, "raid_styled_") or self.raid_styled_ is False:
            self.style_raid()
