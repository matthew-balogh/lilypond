import numpy as np
import plotly.graph_objects as go

from lilypond.pond import Pond


class RaidedPond():
    """
    Raids the given pond with the given data points. Be they anomalous or normal, that is intruders or beneficiaries of the pond.
    """

    def __init__(self, pond: Pond, verb=False):
        self.pond = pond
        self.verb = verb

        if self.verb: print("RaidedPond is initialized.")

    def clean(self):
        if hasattr(self, "data_normal_"): del self.data_normal_
        if hasattr(self, "data_abnormal_"): del self.data_abnormal_
        if hasattr(self, "raided_"): del self.raided_

        if self.verb: print("RaidedPond has been cleaned.")
    
    def raid(self, data, abnormal = False):
        if (data is None) or (len(data) == 0):
            if self.verb: print("No data is given to raid the pond with.")
            return

        if not abnormal:
            if not hasattr(self, "data_normal_"): self.data_normal_ = data
            else: self.data_normal_ = np.vstack((self.data_normal_, data))

        else:
            if not hasattr(self, "data_abnormal_"): self.data_abnormal_ = data
            else: self.data_abnormal_ = np.vstack((self.data_abnormal_, data))


        self.raided_ = True
        if self.verb: print("RaidedPond has been raid.")

        return self
    
    def visualize(self,
                  pond_cmap="Viridis_r", pond_showscale=False, pond_opacity=.75,
                  normal_marker=dict(size=3, color='blue', opacity=1, symbol="circle"),
                  abnormal_marker=dict(size=3, color='red', opacity=1, symbol="cross"),
                  return_fig=True):
        if (not hasattr(self, "raided_") or not self.raided_) and self.verb: print("Pond has not yet been raided.")

        som = self.pond.som
        distmap = som.distance_map()

        r, c = self.pond.lattice_shape_
        gridX, gridY = np.linspace(0, c-1, c), np.linspace(0, r-1, r)
        gridXX, gridYY = np.meshgrid(gridX, gridY)

        fig = go.Figure()

        # pond surface
        fig.add_trace(go.Surface(
            x=gridXX, y=gridYY, z=np.zeros_like(gridXX),
            surfacecolor=np.flipud(distmap),
            colorscale=pond_cmap,
            showscale=pond_showscale,
            opacity=pond_opacity
        ))

        # Raid points (normal)
        if hasattr(self, "data_normal_") and len(self.data_normal_) > 0:

            bmu_x, bmu_y, bmu_dist = [], [], []
            winmap_normal = som.win_map(self.data_normal_)

            for (i, j), val_list in winmap_normal.items():
                for val in val_list:
                    bmu_x.append(i)
                    bmu_y.append(j)
                    bmu_dist.append(np.linalg.norm([val] - som.quantization([val])))


            ## points
            fig.add_trace(go.Scatter3d(
                x=bmu_x, y=bmu_y, z=bmu_dist,
                mode='markers',
                marker=normal_marker
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
        if hasattr(self, "data_abnormal_") and len(self.data_abnormal_) > 0:

            bmu_x, bmu_y, bmu_dist = [], [], []
            winmap_abnormal = som.win_map(self.data_abnormal_)

            for (i, j), val_list in winmap_abnormal.items():
                for val in val_list:
                    bmu_x.append(i)
                    bmu_y.append(j)
                    bmu_dist.append(np.linalg.norm([val] - som.quantization([val])))


            ## points
            fig.add_trace(go.Scatter3d(
                x=bmu_x, y=bmu_y, z=bmu_dist,
                mode='markers',
                marker=abnormal_marker
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
                zaxis = dict(range=[0, None])
            )
        )

        self.clean()

        if return_fig: return fig
        else: fig.show()
