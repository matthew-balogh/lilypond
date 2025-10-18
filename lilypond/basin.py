from minisom import MiniSom
from typing import Literal
from matplotlib.colors import ListedColormap

class Basin:

    def __init__(self, som: MiniSom, data, verb=False):
        self.som = som
        self.data = data
        self.verb = verb

        if self.verb: print("Basin has been initialized.")

    def prepare(self, neighbor_distance_scaling: Literal["sum", "mean"]="mean"):
        hitmap = self.som.activation_response(self.data).astype(int)
        distmap = self.som.distance_map(scaling=neighbor_distance_scaling)
        lattice_shape = hitmap.shape

        self.hitmap_ = hitmap
        self.distmap_ = distmap
        self.lattice_shape_ = lattice_shape
        self.rows_, self.cols_ = lattice_shape

        self.cmapWaterBlue_ = ListedColormap(['royalblue'])

        self.prepared_ = True
        if self.verb: print("Basin has been prepared.")

        return self

    def pond(self):
        from lilypond.pond import Pond
        self.__assert_prepared()
        return Pond(self, self.verb)
    
    def legacy_pond(self):
        from lilypond.legacy_pond import LegacyPond
        self.__assert_prepared()
        return LegacyPond(self, self.verb)
    
    def __assert_prepared(self):
        assert hasattr(self, "prepared_") and self.prepared_ is True, "The Basin must be prepared before creating a pond."
