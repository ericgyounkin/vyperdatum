import os
import numpy as np
from osgeo import gdal

from vyperdatum.core import VyperCore

gdal.UseExceptions()


class VyperPoints(VyperCore):
    def __init__(self,  vdatum_directory: str = None):
        super().__init__(vdatum_directory)
        self.x = None
        self.y = None
        self.z = None
        self.unc = None

    def convert_points(self, x: np.array, y: np.array, z: np.array = None, include_vdatum_uncertainty: bool = True):
        self.x, self.y, self.z, self.unc = self.transform_dataset(x, y, z, include_vdatum_uncertainty=include_vdatum_uncertainty)

    def export_to_csv(self, output_file: str, delimiter: str = ' '):
        if self.unc:
            dset = np.c_[self.x, self.y, self.z, self.unc]
        else:
            dset = np.c_[self.x, self.y, self.z]
        np.savetxt(output_file, dset, delimiter=delimiter, comments='')
