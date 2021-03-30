import os
import numpy as np
from osgeo import gdal
from typing import Union

from vyperdatum.core import VyperCore


class VyperPoints(VyperCore):
    """
    An extension of VyperCore with methods for dealing with point data.  Currently not much here, just some examples
    of exporting and storing the transformed data.
    """

    def __init__(self,  vdatum_directory: str = None):
        # ensure that vdatum_directory is passed in the first time this is run, to store that path
        super().__init__(vdatum_directory)
        self.x = None
        self.y = None
        self.z = None
        self.unc = None
        self.region_index = None

    def transform_points(self, input_datum: Union[str, int], output_datum: str, x: np.array, y: np.array,
                         z: np.array = None, include_vdatum_uncertainty: bool = True, include_region_index: bool = False):
        """
        Run transform_dataset to get the vertical transformed result / 3d transformed result.

        See core.VyperCore.transform_dataset

        Parameters
        ----------
        input_datum
            either a string identifier (ex: 'nad83', 'mllw') or epsg code for 3d transformation
        output_datum
            a string identifier (ex: 'nad83', 'mllw')
        x
            longitude of the input data
        y
            latitude of the input data
        z
            optional, depth value of the input data, if not provided will use all zeros include_vdatum_uncertainty
        include_vdatum_uncertainty
            if True, will return the combined separation uncertainty for each point
        include_region_index
            if True, will return the integer index of the region used for each point
        """

        self.set_region_by_bounds(min(x), min(y), max(x), max(y))
        self.set_input_datum(input_datum)
        self.set_output_datum(output_datum)
        self.x, self.y, self.z, self.unc, self.region_index = self.transform_dataset(x, y, z,
                                                                                     include_vdatum_uncertainty=include_vdatum_uncertainty,
                                                                                     include_region_index=include_region_index)

    def export_to_csv(self, output_file: str, delimiter: str = ' '):
        """
        Export all point variables to csv.  Includes uncertainty and region index if that data is contained in this class.

        Parameters
        ----------
        output_file
            the file path to the output file you want to write
        delimiter
            optional, delimiter character if you don't want space delimited data
        """

        dset_vars = [dvar for dvar in [self.x, self.y, self.z, self.unc, self.region_index] if dvar is not None]
        dset = np.c_[dset_vars]
        np.savetxt(output_file, dset, delimiter=delimiter, comments='')
