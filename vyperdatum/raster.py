from time import perf_counter
import os
import numpy as np
from osgeo import gdal
from pyproj import Transformer, CRS


from vyperdatum.core import VyperCore
from vyperdatum.vypercrs import VerticalPipelineCRS


class VyperRaster(VyperCore):
    """
    Transform raster methods
    """

    def __init__(self, input_file: str = None, vdatum_directory: str = None):
        super().__init__(vdatum_directory)
        self.input_file = input_file
        self.geotransform = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.resolution_x = None
        self.resolution_y = None
        self.width = None
        self.height = None

        self.geographic_min_x = None
        self.geographic_min_y = None
        self.geographic_max_x = None
        self.geographic_max_y = None

        self.layers = []
        self.layernames = []
        self.nodatavalue = []

        self.raster_vdatum_sep = None
        self.raster_vdatum_uncertainty = None
        self.raster_vdatum_region_index = None

        if input_file:
            self.initialize()

    def initialize(self, input_file: str = None):
        """
        Get all the data we need from the input raster.  This is run automatically on instancing this class, if an input
        file is provided then.  Otherwise, use this method and provide a gdal supported file to initialize.

        Parameters
        ----------
        input_file
            file path to a gdal supported raster file
        """

        if input_file:  # can re-initialize by passing in a new file here
            self.input_file = input_file

        ofile = gdal.Open(self.input_file)
        if not ofile:
            raise ValueError(f'Unable to open {self.input_file} with gdal')

        self.layers = [ofile.GetRasterBand(i + 1).ReadAsArray() for i in range(ofile.RasterCount)]
        self.nodatavalue = [ofile.GetRasterBand(i + 1).GetNoDataValue() for i in range(ofile.RasterCount)]
        self.layernames = [ofile.GetRasterBand(i + 1).GetDescription() for i in range(ofile.RasterCount)]

        # readasarray doesn't seem to handle gdal nodatavalue NaN
        for lyr, ndv in zip(self.layers, self.nodatavalue):
            lyr[lyr == ndv] = np.nan

        # geotransform in this format [x origin, x pixel size, x rotation, y origin, y rotation, -y pixel size]
        self.geotransform = ofile.GetGeoTransform()
        self.min_x, self.resolution_x, _, self.max_y, _, self.resolution_y = self.geotransform
        self.width, self.height = ofile.RasterXSize, ofile.RasterYSize
        self.max_x = self.min_x + self.width * self.resolution_x
        self.min_y = self.max_y + self.height * self.resolution_y
        self.resolution_y = abs(self.resolution_y)  # store this as positive for future use

        # if this is a raster with a vyperdatum crs, try to automatically build the input crs
        input_crs = ofile.GetSpatialRef()
        if input_crs.GetAttrValue('AUTHORITY') == 'EPSG':
            epsg = input_crs.GetAttrValue('AUTHORITY', 1)
            if epsg:
                self.set_input_datum(int(epsg))
        ofile = None

    def _get_elevation_layer_index(self):
        """
        Find the elevation layer index

        Returns
        -------
        int
            integer index in self.layernames for the elevation layer, -1 if it does not exist
        """
        check_layer_names = [lname.lower() for lname in self.layernames]
        if 'depth' in check_layer_names:
            depth_idx = check_layer_names.index('depth')
        elif 'elevation' in check_layer_names:
            depth_idx = check_layer_names.index('elevation')
        else:
            depth_idx = -1
            print(f'Unable to find depth or elevation layer by name, layers={check_layer_names}')
        return depth_idx

    def _get_uncertainty_layer_index(self):
        """
        Find the uncertainty layer index

        Returns
        -------
        int
            integer index in self.layernames for the uncertainty layer, -1 if it does not exist
        """
        check_layer_names = [lname.lower() for lname in self.layernames]
        if 'uncertainty' in check_layer_names:
            unc_idx = check_layer_names.index('uncertainty')
        elif 'vertical uncertainty' in check_layer_names:
            unc_idx = check_layer_names.index('vertical uncertainty')
        else:
            unc_idx = -1
            print(f'Unable to find uncertainty or vertical uncertainty layer by name, layers={check_layer_names}')
        return unc_idx

    def _get_contributor_layer_index(self):
        """
        Find the contributor layer index

        Returns
        -------
        int
            integer index in self.layernames for the contributor layer, -1 if it does not exist
        """
        check_layer_names = [lname.lower() for lname in self.layernames]
        if 'contributor' in check_layer_names:
            cont_idx = check_layer_names.index('contributor')
        else:
            cont_idx = -1
            print(f'Unable to find contributor layer by name, layers={check_layer_names}')
        return cont_idx

    def set_input_datum(self, input_datum: int):
        """
        An additional task is run on setting the input datum.  We first need to determine the nad83 geographic coordinates
        to determine which vdatum regions apply (set_region_by_bounds).  Afterwards we call the vypercore set_input_datum
        process.

        One difference between this method and all other VyperCore instances, is that VyperRaster relies on an EPSG
        input datum.  We must have the input datum to transform the extents to NAD83 to determine vdatum region.

        Parameters
        ----------
        input_datum
            EPSG code for the input datum of the raster
        """

        if not self.min_x or not self.min_y or not self.max_x or not self.max_y:
            raise ValueError('You must initialize first, before setting input datum, as we transform the extents here')

        # epsg which lets us transform, otherwise assume raster extents are geographic
        # transform the raster extents so we can use them to find the vdatum regions
        transformer = Transformer.from_crs(CRS.from_epsg(input_datum), CRS.from_epsg(6319), always_xy=True)
        self.geographic_min_x, self.geographic_min_y, _ = transformer.transform(self.min_x, self.min_y, 0)
        self.geographic_max_x, self.geographic_max_y, _ = transformer.transform(self.max_x, self.max_y, 0)
        self.set_region_by_bounds(self.geographic_min_x, self.geographic_min_y, self.geographic_max_x, self.geographic_max_y)

        # run the core process
        super().set_input_datum(input_datum)

    def get_datum_sep(self, sampling_distance: float, include_region_index: bool = False):

        """
        Use the provided raster and pipeline to get the separation over the raster area.

        Parameters
        ----------
        sampling_distance
            interval in meters that you want to sample the raster coordinates to get the sep value
        include_region_index
            if True, will return the integer index of the region used for each point
        """

        if self.regions is None:
            raise ValueError('Initialization must have failed, re-initialize with a new gdal supported file')
        if not self.in_crs:
            raise ValueError('Input datum must be set with the set_input_datum method before operation')
        if not self.out_crs:
            raise ValueError('Output datum must be set with the set_output_datum method before operation')

        xx_sampled, yy_sampled, x_range, y_range = sample_array(self.min_x, self.max_x, self.min_y, self.max_y, sampling_distance)
        # get sep value across all regions for all grid cells at sampling distance
        geo_xx, geo_yy, sep, sep_unc, sep_region_index = self.transform_dataset(xx_sampled.ravel(), yy_sampled.ravel(),
                                                                                include_vdatum_uncertainty=True,
                                                                                include_region_index=include_region_index)
        # ignore the geographic return from transform, keep on with the raster coordinates
        valid_count = np.count_nonzero(sep)
        invalid_count = sep.size - valid_count
        print(f'Found {valid_count} valid cells at {sampling_distance} m spacing, unable to determine sep value for '
              f'{invalid_count} cells')

        # get grids of equal size to the raster layers we are transforming
        if self.resolution_x != self.resolution_y:
            raise NotImplementedError('get_datum_sep: This currently only works when resx and resy are the same')
        raster_sep_x, raster_sep_y, _, _ = sample_array(self.min_x, self.max_x, self.min_y, self.max_y, self.resolution_x,
                                                        center=False)
        # bin the raster cell locations to get which sep value applies
        x_bins = np.digitize(raster_sep_x.ravel(), x_range[:-1])
        y_bins = np.digitize(raster_sep_y.ravel(), y_range[:-1])

        sep = sep.reshape(xx_sampled.shape)
        self.raster_vdatum_sep = sep[y_bins, x_bins].reshape(self.height, self.width).astype(np.float32)
        if sep_unc is not None:
            sep_unc = sep_unc.reshape(xx_sampled.shape)
            self.raster_vdatum_uncertainty = sep_unc[y_bins, x_bins].reshape(self.height, self.width).astype(np.float32)
        if sep_region_index is not None:
            sep_region_index = sep_region_index.reshape(xx_sampled.shape)
            self.raster_vdatum_region_index = sep_region_index[y_bins, x_bins].reshape(self.height, self.width)

    def apply_sep(self, allow_points_outside_coverage: bool = False):
        """
        After getting the datum separation model from vdatum, use this method to apply the separation and added
        separation uncertainty.

        If allow_points_outside_coverage is True, this will pass through z values that are outside of vdatum coverage,
        but add additional uncertainty

        Parameters
        ----------
        allow_points_outside_coverage
            if True, allows through points outside of vdatum coverage

        Returns
        -------
        tuple
            tuple of layers, including elevation, uncertainty and possibly contributor
        tuple
            tuple of layer names for each layer in returned layers
        tuple
            tuple of layer nodata value for each layer in returned layers
        """

        if self.raster_vdatum_sep is None:
            raise ValueError('Unable to find sep model, make sure you run get_datum_sep first')
        elevation_layer_idx = self._get_elevation_layer_index()
        uncertainty_layer_idx = self._get_uncertainty_layer_index()
        contributor_layer_idx = self._get_contributor_layer_index()

        if elevation_layer_idx == -1:
            raise ValueError('Unable to find elevation layer')
        if uncertainty_layer_idx == -1:
            print('Unable to find uncertainty layer, uncertainty will be entirely based off of vdatum sep model')

        elevation_layer = self.layers[elevation_layer_idx]
        layernames = [self.layernames[elevation_layer_idx]]
        layernodata = [self.nodatavalue[elevation_layer_idx]]
        uncertainty_layer = None
        contributor_layer = None
        if uncertainty_layer_idx:
            uncertainty_layer = self.layers[uncertainty_layer_idx]
            layernames.append(self.layernames[uncertainty_layer_idx])
            layernodata.append(self.nodatavalue[uncertainty_layer_idx])
        else:
            layernames.append('Uncertainty')
            layernodata.append(np.nan)
        if contributor_layer_idx:
            contributor_layer = self.layers[contributor_layer_idx]
            layernames.append(self.layernames[contributor_layer_idx])
            layernodata.append(self.nodatavalue[contributor_layer_idx])

        missing_idx = np.isnan(self.raster_vdatum_sep)
        missing_count = np.count_nonzero(missing_idx)
        missing_idx = np.where(missing_idx)
        print(f'Applying vdatum separation model to {self.raster_vdatum_sep.size} total points')

        final_elevation_layer = elevation_layer - self.raster_vdatum_sep
        if uncertainty_layer:
            final_uncertainty_layer = uncertainty_layer + self.raster_vdatum_uncertainty
        else:
            final_uncertainty_layer = self.raster_vdatum_uncertainty

        if not allow_points_outside_coverage:
            print(f'applying nodatavalue to {missing_count} points that are outside of vdatum coverage')
            final_elevation_layer[missing_idx] = self.nodatavalue[elevation_layer_idx]
            final_uncertainty_layer[missing_idx] = self.nodatavalue[uncertainty_layer_idx]
            if contributor_layer:
                contributor_layer[missing_idx] = self.nodatavalue[contributor_layer_idx]
        else:
            print(f'Allowing {missing_count} points that are outside of vdatum coverage, using CATZOC D vertical uncertainty')
            z_values = final_elevation_layer[missing_idx]
            u_values = 3 - 0.06 * z_values
            u_values[np.where(z_values > 0)] = 3.0
            final_uncertainty_layer[missing_idx] = u_values

        layers = (final_elevation_layer, final_uncertainty_layer, contributor_layer)
        return layers, layernames, layernodata

    def transform_raster(self, input_datum: int, output_datum: str, sampling_distance: float,
                         include_region_index: bool = False, allow_points_outside_coverage: bool = False,
                         output_file: str = None):
        """
        Main method of this class, contains all the other methods and allows you to transform the source raster to a
        different vertical datum using VDatum.

        Parameters
        ----------
        input_datum
            EPSG code for the datum of the source raster, only necessary if the EPSG is not encoded in the source SpatialReference
        output_datum
            datum identifier string, see pipeline.datum_definition keys for possible options for string
        sampling_distance
            interval in meters that you want to sample the raster coordinates to get the sep value
        include_region_index
            if True, will return the integer index of the region used for each point
        allow_points_outside_coverage
            if True, allows through points outside of vdatum coverage
        output_file
            if provided, writes the new raster to geotiff

        Returns
        -------
        tuple
            tuple of layers, including elevation, uncertainty and possibly contributor
        tuple
            tuple of layer names for each layer in returned layers
        tuple
            tuple of layer nodata value for each layer in returned layers
        """

        if input_datum:
            self.set_input_datum(input_datum)
        if output_datum:
            self.set_output_datum(output_datum)

        if self.regions is None:
            raise ValueError(f'Unable to find regions for raster using ({self.geographic_min_x},{self.geographic_min_y}), '
                             f'({self.geographic_max_x},{self.geographic_max_y})')

        start_cnt = perf_counter()
        print(f'Begin work on {os.path.basename(self.input_file)}')
        self.get_datum_sep(sampling_distance, include_region_index=include_region_index)
        layers, layernames, layernodata = self.apply_sep(allow_points_outside_coverage=allow_points_outside_coverage)
        if output_file:
            if layernodata[2]:  # contributor
                tiffdata = np.c_[layers[0], layers[1], layers[2]]
            else:
                tiffdata = np.c_[layers[0], layers[1]]
            tiffdata = np.round(tiffdata, 3)
            self._write_gdal_geotiff(output_file, tiffdata, layernames, layernodata)
        end_cnt = perf_counter()
        print(f'Raster transformation complete: Elapsed time {end_cnt - start_cnt} seconds')
        return layers, layernames, layernodata
        
    def _write_gdal_geotiff(self, outfile: str, data: tuple, band_names: tuple, nodatavalue: tuple):
        """
        Build a geotiff from the transformed raster data

        Parameters
        ----------
        outfile
            output file that we write the raster data to
        data
            arrays for each layer that we want to write to file
        band_names
            names for each layer that we want to write to file
        nodatavalue
            nodatavalues for each layer that we want to write to file
        """

        numlyrs, rows, cols = data.shape
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(outfile, cols, rows, numlyrs, gdal.GDT_Float32)
        out_raster.SetGeoTransform(self.geotransform)
        for lyr_num in range(numlyrs):
            outband = out_raster.GetRasterBand(lyr_num + 1)
            outband.SetNoDataValue(nodatavalue[lyr_num])
            outband.SetDescription(band_names[lyr_num])
            outband.WriteArray(data[lyr_num])
            outband = None
        out_raster.SetProjection(self.out_crs.to_wkt())
        out_raster = None


def sample_array(min_x: float, max_x: float, min_y: float, max_y: float, sampling_distance: float, center: bool = True):
    """
    Build coordinates for a sampled grid using the extents of the main grid.  The new grid will have the same extents,
    but be sampled at sampling_distance.

    Parameters
    ----------
    min_x
        minimum x value of the grid
    max_x
        maximum x value of the grid
    min_y
        minimum y value of the grid
    max_y
        maximum y value of the grid
    sampling_distance
        distance in grid units to sample
    center
        optional, if True returns the sampled grid coordinates at the center of the sampled grid, rather than the edges

    Returns
    -------
    np.ndarray
        2d array of x values for the new sampled grid
    np.ndarray
        2d array of y values for the new sampled grid
    np.array
        1d array of the x values for one column of the grid, i.e. the x range of the grid
    np.array
        1d array of the y values for one column of the grid, i.e. the y range of the grid
    """

    nx = np.ceil((max_x - min_x) / sampling_distance).astype(int)
    ny = np.ceil((max_y - min_y) / sampling_distance).astype(int)
    x_sampled = np.linspace(min_x, max_x, nx)
    y_sampled = np.linspace(min_y, max_y, ny)

    if center:
        # sampled coords are now the cell borders, we want cell centers
        x_sampled = x_sampled[:-1] + (sampling_distance / 2)
        y_sampled = y_sampled[:-1] + (sampling_distance / 2)

    # grid with yx order to match gdal
    yy, xx = np.meshgrid(y_sampled, x_sampled, indexing='ij')

    return xx, yy, x_sampled, y_sampled
