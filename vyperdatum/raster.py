# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 07:55:27 2021

@author: grice
"""

'''
Read raster (gdal) with vertical and horizontal coordinate systems
Write raster (gdal) with vertical and horizontal coordinate systems
    Cloud Optimized GeoTIFF should be one of these
    Allow for the writing of abstract metadata string
Create separation model for raster
    for a region
        use a core object to get gridded points
        interpolate to those points
        try to transform the remaining points
Apply separation model to raster input
Enforce logic about reprojecting rasters vs projecting / unprojecting
    return points for rasters that are being (un)projected?
'''
from time import perf_counter
import os, datetime
import numpy as np
from scipy.interpolate import griddata
import gdal

gdal.UseExceptions()

from vyperdatum.core import VyperCore


class VyperRaster(VyperCore):
    """
    Transform raster methods
    """
    def __init__(self, input_file: str, output_datum: str, vdatum_directory: str = None):
        super().__init__(output_datum, vdatum_directory)
        self.input_file = input_file
        self.geotransform = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.width = None
        self.height = None
        self.input_epsg = None

        self.layers = []
        self.layernames = []
        self.nodatavalue = []

        self.initialize()

    def initialize(self, input_file: str = None):
        if input_file:  # can re-initialize by passing in a new file here
            self.input_file = input_file

        ofile = gdal.Open(self.input_file)
        if not ofile:
            raise ValueError(f'Unable to open {self.input_file} with gdal')

        is_epsg = ofile.GetSpatialRef().GetAttrValue("AUTHORITY", 0) == 'EPSG'
        if is_epsg:
            self.input_epsg = ofile.GetSpatialRef().GetAttrValue("AUTHORITY", 1)
        else:
            raise NotImplementedError('Expect the input tif to have an EPSG attribute in the SpatialReference WKT string')

        self.layers = [ofile.GetRasterBand(i + 1).ReadAsArray() for i in range(ofile.RasterCount)]
        self.nodatavalue = [ofile.GetRasterBand(i + 1).GetNoDataValue() for i in range(ofile.RasterCount)]
        self.layernames = [ofile.GetRasterBand(i + 1).GetDescription() for i in range(ofile.RasterCount)]

        self.geotransform = ofile.GetGeoTransform()
        self.min_x = self.geotransform[0]
        self.max_y = self.geotransform[3]
        self.max_x = self.min_x + self.geotransform[1] * ofile.RasterXSize
        self.min_y = self.max_y + self.geotransform[5] * ofile.RasterYSize
        self.height = self.max_y - self.min_y
        self.width = self.max_x - self.min_x
        self.set_region_by_bounds(self.min_x, self.min_y, self.max_x, self.max_y)
        ofile = None

    def get_datum_sep(self, sampling_distance):
        """
        Use the provided raster and pipeline to get the separation over the raster area.

        Returns
        -------
        sep

        """
        if self.regions is None:
            raise ValueError('Initialization must have failed, re-initialize with a new gdal supported file')
        new_crs = None
        
        self._sample_array(sampling_distance)

        for region in self.regions:
            start = perf_counter()
            # first convert sampled raster
            try:
                result, new_crs = self.run_pipeline(xx.flatten(), yy.flatten(), zz.flatten(), self.input_crs, region)
            except pyproj.ProjError as e:
                print_paths = '\n'.join(pyproj.datadir.get_data_dir().split(';'))
                print(f'Proj pipeline failed. pyproj paths: \n{print_paths}')
                raise e
            end = perf_counter()
            print(f'Transforming {len(yy.flatten())} points took {end - start} seconds for {region}')
            vals = result[2].flatten()
            valid_idx = np.squeeze(np.argwhere(~np.isinf(vals)))

            if len(valid_idx) == 0:
                print(f'No valid points found from gridding in {region}. Putting all points through proj pipeline directly.')
            else:
                print(f'interpolating to original grid for {region}')
                start = datetime.now()
                points = np.array([result[1][valid_idx], result[0][valid_idx]]).T
                valid_vals = vals[valid_idx]
                try:
                    region_sep = griddata(points, valid_vals, (y, x))
                    idx = ~np.isnan(region_sep)
                    sep[idx] = region_sep[idx]
                except Exception as error:
                    msg = f'{error.__class__.__name__}: {error}\n\n'
                    print(msg)
                    return None, None
                dt = datetime.now() - start
                print(f'Interpolating {len(y.flatten())} points took {dt} seconds for {region}')
            # find all points where a separation is desired but not yet found.
            missing_idx = np.where(np.isnan(sep) & (self.input_elevation != self.input_nodata))
            num_remaining = len(missing_idx[0])
            # attempt to transform the remaining points directly
            if num_remaining > 0:
                print(f'Transforming {num_remaining} remaining points not found in interpolated region for {region}.')
                missing, new_crs = self.run_pipeline(x[missing_idx], y[missing_idx], np.zeros(num_remaining), self.input_crs, region)
                missing_sep = missing[2]
                still_inf_idx = np.where(np.isinf(missing_sep))
                missing_sep[still_inf_idx] = np.nan
                sep[missing_idx] = missing_sep
        return sep, new_crs
    
    def _sample_array(self, transform_sampling_distance):
        """Sample the raster to get a subset of points and then interpolate"""
        # create empty sampled array representing original raster
        sy, sx = self.height, self.width
        resy, resx = self.geotransform[4], self.geotransform[0]
        y0, x0 = self.geotransform[5], self.geotransform[2]
        y1 = y0 + sy * resy
        x1 = x0 + sx * resx
        nx = np.round(np.abs((x1 - x0) / transform_sampling_distance)).astype(int)
        ny = np.round(np.abs((y1 - y0) / transform_sampling_distance)).astype(int)
        x_sampled = np.linspace(x0, x1, nx)
        y_sampled = np.linspace(y0, y1, ny)
        yy, xx = np.meshgrid(y_sampled, x_sampled, indexing='ij')
        zz = np.zeros(yy.shape)
        # raster to points with points at corners rather than cell centers
        y, x = np.mgrid[y0:y1:resy, x0:x1:resx]
        # make empty separation raster
        sep = np.full(y.shape, np.nan)

    def apply_sep(self, sep, new_crs, pass_untransformed=True):
        """


        Parameters
        ----------
        outfile : TYPE
            DESCRIPTION.
        sep : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        missing_idx = np.where(np.isnan(sep) & (self.input_elevation != self.input_nodata))
        num_nan = len(missing_idx[0])
        print(f'Found {num_nan} values that are outside the separation model')
        if pass_untransformed:
            z_values = self.input_elevation[missing_idx[0], missing_idx[1]]
            # make the new uncertainty
            u_values = 3 - 0.06 * z_values
            u_values[np.where(z_values > 0)] = 3.0
            uncert = self.input_uncertainty
            uncert[missing_idx] = u_values
            if num_nan > 0:
                print(f'Inserting {num_nan} values that were not transformed into new file, but applying CATZOC D vertical uncertainty.')
        elev = (self.input_elevation - sep)
        elev[np.where(np.isnan(elev))] = self.input_nodata
        if pass_untransformed:
            elev[missing_idx] = z_values
            self.input_elevation = elev
            self.input_uncertainty = uncert
        else:
            self.input_elevation[missing_idx[0], missing_idx[1]] = self.input_nodata
            self.input_uncertainty[missing_idx[0], missing_idx[1]] = self.input_nodata
            self.input_contributor[missing_idx[0], missing_idx[1]] = self.input_nodata

        tiffdata = np.stack([self.input_elevation, self.input_uncertainty])
        tiffdata = np.round(tiffdata.astype(np.float), decimals=2).astype(np.float32)
        tiffnames = self.input_band_names[:2]

        self.write_gdal_geotiff(self.output_path, tiffdata, new_crs, self.input_transform, self.input_nodata, tiffnames)

    def transform_raster(self):
        """


        Returns
        -------
        None.

        """

        if self.regions is None:
            raise ValueError('Initialization must have failed, re-initialize with a new gdal supported file')
        print(f'Begin work on {os.path.basename(self.input_file)}')
        if len(self.regions) > 0:
            sep, crs = self.get_datum_sep(100)
            self.apply_sep(sep, crs)
        else:
            print(f'no region found for {self.input_file}')
        
    def read_raster(self, infilename):
        """Read a raster"""
        pass
        
    def write_gdal_geotiff(self, outfile, data, pyproj_crs, transform, nodata, band_names):
    
        numlyrs, rows, cols = data.shape
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(outfile, cols, rows, numlyrs, gdal.GDT_Float32)
        outRaster.SetGeoTransform(transform.to_gdal())
        for lyr_num in range(numlyrs):
            outband = outRaster.GetRasterBand(lyr_num + 1)
            outband.SetNoDataValue(nodata)
            outband.SetDescription(band_names[lyr_num])
            outband.WriteArray(data[lyr_num])
        outRaster.SetProjection(pyproj_crs.to_wkt())
        # outRaster.SetMetadata({'TIFFTAG_IMAGEDESCRIPTION' : TIFFTAG_IMAGEDESCRIPTION,
        #                        'TIFFTAG_ARTIST': TIFFTAG_ARTIST,
        #                        'TIFFTAG_COPYRIGHT': TIFFTAG_COPYRIGHT,
        #                        })
        outband.FlushCache()