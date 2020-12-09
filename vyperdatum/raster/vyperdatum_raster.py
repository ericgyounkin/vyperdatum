# run in pydro38_test env
import os, sys, glob, logging, configparser
import argparse
from datetime import datetime
import numpy as np
from scipy.interpolate import griddata
import pyproj
from pyproj import Transformer
from pyproj.crs import CompoundCRS
import gdal
from gdal import ogr
import rasterio
import collections
gdal.UseExceptions()

LOGGER = logging.getLogger('vyperdatum')


class VyperRaster:
    """
    Operational Class, transforming raster
    """
    def __init__(self, input_file: str = None, output_path: str = None):
        """
        Include an input file and a path to an output, or optionally leave these as None to use the last stored value(s)

        Parameters
        ----------
        input_file
            Input raster to transform.  If none, will load from config (last inputfile)
        output_path
            Path to where you want to save the output.  If none will load from config (last output file).  If that is
            also not populated, will convert alongside the input_file
        """

        self.config = VyperConfig()
        self.input_file = ''
        self.input_path = ''
        self.output_path = ''
        
        if input_file is None:
            self.input_file = self.config.settings['inpath']
        else:
            self.config.settings['inpath'] = input_file
            self.input_file = input_file
        if output_path is None:
            self.output_path = self.config.settings['outpath']
            if self.output_path == '':
                if self.input_path:
                    self.output_path = os.path.join(os.path.split(self.input_file)[0], os.path.splitext(self.input_file)[0] + '_converted.tiff')
                self.config.settings['outpath'] = self.output_path
        else:
            self.config.settings['outpath'] = output_path
            self.output_path = output_path

        self.intersects = None

        self.input_elevation = None
        self.input_uncertainty = None
        self.input_contributor = None

        self.input_transform = None
        self.input_crs = None
        self.input_nodata = None
        self.input_profile = None
        self.input_band_names = None

        self._validate_inputs()
        self.config_logger()

    def _validate_inputs(self):
        """
        Ensure the input file exists and the output file directory also exists
        """

        if not os.path.exists(self.input_file) and self.input_file:
            raise ValueError('VyperDatum: {} does not exist'.format(self.input_file))
        basedir, output_name = os.path.split(self.output_path)
        if not os.path.exists(basedir) and self.output_path:
            raise ValueError('VyperDatum: base folder for {} does not exist: {}'.format(self.output_path, basedir))

        if self.input_file:
            with rasterio.open(self.input_file) as raster:
                self.input_elevation, self.input_uncertainty, self.input_contributor = raster.read()
                self.input_transform = raster.transform
                self.input_crs = raster.crs
                self.input_nodata = raster.nodata
                self.input_profile = raster.profile
                self.input_band_names = raster.descriptions
                expected_bandnames = ('Elevation', 'Uncertainty', 'Contributor')
                if self.input_band_names != expected_bandnames:
                    raise ValueError('VyperRaster: Expected {} band names, found {}'.format(expected_bandnames, self.input_band_names))

    def get_interesecting_vdatum_regions(self):
        """
        Find the vdatum regions that intersect the given data.
        """
        dataset = gdal.Open(self.input_file)
        is_raster = dataset.RasterCount > 0
        if is_raster:
            # get raster bounds
            crs = pyproj.CRS.from_wkt(dataset.GetProjectionRef())
            transform = dataset.GetGeoTransform()
            pixel_width = transform[1]
            pixel_height = transform[5]
            cols = dataset.RasterXSize
            rows = dataset.RasterYSize
            x0 = transform[0]
            y1 = transform[3]
            x1 = x0 + cols * pixel_width
            y0 = y1 - rows * pixel_height
            if crs.is_projected:
                unproject = pyproj.Proj(proj='utm', zone=19, ellps='WGS84')
                ul = unproject(x0, x1, inverse=True)
                ur = unproject(x1, y1, inverse=True)
                lr = unproject(x1, y0, inverse=True)
                ll = unproject(x0, y0, inverse=True)
            else:
                ul = (x0, y1)
                ur = (x1, y1)
                lr = (x1, y0)
                ll = (x0, y0)
            # build polygon from raster
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ul[0], ul[1])
            ring.AddPoint(ur[0], ur[1])
            ring.AddPoint(lr[0], lr[1])
            ring.AddPoint(ll[0], ll[1])
            ring.AddPoint(ul[0], ul[1])
            dataGeometry = ogr.Geometry(ogr.wkbPolygon)
            dataGeometry.AddGeometry(ring)
        else:
            raise NotImplementedError('Not handling XYZ data yet')
        dataset = None
        # see if the regions intersect with the provided geometries
        intersecting_regions = []
        for region in self.config.polygon_files:
            vector = ogr.Open(self.config.polygon_files[region])
            layer_count = vector.GetLayerCount()
            for m in range(layer_count):
                layer = vector.GetLayerByIndex(m)
                feature_count = layer.GetFeatureCount()
                for n in range(feature_count):
                    feature = layer.GetFeature(n)
                    feature_name = feature.GetField(0)
                    if feature_name[:15] == 'valid-transform':
                        valid_vdatum_poly = feature.GetGeometryRef()
                        if dataGeometry.Intersect(valid_vdatum_poly):
                            intersecting_regions.append(region)
            vector = None
        self.intersects = intersecting_regions

    def run_pipeline(self, xx, yy, zz, incrs, region_name):
        """
        MLLW to NAVD88
        """

        req_hcrs_epsg = 26919
        req_vcrs_epsg = 'mllw'
        out_vcrs_epsg = 5703

        # parse the provided CRS
        cmpd_incrs = CompoundCRS.from_wkt(incrs.to_wkt())
        if len(cmpd_incrs.sub_crs_list) == 2:
            inhcrs, invcrs = cmpd_incrs.sub_crs_list
            assert inhcrs.to_epsg() == req_hcrs_epsg
            assert invcrs.to_epsg() == req_vcrs_epsg
        elif not cmpd_incrs.is_vertical:
            assert incrs.to_epsg() == req_hcrs_epsg

        # build the output crs
        comp_crs = CompoundCRS(name="NAD83 UTM19 + NAVD88", components=[f"EPSG:{req_hcrs_epsg}", f"EPSG:{out_vcrs_epsg}"])
        # get the transform at the sparse points
        transformer = Transformer.from_pipeline(f'proj=pipeline \
                                                  step inv proj=utm zone=19 \
                                                  step inv proj=vgridshift grids={self.config.grid_files[f"{region_name}/mllw.gtx"]} \
                                                  step proj=vgridshift grids={self.config.grid_files[f"{region_name}/tss.gtx"]} \
                                                  step proj=utm zone=19')

        result = transformer.transform(xx=xx, yy=yy, zz=zz)
        self.config.logger.debug('Applying pipeline: {}'.format(transformer))
        return result, comp_crs

    def get_datum_sep(self, transform_sampling_distance):
        """
        Use the provided raster and pipeline to get the separation over the raster area.

        Returns
        -------
        sep

        """
        if self.intersects is None:
            self.get_interesecting_vdatum_regions()
        new_crs = None
        # with rasterio.open(self.input_file) as raster:
        #     elev = raster.read(1)
        #     transform = raster.transform
        #     crs = raster.crs
        #     nodata = raster.nodata
        #     profile = raster.profile
        # get sparse raster x and y positions
        sy, sx = self.input_profile['height'], self.input_profile['width']
        resy, resx = self.input_transform[4], self.input_transform[0]
        y0, x0 = self.input_transform[5], self.input_transform[2]
        y1 = y0 + sy * resy
        x1 = x0 + sx * resx
        nx = np.round(np.abs((x1 - x0) / transform_sampling_distance)).astype(int)
        ny = np.round(np.abs((y1 - y0) / transform_sampling_distance)).astype(int)
        x_sampled = np.linspace(x0, x1, nx)
        y_sampled = np.linspace(y0, y1, ny)
        yy, xx = np.meshgrid(y_sampled, x_sampled, indexing='ij')
        zz = np.zeros(yy.shape)
        y, x = np.mgrid[y0:y1:resy, x0:x1:resx]
        sep = np.full(y.shape, np.nan)

        for region in self.intersects:
            start = datetime.now()
            try:
                result, new_crs = self.run_pipeline(xx.flatten(), yy.flatten(), zz.flatten(), self.input_crs, region)
            except pyproj.ProjError as e:
                print_paths = '\n'.join(pyproj.datadir.get_data_dir().split(';'))
                self.config.logger.error('Proj pipeline failed. pyproj paths: \n{}'.format(print_paths))
                raise e
            dt = datetime.now() - start
            self.config.logger.debug(f'Transforming {len(yy.flatten())} points took {dt} seconds for {region}')
            vals = result[2].flatten()
            valid_idx = np.squeeze(np.argwhere(~np.isinf(vals)))

            if len(valid_idx) == 0:
                self.config.logger.debug(
                    'No valid points found from gridding in {region}. Putting all points through proj pipeline directly.')
            else:
                self.config.logger.debug('interpolating to original grid for {region}')
                start = datetime.now()
                points = np.array([result[1][valid_idx], result[0][valid_idx]]).T
                valid_vals = vals[valid_idx]
                try:
                    region_sep = griddata(points, valid_vals, (y, x))
                    idx = ~np.isnan(region_sep)
                    sep[idx] = region_sep[idx]
                except Exception as error:
                    msg = f'{error.__class__.__name__}: {error}\n\n'
                    self.config.logger.error(msg)
                    return None, None
                dt = datetime.now() - start
                self.config.logger.debug(f'Interpolating {len(y.flatten())} points took {dt} seconds for {region}')
            missing_idx = np.where(np.isnan(sep) & (self.input_elevation != self.input_nodata))
            num_nan = len(missing_idx[0])
            if num_nan > 0:
                missing, new_crs = self.run_pipeline(x[missing_idx], y[missing_idx], np.zeros(num_nan), self.input_crs, region)
                missing_sep = missing[2]
                still_inf_idx = np.where(np.isinf(missing_sep))
                missing_sep[still_inf_idx] = np.nan
                sep[missing_idx] = missing_sep
        return sep, new_crs

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
        self.config.logger.debug(f'Found {num_nan} values that are outside the separation model')
        if pass_untransformed:
            z_values = self.input_elevation[missing_idx[0], missing_idx[1]]
            # make the new uncertainty
            u_values = 3 - 0.06 * z_values
            u_values[np.where(z_values > 0)] = 3.0
            uncert = self.input_uncertainty
            uncert[missing_idx] = u_values
            if num_nan > 0:
                self.config.logger.debug(
                    f'Inserting {num_nan} values that were not transformed into new file, but applying CATZOC D vertical uncertainty.')
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

        write_gdal_geotiff(self.output_path, tiffdata, new_crs, self.input_transform, self.input_nodata, tiffnames)

    def transform_raster(self):
        """


        Returns
        -------
        None.

        """

        if self.intersects is None:
            self.get_interesecting_vdatum_regions()
        self.config.logger.debug(f'Begin work on {os.path.basename(self.input_file)}')
        if len(self.intersects) > 0:
            sep, crs = self.get_datum_sep(100)
            self.apply_sep(sep, crs)
        else:
            self.config.logger.debug(f'no region found for {self.input_file}')

    def config_logger(self):
        self.config.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter('[%(asctime)s] %(name)-9s %(levelname)-8s: %(message)s')
        
        if self.output_path:
            log_name = f'_to_NAVD88_{datetime.now():%Y%m%d_%H%M%S}.log'
            log_filename = os.path.join(os.path.dirname(self.output_path), log_name)
            log_file = logging.FileHandler(log_filename)
            log_file.setFormatter(log_formatter)
            log_file.setLevel(logging.DEBUG)
            self.config.logger.addHandler(log_file)

        output_console = logging.StreamHandler(sys.stdout)
        output_console.setFormatter(log_formatter)
        output_console.setLevel(logging.DEBUG)
        self.config.logger.addHandler(output_console)

    def close_logger(self):
        for handler in self.config.logger.handlers[:]:
            handler.close()
            self.config.logger.removeHandler(handler)

    def close(self):
        self.close_logger()


class VyperConfig:
    """
    Contains all files and configuration information for VyperDatum

    Saves as settings are changed to the appdata folder on your machine, something like:

    C:\\Users\\eyou1\\AppData\\Roaming\\vyperdatum\\vyperdatum.config
    """

    def __init__(self):
        self.settings_filepath = None  # file path to config file
        self.settings_object = None  # ConfigParser object generated in _load_settings
        self.default_settings = {'vdatum_directory': r'C:\VDatum', 'inpath': '', 'outpath': '', 'logger_name': ''}

        self.grid_files = {}
        self.polygon_files = {}

        self.settings = VyperSettings(self)
        self._load_settings()

        self.logger = None
        self._init_logger()

        self.gdal_version = None
        self.get_gdal_version()

    def _load_settings(self):
        """
        Call on initializing the class, will find an existing config file or create a new one if it does not exist
        Will then populate the settings attribute with existing settings or the default settings we wrote to the
        new config file.
        """

        vyperdatum_folder = os.path.join(os.getenv('APPDATA'), 'vyperdatum')
        vyperdatum_file = os.path.join(vyperdatum_folder, 'vyperdatum.config')
        self.settings_filepath = vyperdatum_file

        if os.path.exists(vyperdatum_file):
            self.settings_object, settings = read_from_config_file(vyperdatum_file)
        else:
            if not os.path.exists(vyperdatum_folder):
                print('generating appdata folder: {}'.format(vyperdatum_folder))
                os.makedirs(vyperdatum_folder)
            print('writing a new appdata config file: {}'.format(vyperdatum_file))
            self.settings_object, settings = create_new_config_file(vyperdatum_file, self.default_settings)

        # populate our settings with the new/existing settings found
        if settings is not None:
            for ky, val in settings.items():
                self.settings[ky] = val
                

    def _init_logger(self):
        """
        Build the logger instance
        """

        if 'logger_name' in self.settings and self.settings['logger_name'] != '':
            self.logger = logging.getLogger(self.settings['logger_name'])
        else:
            self.logger = logging.getLogger('vyperdatum')

    def _set_vdatum_directory(self, value):
        """
        Called when self.settings['vdatum_directory'] is updated.  We find all the grids and polygons in the vdatum
        directory and save the dicts to the attributes in this class.
        """

        # special case for vdatum directory, we want to give pyproj the new path if it isn't there already
        orig_proj_paths = pyproj.datadir.get_data_dir()
        if value not in orig_proj_paths:
            pyproj.datadir.append_data_dir(value)

        # also want to populate grids and polygons with what we find
        newgrids = get_gtx_grid_list(value)
        for gname, gpath in newgrids.items():
            self.grid_files[gname] = gpath
        newpolys = get_vdatum_region_polygons(value)
        for pname, ppath in newpolys.items():
            self.polygon_files[pname] = ppath

    def get_gdal_version(self):
        """
        Check the version of gdal imported to ensure it meets the requirements of this class
        """

        version = gdal.VersionInfo()
        major = int(version[0])
        minor = int(version[1:3])
        bug = int(version[3:5])
        if not (major == 3 and minor >= 1):
            msg = 'The version of GDAL must be >= 3.1.  Version found: {}.{}.{}'.format(major, minor, bug)
            self.logger.error(msg)
#            raise ValueError(msg)
        self.gdal_version = version


class VyperSettings(collections.UserDict):
    """
    Inherit from UserDict, provide a dictionary like object that we can save settings to and simultaneously update
    the configfile in VyperConfig.

    vyp = VyperConfig()
    vyp.settings['vdatum_directory'] = r'C:\vdatum_all_20201203\vdatum'
    vyp.settings_object.get('Default', 'vdatum_directory')
    Out[7]: 'C:\\vdatum_all_20201203\\vdatum'
    """

    def __init__(self, parent: VyperConfig = None):
        super().__init__()
        self.parent = parent

    def __setitem__(self, key: str, value: str):
        """
        setting an item in the dict will update the settings object and run any specific methods relevant to that key,
        for example, vdatum_directory changes call the _set_vdatum_directory in the VyperConfig object
        """

        super().__setitem__(key, value)

        if self.parent is None:
            raise ValueError('SettingsDict: parent object (VyperConfig) required for operation')

        # special case for vdatum directory
        if key == 'vdatum_directory':
            self.parent._set_vdatum_directory(value)

        # also update the config file
        self.parent.settings_object.set('Default', key, value)
        with open(self.parent.settings_filepath, 'w') as configfile:
            self.parent.settings_object.write(configfile)


def get_gtx_grid_list(vdatum_directory: str, logger: logging.Logger = None):
    """
    Search the vdatum directory to find all gtx files

    Parameters
    ----------
    vdatum_directory
        absolute folder path to the vdatum directory
    logger
        Logger instance if you wish to include it

    Returns
    -------
    dict
        dictionary of {grid name: grid path, ...}
    """

    search_path = os.path.join(vdatum_directory, '*/*.gtx')
    gtx_list = glob.glob(search_path)
    if len(gtx_list) == 0:
        errmsg = 'No GTX files found in the provided VDatum directory: {}'.format(vdatum_directory)
        if logger is None:
            print(errmsg)
        else:
            logger.warning(errmsg)
    grids = {}
    for gtx in gtx_list:
        gtx_path, gtx_file = os.path.split(gtx)
        gtx_path, gtx_folder = os.path.split(gtx_path)
        gtx_name = '/'.join([gtx_folder, gtx_file])
        gtx_subpath = os.path.join(gtx_folder, gtx_file)
        grids[gtx_name] = gtx_subpath
    return grids


def get_vdatum_region_polygons(vdatum_directory: str, logger: logging.Logger = None):
    """"
    Search the vdatum directory to find all kml files

    Parameters
    ----------
    vdatum_directory
        absolute folder path to the vdatum directory
    logger
        Logger instance if you wish to include it

    Returns
    -------
    dict
        dictionary of {kml name: kml path, ...}
    """

    search_path = os.path.join(vdatum_directory, '*/*.kml')
    kml_list = glob.glob(search_path)
    if len(kml_list) == 0:
        errmsg = 'No kml files found in the provided VDatum directory: {}'.format(vdatum_directory)
        if logger is None:
            print(errmsg)
        else:
            logger.warning(errmsg)
    geom = {}
    for kml in kml_list:
        kml_path, kml_file = os.path.split(kml)
        root_dir, kml_name = os.path.split(kml_path)
        geom[kml_name] = kml
    return geom


def read_from_config_file(filepath: str):
    """
    Read from the generated configparser file path, return the settings and the configparser object

    Parameters
    ----------
    filepath
        absolute filepath to the configparser object

    Returns
    -------
    configparser.ConfigParser
        configparser object used to read the file
    dict
        settings within the file
    """

    settings = {}
    config_file = configparser.ConfigParser()
    config_file.read(filepath)
    sections = config_file.sections()
    for section in sections:
        config_file_section = config_file[section]
        for key in config_file_section:
            settings[key] = config_file_section[key]
    return config_file, settings


def create_new_config_file(filepath: str, default_settings: dict):
    """
    Create a new configparser file, return the settings and the configparser object

    Parameters
    ----------
    filepath
        absolute filepath to the configparser object you wish to create
    default_settings
        new settings we want to write to the configparser file

    Returns
    -------
    configparser.ConfigParser
        configparser object used to read the file
    dict
        settings within the file
    """

    config = configparser.ConfigParser()
    config['Default'] = default_settings
    with open(filepath, 'w') as configfile:
        config.write(configfile)
    return config, default_settings


def write_gdal_geotiff(outfile, data, pyproj_crs, transform, nodata, band_names):

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
    
def main():
    parser = argparse.ArgumentParser(description='VyperRaster - Transform Rasters from one vertical datum to another')
    parser.add_argument('-i', '--input_data', type=str, help='either a directory of rasters or a single raster, expect files in tiff format')
    parser.add_argument('-o', '--output_path', type=str, help='either an output directory or file (for single raster conversion), transformed raster(s) as geotiff')
    parser.add_argument('-vd', '--vdatum', nargs='?', type=str, help='vdatum folder path, should only have to provide this once (saved in config file)')

    args = parser.parse_args()

    # set the vdatum directory if provided in the args
    vyp = VyperRaster()
    if args.vdatum:
        print('setting a new vdatum directory: {}'.format(args.vdatum))
        vyp.config.settings['vdatum_directory'] = args.vdatum

    # check for vdatum
    if not vyp.config.grid_files or not vyp.config.polygon_files:
        print('Unable to find vdatum files using {}, try providing a new vdatum directory using -vd argument'.format(vyp.config.settings['vdatum_directory'])) 
        return
    
    # run the transform(s), this could probably be cleaned up, but we basically cover every combination of directory/file path provided
    err = False

    infil, outfil = args.input_data, args.output_path
    if not infil or not outfil:
        print('No infile or outfile provided')
        return
    if os.path.isdir(infil):
        flist = glob.glob(os.path.join(infil, '*.tiff'))
        if os.path.isdir(outfil):
            for datapath in flist:
                root, fname = os.path.split(datapath)
                outputfile = os.path.join(outfil, fname)
                if os.path.exists(outputfile):
                    outputfile = os.path.splitext(outputfile)[0] + '_{}.tiff'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
                vyp = VyperRaster(datapath, outputfile)
                vyp.transform_raster()
                vyp.close()
        elif os.path.isfile(outfil):
            for datapath in flist:
                if os.path.exists(outfil):
                    outfil = os.path.splitext(outfil)[0] + '_{}.tiff'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
                vyp = VyperRaster(datapath, outfil)
                vyp.transform_raster()
                vyp.close()
        else:
            err = True
    elif os.path.isfile(infil):
        if os.path.isdir(outfil):
            root, fname = os.path.split(infil)
            outputfile = os.path.join(outfil, fname)
            if os.path.exists(outputfile):
                outputfile = os.path.splitext(outputfile)[0] + '_{}.tiff'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        elif os.path.isfile(outfil):
            if os.path.exists(outfil):
                outfil = os.path.splitext(outfil)[0] + '_{}.tiff'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        else:
            err = True
        vyp = VyperRaster(infil, outfil)
        vyp.transform_raster()
        vyp.close()
    else:
        err = True

    if err:
        raise ValueError('Invalid arguments, ensure both are valid file or directory paths: input_data {}, output_data {}'.format(infil, outfil))

if __name__ == '__main__':
    main()