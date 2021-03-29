import os, sys, glob, configparser
import numpy as np
from pyproj import Transformer, datadir
from osgeo import gdal, ogr
from typing import Any

from vyperdatum.vypercrs import VerticalPipelineCRS, get_transformation_pipeline
from vyperdatum.pipeline import get_regional_pipeline

gdal.UseExceptions()


class VyperCore:
    """
    The core object for conducting transformations.
    
    provide output_datums -> pipeline
    
    TODO:
        Confirm horizontal datum is either as needed by pipeline or add in horiontal
        transformation.
        
    TODO: We need to pull in the uncertainty associated with a datum transformation
        and return that value by point.
    """
    def __init__(self, vdatum_directory: str = None):
        # if vdatum_directory is provided initialize VdatumData with that path
        self.vdatum = VdatumData(vdatum_directory=vdatum_directory)

        self.in_crs = None
        self.out_crs = None
        self.regions = []

    def set_region_by_bounds(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """
        Set the regions that intersect with the provided bounds and return a list of region names that overlap

        Parameters
        ----------
        x_min
            the minimum longitude of the area of interest
        y_min
            the minimum latitude of the area of interest
        x_max
            the maximum longitude of the area of interest
        y_max
            the maximum latitude of the area of interest
        """

        assert x_min < x_max
        assert y_min < y_max

        # build corners from the provided bounds
        ul = (x_min, y_max)
        ur = (x_max, y_max)
        lr = (x_max, y_min)
        ll = (x_min, y_min)

        # build polygon from corners
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(ul[0], ul[1])
        ring.AddPoint(ur[0], ur[1])
        ring.AddPoint(lr[0], lr[1])
        ring.AddPoint(ll[0], ll[1])
        ring.AddPoint(ul[0], ul[1])
        data_geometry = ogr.Geometry(ogr.wkbPolygon)
        data_geometry.AddGeometry(ring)

        # see if the regions intersect with the provided geometries
        intersecting_regions = []
        for region in self.vdatum.polygon_files:
            vector = ogr.Open(self.vdatum.polygon_files[region])
            layer_count = vector.GetLayerCount()
            for m in range(layer_count):
                layer = vector.GetLayerByIndex(m)
                feature_count = layer.GetFeatureCount()
                for n in range(feature_count):
                    feature = layer.GetFeature(n)
                    feature_name = feature.GetField(0)
                    if feature_name[:15] == 'valid-transform':
                        valid_vdatum_poly = feature.GetGeometryRef()
                        if data_geometry.Intersect(valid_vdatum_poly):
                            intersecting_regions.append(region)
                    feature = None
                layer = None
            vector = None
        self.regions = intersecting_regions

    def is_alaska(self):
        if self.regions:
            is_alaska = all([t.find('AK') != -1 for t in self.regions])
            is_not_alaska = all([t.find('AK') == -1 for t in self.regions])
            if not is_alaska and not is_not_alaska:
                raise NotImplementedError('Regions in and not in alaska specified, not currently supported')
            if is_alaska:
                return True
            else:
                return False
        else:
            raise ValueError('No regions specified, unable to determine is_alaska')

    def _build_datum_from_string(self, datum: str):
        if self.regions:
            new_crs = VerticalPipelineCRS(datum)
            base_region = self.regions[0]
            base_pipeline = get_regional_pipeline('nad83', datum, base_region, is_alaska=self.is_alaska())
            for r in self.regions:
                new_crs.add_pipeline(base_pipeline, r)
            return new_crs
        else:
            raise ValueError('No regions specified, unable to construct new vyperdatum crs')

    def set_input_datum(self, input_datum: str):
        try:
            incrs = VerticalPipelineCRS()
            incrs.from_wkt(input_datum)
        except ValueError:
            incrs = self._build_datum_from_string(input_datum)
        self.in_crs = incrs

    def set_output_datum(self, output_datum: str):
        try:
            outcrs = VerticalPipelineCRS()
            outcrs.from_wkt(output_datum)
        except ValueError:
            outcrs = self._build_datum_from_string(output_datum)
        self.out_crs = outcrs

    def _run_pipeline(self, x, y, pipeline, z=None):
        if z is None:
            z = np.zeros(len(x))
        assert len(x) == len(y) and len(y) == len(z)

        # get the transform at the sparse points
        transformer = Transformer.from_pipeline(pipeline)
        result = transformer.transform(xx=x, yy=y, zz=z)
        message = f'Applying pipeline: {transformer}'
        print(message)
        return result

    def tranform_dataset(self, x: np.array, y: np.array, z: np.array = None):
        if self.regions:
            ans_x = np.full_like(x, np.nan)
            ans_y = np.full_like(y, np.nan)
            if z is None:
                z = np.zeros(len(x))
            ans_z = np.full_like(z, np.nan)
            for region in self.regions:
                # get the pipeline
                pipeline = get_transformation_pipeline(self.in_crs, self.out_crs, region)
                tmp_x, tmp_y, tmp_z = self._run_pipeline(x, y, pipeline, z=z)

                # areas outside the coverage of the vert shift are inf
                valid_index = ~np.isinf(tmp_z)
                ans_x[valid_index] = tmp_x[valid_index]
                ans_y[valid_index] = tmp_y[valid_index]
                ans_z[valid_index] = tmp_z[valid_index]
            return ans_x, ans_y, np.round(ans_z, 3)
        else:
            raise ValueError('No regions specified, unable to transform points')


class VdatumData:
    """
    Gets and maintains VDatum information for use with Vyperdatum.
    
    The VDatum path location is stored in a config file which is in the user's directory.  Use configparser to sync
    self._config and the ini file.

    Optionally, user may provide a vdatum directory here on initialization to set the vdatum path the first time
    """

    def __init__(self, vdatum_directory: str = None):
        self.grid_files = {}  # dict of file names to file paths for the gtx files
        self.polygon_files = {}  # dict of file names to file paths for the kml files
        self.vdatum_path = ''  # path to the parent vdatum folder

        self._config = {}  # dict of all the settings
        self.config_path_file = ''  # path to the config file that maintains the settings between runs

        self._get_stored_vdatum_config()
        if vdatum_directory:  # overwrite the loaded path if you want to change it on initialization
            self.set_vdatum_directory(vdatum_directory)
        else:
            self.set_vdatum_directory(self.vdatum_path)

    def set_config(self, ky: str, value: Any):
        """
        Setter for the _config attribute.  Use this instead of setting _config directly, will set both the _config
        key/value and the configparser ini file.

        Parameters
        ----------
        ky
            key to set in the dict
        value
            value to set in the dict
        """

        config = configparser.ConfigParser()
        config.read(self.config_path_file)
        for k, v in self._config.items():
            config['Default'][k] = v

        self._config[ky] = value  # set the class attribute
        config['Default'][ky] = value  # set the ini matching attribute
        with open(self.config_path_file, 'w') as configfile:
            config.write(configfile)

        if ky == 'vdatum_path':
            self.vdatum_path = value

    def _get_stored_vdatum_config(self):
        """
        Runs on initialization, will read from the ini file and set the vdatum path, config attribute
        """
        vyperdatum_folder = os.path.join(os.path.expanduser('~'), 'vyperdatum')
        self.config_path_file = os.path.join(vyperdatum_folder, 'vdatum.config')
        # get the config
        if os.path.exists(self.config_path_file):
            self._config = self._read_from_config_file()
        else:
            default_vdatum_path = os.path.join(os.path.splitdrive(sys.executable)[0], '/VDatum')
            self._config = self._create_new_config_file({'vdatum_path': default_vdatum_path})
        self.vdatum_path = self._config['vdatum_path']
            
    def _read_from_config_file(self):
        """
        Read from the generated configparser file path, set the object vdatum 
        settings.
    
        Returns
        -------
        dict
            dictionary of settings
        """
    
        settings = {}
        config_file = configparser.ConfigParser()
        config_file.read(self.config_path_file)
        sections = config_file.sections()
        for section in sections:
            config_file_section = config_file[section]
            for key in config_file_section:
                settings[key] = config_file_section[key]
        return settings

    def _create_new_config_file(self, default_settings: dict) -> dict:
        """
        Create a new configparser file, return the settings and the configparser object
    
        Parameters
        ----------
        default_settings
            default settings we want to write to the configparser file
    
        Returns
        -------
        configparser.ConfigParser
            configparser object used to read the file
        dict
            settings within the file
        """
        config_folder, config_file = os.path.split(self.config_path_file)
        if not os.path.exists(config_folder):
            os.mkdir(config_folder)
        config = configparser.ConfigParser()
        config['Default'] = default_settings
        with open(self.config_path_file, 'w') as configfile:
            config.write(configfile)
        return default_settings

    def set_vdatum_directory(self, vdatum_path: str):
        """
        Called when self.settings['vdatum_directory'] is updated.  We find all the grids and polygons in the vdatum
        directory and save the dicts to the attributes in this class.
        """
        self.set_config('vdatum_path', vdatum_path)
        if not os.path.exists(self.vdatum_path):
            raise ValueError(f'VDatum is not found at the provided path: {self.vdatum_path}')

        # special case for vdatum directory, we want to give pyproj the new path if it isn't there already
        orig_proj_paths = datadir.get_data_dir()
        if vdatum_path not in orig_proj_paths:
            datadir.append_data_dir(vdatum_path)
    
        # also want to populate grids and polygons with what we find
        newgrids = get_gtx_grid_list(vdatum_path)
        for gname, gpath in newgrids.items():
            self.grid_files[gname] = gpath
        newpolys = get_vdatum_region_polygons(vdatum_path)
        for pname, ppath in newpolys.items():
            self.polygon_files[pname] = ppath
        self.vdatum_path = self._config['vdatum_path']


def get_gtx_grid_list(vdatum_directory: str):
    """
    Search the vdatum directory to find all gtx files

    Parameters
    ----------
    vdatum_directory
        absolute folder path to the vdatum directory

    Returns
    -------
    dict
        dictionary of {grid name: grid path, ...}
    """

    search_path = os.path.join(vdatum_directory, '*/*.gtx')
    gtx_list = glob.glob(search_path)
    if len(gtx_list) == 0:
        errmsg = f'No GTX files found in the provided VDatum directory: {vdatum_directory}'
        print(errmsg)
    grids = {}
    for gtx in gtx_list:
        gtx_path, gtx_file = os.path.split(gtx)
        gtx_path, gtx_folder = os.path.split(gtx_path)
        gtx_name = '/'.join([gtx_folder, gtx_file])
        gtx_subpath = os.path.join(gtx_folder, gtx_file)
        grids[gtx_name] = gtx_subpath
    return grids


def get_vdatum_region_polygons(vdatum_directory: str):
    """"
    Search the vdatum directory to find all kml files

    Parameters
    ----------
    vdatum_directory
        absolute folder path to the vdatum directory


    Returns
    -------
    dict
        dictionary of {kml name: kml path, ...}
    """

    search_path = os.path.join(vdatum_directory, '*/*.kml')
    kml_list = glob.glob(search_path)
    if len(kml_list) == 0:
        errmsg = f'No kml files found in the provided VDatum directory: {vdatum_directory}'
        print(errmsg)
    geom = {}
    for kml in kml_list:
        kml_path, kml_file = os.path.split(kml)
        root_dir, kml_name = os.path.split(kml_path)
        geom[kml_name] = kml
    return geom
