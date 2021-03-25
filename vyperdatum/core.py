"""
vyperdatum_core.py

grice 2021-02-25

The core object and supporting methods for transforming a series of points
from one vertical datum to another using proj pipelines.
"""

import os, sys, glob, logging, configparser
import numpy as np
from pyproj import Transformer, datadir
from pyproj.crs import CompoundCRS
import gdal
from gdal import ogr
from typing import Any

gdal.UseExceptions()

from vyperdatum.pipeline import get_regional_pipeline


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
    def __init__(self, output_datum: str, vdatum_directory: str = None) -> bool:
        self.vdatum = VdatumData(vdatum_directory=vdatum_directory)
        # build output WKT string
        self.output_datum = output_datum

        self.regions = []
        
    def set_region_by_bounds(self, x_min, y_min, x_max, y_max):
        """
        Set the regions that intersect with the provided bounds and return.

        Parameters
        ----------
        x_min : TYPE
            DESCRIPTION.
        y_min : TYPE
            DESCRIPTION.
        x_max : TYPE
            DESCRIPTION.
        y_max : TYPE
            DESCRIPTION.

        Returns
        -------
        None

        """
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
        dataGeometry = ogr.Geometry(ogr.wkbPolygon)
        dataGeometry.AddGeometry(ring)
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
                        if dataGeometry.Intersect(valid_vdatum_poly):
                            intersecting_regions.append(region)
            vector = None
        self.regions = intersecting_regions

    def run_pipeline(self, x, y, pipeline, z=None):
        if not z:
            z = np.zeros(len(x))
        assert len(x) == len(y) and len(y) == len(z)

        # get the transform at the sparse points
        transformer = Transformer.from_pipeline(pipeline)
        result = transformer.transform(xx=x, yy=y, zz=z)
        message = f'Applying pipeline: {transformer}'
        print(message)
        return result

    def check_datums(self, incrs):
        # this section is just copy and pasted junk
        req_hcrs_epsg = 26919
        req_vcrs_epsg = 'mllw'

        # parse the provided CRS
        cmpd_incrs = CompoundCRS.from_wkt(incrs.to_wkt())
        if len(cmpd_incrs.sub_crs_list) == 2:
            inhcrs, invcrs = cmpd_incrs.sub_crs_list
            assert inhcrs.to_epsg() == req_hcrs_epsg
            assert invcrs.to_epsg() == req_vcrs_epsg
        elif not cmpd_incrs.is_vertical:
            assert incrs.to_epsg() == req_hcrs_epsg

    def build_crs(self):
        # this section is just copy and pasted junk
        req_hcrs_epsg = 26919
        out_vcrs_epsg = 5703
        comp_crs = CompoundCRS(name="NAD83 UTM19 + NAVD88", components=[f"EPSG:{req_hcrs_epsg}", f"EPSG:{out_vcrs_epsg}"])
        return comp_crs

    def tranform_dataset(self, x: np.array, y: np.array, input_datum, region: str, z: np.array = None):
        # get the pipeline
        pipeline = get_regional_pipeline(input_datum, self.output_datum, region)
        x, y, z = self.run_pipeline(x, y, pipeline, z=z)
        # areas outside the coverage of the vert shift are inf, set them to NaN
        z[np.isinf(z)] = np.nan
        return z


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
