"""
vyperdatum_core.py

grice 2021-02-25

The core object and supporting methods for transforming a series of points
from one vertical datum to another using proj pipelines.
"""

import os, sys, glob, logging, configparser
import numpy as np
import pyproj
from pyproj import Transformer
from pyproj.crs import CompoundCRS
import gdal
from gdal import ogr
gdal.UseExceptions()

from pipelines import get_regional_pipeline


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
    def __init__(self, output_datum: str) -> bool:
        self.vdatum = VdatumData()
        # test of output_datum is valid
        self.output_datum = output_datum
        # build output WKT string
        
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
        self.regions = intersecting_regions
        
    
    def run_pipeline(self, xx, yy, pipeline):
        assert len(xx) == len(yy)
        zz = np.zeros(len(xx))
        # get the transform at the sparse points
        transformer = Transformer.from_pipeline(pipeline)
        result = transformer.transform(xx=xx, yy=yy, zz=zz)
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
    
    
    def tranform_dataset(self, xy_dataset: np.ndarray, region: str):
        # get out vertical and horizontal datums
        input_datum = ''
        # get x and y as an array
        x = []
        y = []
        z = np.full(len(y), np.nan)
        r = np.full(len(y), 'NoRegion')
        # get the pipeline
        pipeline = get_regional_pipeline(input_datum, self.output_datum, region)
        sep = self.run_pipeline(x, y, pipeline)
        # find the non-nodata values and move into z
        # populate these same indicies in the r array with the same values
        return z, r

class VdatumData:
    """
    Gets and maintains VDatum information for use with Vyperdatum.
    
    The VDatum path location is stored in a config file which is in the user's 
    directory.
    """

    def __init__(self):
        self._get_stored_vdatum_config()
        
    def _get_stored_vdatum_config(self):
        vyperdatum_folder = os.path.join(os.path.expanduser('~'), 'vyperdatum')
        vdatum_path_file = os.path.join(vyperdatum_folder, 'vdatum.config')
        self.vdatum_path_file = vdatum_path_file
        # get the config
        if os.path.exists(vdatum_path_file):
            self._config = self._read_from_config_file(vdatum_path_file)
        else:
            default_vdatum_path = os.path.join(os.path.splitdrive(sys.executable)[0],'/VDatum')
            self._config = self._create_new_config_file(vdatum_path_file, {'vdatum_path': default_vdatum_path})
        self.vdatum_path = self._config['vdatum_path']
        
            
    def _read_from_config_file(self, filepath: str):
        """
        Read from the generated configparser file path, set the object vdatum 
        settings.
    
        Parameters
        ----------
        filepath
            absolute filepath to the configparser object
    
        Returns
        -------
        None
        """
    
        settings = {}
        config_file = configparser.ConfigParser()
        config_file.read(filepath)
        sections = config_file.sections()
        for section in sections:
            config_file_section = config_file[section]
            for key in config_file_section:
                settings[key] = config_file_section[key]
        return settings
    
    
    def _create_new_config_file(self, filepath: str, new_settings: dict) -> dict:
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
        config_folder, config_file = os.path.split(filepath)
        if not os.path.exists(config_folder):
            os.mkdir(config_folder)
        config = configparser.ConfigParser()
        config['Default'] = new_settings
        with open(filepath, 'w') as configfile:
            config.write(configfile)
        return new_settings    
    
    def update_vdatum_path(self, vdatum_path: str):
        """
        Set a new path to VDatum.

        Parameters
        ----------
        vdatum_path : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._config = self._create_new_config_file(self.vdatum_path_file, {'vdatum_path': vdatum_path})
        self.vdatum_path = self._config['vdatum_path']
        self._set_vdatum_directory()

    def _set_vdatum_directory(self, vdatum_path):
        """
        Called when self.settings['vdatum_directory'] is updated.  We find all the grids and polygons in the vdatum
        directory and save the dicts to the attributes in this class.
        """
        if not os.path.exists(self.vdatum_path):
            raise ValueError('VDatum is not found.')
        # special case for vdatum directory, we want to give pyproj the new path if it isn't there already
        orig_proj_paths = pyproj.datadir.get_data_dir()
        if vdatum_path not in orig_proj_paths:
            pyproj.datadir.append_data_dir(vdatum_path)
    
        # also want to populate grids and polygons with what we find
        newgrids = self._get_gtx_grid_list(vdatum_path)
        for gname, gpath in newgrids.items():
            self.grid_files[gname] = gpath
        newpolys = self._get_vdatum_region_polygons(vdatum_path)
        for pname, ppath in newpolys.items():
            self.polygon_files[pname] = ppath
    
    
    def _get_gtx_grid_list(self, vdatum_directory: str):
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
    
    
    def _get_vdatum_region_polygons(self, vdatum_directory: str):
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
