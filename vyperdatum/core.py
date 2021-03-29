import os, sys, glob, configparser
import numpy as np
from pyproj import Transformer, datadir, CRS
from osgeo import gdal, ogr
from typing import Any, Union

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
        self.transformed_from = None

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

    def _transform_to_nad83(self, source_epsg: int, x: np.array, y: np.array, z: np.array = None):
        """
        
        Parameters
        ----------
        source_epsg
        x
        y
        z

        Returns
        -------

        """
        in_crs = CRS.from_epsg(source_epsg)
        out_crs = CRS.from_epsg(6319)
        # Transformer.transform input order is based on the CRS, see CRS.geodetic_crs.axis_info
        # - lon, lat - this appears to be valid when using CRS from proj4 string
        # - lat, lon - this appears to be valid when using CRS from epsg
        # use the always_xy option to force the transform to expect lon/lat order
        transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)

        if z is None:
            z = np.zeros_like(x)
        x, y, z = transformer.transform(x, y, z)
        return x, y, z

    def set_input_datum(self, input_datum: Union[str, int]):
        if isinstance(input_datum, int):
            self.transformed_from = input_datum
            input_datum = 'NAD83'
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
        return result

    def _get_output_uncertainty(self, region: str):
        """
        Get the output uncertainty for each point by reading the vdatum_sigma.inf file and combining the uncertainties
        that apply for this region.

        Currently we use the output datum pipeline as the source of uncertainty.  Might
        be better to use the transformation pipeline instead.  The way it currently works, if your output datum is NAD83,
        there would be no pipeline (as nad83 is the pivot datum) and so you would have 0 uncertainty, even if you did transform
        from MLLW to NAD83.

        Parameters
        ----------
        region
            region name as string

        Returns
        -------
        float
            uncertainty associated with each transformed point
        """

        if not self.out_crs.pipeline_string:  # if nad83 is the output datum, no transformation is done
            return 0
        final_uncertainty = 0
        layer_names = ['lmsl', 'mhhw', 'mhw', 'mtl', 'dtl', 'mlw', 'mllw']
        for lyr in layer_names:
            if self.out_crs.pipeline_string.find(lyr) != -1:
                final_uncertainty += self.vdatum.uncertainties[region][lyr]

        if self.out_crs.pipeline_string.find('geoid12b') != -1:
            final_uncertainty += self.vdatum.uncertainties['geoid12b']
        elif self.out_crs.pipeline_string.find('xgeoid18b') != -1:
            final_uncertainty += self.vdatum.uncertainties['xgeoid18b']
        else:
            raise ValueError('Unable to find either geoid12b or xgeoid18b in the output datum pipeline, which geoid is used?')
        return final_uncertainty

    def transform_dataset(self, x: np.array, y: np.array, z: np.array = None, include_vdatum_uncertainty: bool = True):
        if self.regions:
            if self.transformed_from:
                x, y, z = self._transform_to_nad83(self.transformed_from, x, y, z)
            ans_x = np.full_like(x, np.nan)
            ans_y = np.full_like(y, np.nan)
            if z is None:
                z = np.zeros(len(x))
            ans_z = np.full_like(z, np.nan)
            if include_vdatum_uncertainty:
                ans_unc = np.full_like(z, np.nan)
            else:
                ans_unc = None
            for region in self.regions:
                # get the pipeline
                pipeline = get_transformation_pipeline(self.in_crs, self.out_crs, region)
                tmp_x, tmp_y, tmp_z = self._run_pipeline(x, y, pipeline, z=z)

                # areas outside the coverage of the vert shift are inf
                valid_index = ~np.isinf(tmp_z)
                ans_x[valid_index] = tmp_x[valid_index]
                ans_y[valid_index] = tmp_y[valid_index]
                ans_z[valid_index] = tmp_z[valid_index]
                if include_vdatum_uncertainty:
                    ans_unc[valid_index] = self._get_output_uncertainty(region)
            return ans_x, ans_y, np.round(ans_z, 3), ans_unc
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
        self.uncertainties = {}  # dict of file names to uncertainties for each grid
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
        self.grid_files = get_gtx_grid_list(vdatum_path)
        self.polygon_files = get_vdatum_region_polygons(vdatum_path)
        self.uncertainties = get_vdatum_uncertainties(vdatum_path)

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


def get_vdatum_uncertainties(vdatum_directory: str):
    """"
    Parse the sigma file to build a dictionary of gridname: uncertainty for each layer.

    Parameters
    ----------
    vdatum_directory
        absolute folder path to the vdatum directory

    Returns
    -------
    dict
        dictionary of {kml name: kml path, ...}
    """
    acc_file = os.path.join(vdatum_directory, 'vdatum_sigma.inf')

    # use the polygon search to get a dict of all grids quickly
    grid_dict = get_vdatum_region_polygons(vdatum_directory)
    for k in grid_dict.keys():
        grid_dict[k] = {'lmsl': 0, 'mhhw': 0, 'mhw': 0, 'mtl': 0, 'dtl': 0, 'mlw': 0, 'mllw': 0}
    # add in the geoids we care about
    grid_entries = list(grid_dict.keys())

    with open(acc_file, 'r') as afil:
        for line in afil.readlines():
            data = line.split('=')
            if len(data) == 2:  # a valid line, ex: nynjhbr.lmsl=1.4
                data_entry, val = data
                sub_data = data_entry.split('.')
                if len(sub_data) == 2:
                    prefix, suffix = sub_data  # flpensac.mhw=1.8
                # elif len(sub_data) == 3:
                #     prefix, _, suffix = sub_data  # akyakutat.lmsl.mhhw=6.6
                    if prefix == 'conus':
                        if suffix == 'navd88':
                            grid_dict['geoid12b'] = float(val) * 0.01  # answer in meters
                        elif suffix == 'xgeoid18b':
                            grid_dict['xgeoid18b'] = float(val) * 0.01
                    else:
                        match = np.where(np.array([entry.lower().find(prefix) for entry in grid_entries]) == 0)
                        if match[0].size:
                            if len(match[0]) > 1:
                                raise ValueError(f'Found multiple matches in vdatum_sigma file for entry {data_entry}')
                            elif match:
                                grid_key = grid_entries[match[0][0]]
                                val = val.lstrip().rstrip()
                                if val == 'n/a':
                                    val = 0
                                grid_dict[grid_key][suffix] = float(val) * 0.01
                            else:
                                print(f'No match for vdatum_sigma entry {data_entry}!')
    return grid_dict
