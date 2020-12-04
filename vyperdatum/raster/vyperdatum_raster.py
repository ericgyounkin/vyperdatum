# run in pydro38_test env
import os, sys, glob, logging, configparser
from datetime import datetime
import numpy as np
from scipy.interpolate import griddata
import pyproj
from pyproj import Transformer
from pyproj.crs import CompoundCRS
import gdal
from gdal import ogr
import rasterio
gdal.UseExceptions()

LOGGER = logging.getLogger('vyperdatum')

def load_config(config_filename = 'vyperdatum.config'):
    """
    Load the provided configuration file.
    """
    if not os.path.isfile(config_filename):
        raise ValueError(f'file not found: {config_filename}')
    config = {}
    config_file = configparser.ConfigParser()
    config_file.read(config_filename)
    sections = config_file.sections()
    for section in sections:
        config_file_section = config_file[section]
        for key in config_file_section:
            config[key] = config_file_section[key]
    
    if 'loggername' in config:
        global LOGGER
        LOGGER = logging.getLogger(config['loggername'])
    if 'inpath' in config:
        global inpath
        tmp = config['inpath']
        inpath = fr'{tmp}'
    if 'outpath' in config:
        global outpath
        tmp = config['outpath']
        outpath = fr'{tmp}'
    if 'vdatum_directory' in config:
        global VDATUM_DIRECTORY
        tmp = config['vdatum_directory']
        VDATUM_DIRECTORY = fr'{tmp}'
    else:
        raise ValueError('No VDatum path defined.')

def check_gdal_version():
    """
    Check the version of gdal imported to ensure it meets the requirements
    of this script

    Returns
    -------
    None.

    """
    version = gdal.VersionInfo()
    major = int(version[0])
    minor = int(version[1:3])
    bug = int(version[3:5])
    if major == 3 and minor >= 1:
        pass
    else:
        msg = f'The version of GDAL must be >= 3.1.  Version found: {version}'
        LOGGER.error(msg)
        raise ValueError(msg)


def update_vdatum_data_directory():
    global VDATUM_DIRECTORY
    orig_proj_paths = pyproj.datadir.get_data_dir()
    if VDATUM_DIRECTORY not in orig_proj_paths:
        pyproj.datadir.append_data_dir(VDATUM_DIRECTORY)


def get_gtx_grid_list():
    """
    Built a dicionary of all available VDatum grids.
    """
    global VDATUM_DIRECTORY
    search_path = os.path.join(VDATUM_DIRECTORY, '*/*.gtx')
    gtx_list = glob.glob(search_path)
    if len(gtx_list) == 0:
        raise ValueError(f'No GTX files found in the provided VDatum directory: {VDATUM_DIRECTORY}')
    grids = {}
    for gtx in gtx_list:
        gtx_path, gtx_file = os.path.split(gtx)
        gtx_path, gtx_folder = os.path.split(gtx_path)
        gtx_name = '/'.join([gtx_folder, gtx_file])
        gtx_subpath = os.path.join(gtx_folder, gtx_file)
        grids[gtx_name] = gtx_subpath
    return grids


def get_vdatum_region_polygons():
    """
    Get a list of all the kml files that are in the VDatum directory.

    Returns
    -------
    List

    """
    global VDATUM_DIRECTORY
    search_path = os.path.join(VDATUM_DIRECTORY, '*/*.kml')
    kml_list = glob.glob(search_path)
    if len(kml_list) == 0:
        raise ValueError(f'No kml files found in the provided VDatum directory: {VDATUM_DIRECTORY}')
    geom = {}
    for kml in kml_list:
        kml_path, kml_file = os.path.split(kml)
        root_dir, kml_name = os.path.split(kml_path)
        geom[kml_name] = kml
    return geom


def get_interesecting_vdatum_regions(datafilepath):
    """
    Find the vdatum regions that intersect the given data.
    """
    dataset = gdal.Open(datafilepath)
    is_raster = dataset.RasterCount > 0
    if is_raster:
        # get raster bounds
        crs = pyproj.CRS.from_wkt(dataset.GetProjectionRef())
        transform = dataset.GetGeoTransform()
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        x0 = transform[0]
        y1 = transform[3]
        x1 = x0+cols*pixelWidth
        y0 = y1-rows*pixelHeight
        if crs.is_projected:
            unproject = pyproj.Proj(proj='utm', zone = 19, ellps = 'WGS84')
            ul = unproject(x0,x1, inverse = True)
            ur = unproject(x1, y1, inverse = True)
            lr = unproject(x1, y0, inverse = True)
            ll = unproject(x0, y0, inverse = True)
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
    # get all the regions represented by geometry files
    geom_list = get_vdatum_region_polygons()
    # see if the regions intersect with the provided geometries
    intersecting_regions = []
    for region in geom_list:
        vector = ogr.Open(geom_list[region])
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
    return intersecting_regions
    

def run_pipeline(xx, yy, zz, incrs, region_name):
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
    out_cmpd_crs = CompoundCRS(name="NAD83 UTM19 + NAVD88", components=[f"EPSG:{req_hcrs_epsg}", f"EPSG:{out_vcrs_epsg}"])
    # get the transform at the sparse points
    grids = get_gtx_grid_list()
    transformer = Transformer.from_pipeline(f'proj=pipeline \
                                              step inv proj=utm zone=19 \
                                              step inv proj=vgridshift grids={grids[f"{region_name}/mllw.gtx"]} \
                                              step proj=vgridshift grids={grids[f"{region_name}/tss.gtx"]} \
                                              step proj=utm zone=19')

    result = transformer.transform(xx=xx, yy=yy, zz=zz)
    LOGGER.debug(f'Applying pipeline: {transformer}')
    
    return result, out_cmpd_crs


def get_datum_sep(raster_name, transform_sampling_distance, region_list):
    """
    Use the provided raster and pipeline to get the separation over the raster area.

    Returns
    -------
    sep

    """
    with rasterio.open(raster_name) as raster:
        elev = raster.read(1)
        transform = raster.transform
        crs = raster.crs
        nodata = raster.nodata
        profile = raster.profile
    # get sparse raster x and y positions
    sy, sx = profile['height'], profile['width']
    resy, resx = transform[4], transform[0]
    y0, x0 = transform[5], transform[2]
    y1 = y0 + sy * resy 
    x1 = x0 + sx * resx
    nx = np.round(np.abs((x1 - x0) / transform_sampling_distance)).astype(int)
    ny = np.round(np.abs((y1 - y0) / transform_sampling_distance)).astype(int)
    x_sampled = np.linspace(x0, x1, nx)
    y_sampled = np.linspace(y0, y1, ny)
    yy, xx = np.meshgrid(y_sampled, x_sampled, indexing = 'ij')
    zz = np.zeros(yy.shape)
    y,x = np.mgrid[y0:y1:resy, x0:x1:resx]
    sep = np.full(y.shape, np.nan)
    
    for region in region_list:
        start = datetime.now()
        try:
            result, new_crs = run_pipeline(xx.flatten(), yy.flatten(), zz.flatten(), crs, region)
        except pyproj.ProjError as e:
            print_paths = '\n'.join(pyproj.datadir.get_data_dir().split(';'))
            LOGGER.error(f'Proj pipeline failed. pyproj paths: \n{print_paths}')
            raise e
        dt = datetime.now() - start
        LOGGER.debug(f'Transforming {len(yy.flatten())} points took {dt} seconds for {region}')
        vals = result[2].flatten()
        valid_idx = np.squeeze(np.argwhere(~np.isinf(vals)))
        
        if len(valid_idx) == 0:
            LOGGER.debug('No valid points found from gridding in {region}. Putting all points through proj pipeline directly.')
        else:
            LOGGER.debug('interpolating to original grid for {region}')
            start = datetime.now()
            points = np.array([result[1][valid_idx],result[0][valid_idx]]).T
            valid_vals = vals[valid_idx]
            try:
                region_sep = griddata(points, valid_vals, (y,x))
                idx = ~np.isnan(region_sep)
                sep[idx] = region_sep[idx]
            except Exception as error:
                msg = f'{error.__class__.__name__}: {error}\n\n'
                LOGGER.error(msg)
                return None, None
            dt = datetime.now() - start
            LOGGER.debug(f'Interpolating {len(y.flatten())} points took {dt} seconds for {region}')
        missing_idx = np.where(np.isnan(sep) & (elev != nodata))
        num_nan = len(missing_idx[0])
        if num_nan > 0:
            missing, new_crs = run_pipeline(x[missing_idx], y[missing_idx], np.zeros(num_nan), crs, region)
            missing_sep = missing[2]
            still_inf_idx = np.where(np.isinf(missing_sep))
            missing_sep[still_inf_idx] = np.nan
            sep[missing_idx] = missing_sep
    return sep, new_crs

def apply_sep(infile, outfile, sep, new_crs, pass_untransformed = True):
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
    with rasterio.open(infile) as raster:
            data = raster.read()
            transform = raster.transform
            nodata = raster.nodata
            profile = raster.profile
            band_names = raster.descriptions
            
    missing_idx = np.where(np.isnan(sep) & (data[0] != nodata))
    num_nan = len(missing_idx[0])
    LOGGER.debug(f'Found {num_nan} values that are outside the separation model')
    if pass_untransformed:
        z_values = data[0, missing_idx[0], missing_idx[1]]
        # make the new uncertainty
        u_values = 3 - 0.06 * z_values
        u_values[np.where(z_values > 0)] = 3.0
        uncert = data[1]
        uncert[missing_idx] = u_values
        if num_nan > 0:
            LOGGER.debug(f'Inserting {num_nan} values that were not transformed into new file, but applying CATZOC D vertical uncertainty.')
    elev = (data[0] - sep)
    elev[np.where(np.isnan(elev))] = nodata
    if pass_untransformed:
        elev[missing_idx] = z_values
        data[0] = elev
        data[1] = uncert
    else:
        data[:, missing_idx[0], missing_idx[1]] = nodata
    
    data = np.round(data[:2].astype(np.float), decimals = 2).astype(np.float32)
    band_names = band_names[:2]
    
    write_gdal_geotiff(outfile, data, new_crs, transform, nodata, band_names)
    
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
    
def transform_raster(infilepathname, outfilepathname):
    """
    

    Returns
    -------
    None.

    """
    LOGGER.debug(f'Begin work on {os.path.basename(infilepathname)}')
    region_list = get_interesecting_vdatum_regions(infilepathname)
    if len(region_list) > 0:
        sep,crs = get_datum_sep(infilepathname, 100, region_list)
        apply_sep(infilepathname, outfilepathname, sep, crs)
    else:
        LOGGER.debug(f'no region found for {os.path.basename(datapath)}')
        
def set_logger(outpath):
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('[%(asctime)s] %(name)-9s %(levelname)-8s: %(message)s')

    log_name = f'_to_NAVD88_{datetime.now():%Y%m%d_%H%M%S}.log'
    log_filename = os.path.join(outpath, log_name)
    log_file = logging.FileHandler(log_filename)
    log_file.setFormatter(log_formatter)
    log_file.setLevel(logging.DEBUG)
    LOGGER.addHandler(log_file)
    
    output_console = logging.StreamHandler(sys.stdout)
    output_console.setFormatter(log_formatter)
    output_console.setLevel(logging.DEBUG)
    LOGGER.addHandler(output_console)
    
def clear_logger():
    for handler in LOGGER.handlers[:]:
            handler.close()
            LOGGER.removeHandler(handler)
            
if __name__ == '__main__':
    if len(sys.argv) == 1:
        load_config()
    else:
        config_filename = sys.argv[1]
        load_config(config_filename)
    set_logger(outpath)
    update_vdatum_data_directory()
    check_gdal_version()
    flist = glob.glob(os.path.join(inpath, '*.tiff'))
    for datapath in flist:
        outfilepath = os.path.join(outpath, os.path.basename(datapath))
        transform_raster(datapath, outfilepath)
    clear_logger()
