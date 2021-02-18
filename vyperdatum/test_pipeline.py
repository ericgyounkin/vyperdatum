'''
    this file is for testing/debugging only
'''
import os
from pyproj import Transformer

from raster.vyperdatum_raster import update_vdatum_data_directory
from pipeline import get_pipeline


def test_get_pipeline(from_datum, to_datum, region_name, zone, xx, yy, zz):

    pipeline = get_pipeline(from_datum, to_datum, region_name, zone)
    transformer = Transformer.from_pipeline(pipeline)
    result = transformer.transform(xx=xx, yy=yy, zz=zz)
    print(transformer)
    print('input:', (xx, yy, zz), from_datum, to_datum)
    print('output:', result, '\n')


def test_pipeline():
    region_name = 'MENHMAgome13_8301'
    zone = 19
    #xx = [-70.7]
    #yy = [43]
    xx = [361434.3478397919]
    yy = [4762216.968949459]
    zz = [0]

    test_get_pipeline('MLLW', 'NAVD88', region_name, zone, xx, yy, zz)
    test_get_pipeline('NAVD88', 'MLLW', region_name, zone, xx, yy, zz)
    test_get_pipeline('MLLW', 'NAD83', region_name, zone, xx, yy, zz)
    test_get_pipeline('NAD83', 'MLLW', region_name, zone, xx, yy, zz)
    test_get_pipeline('NAVD88', 'NAD83', region_name, zone, xx, yy, zz)
    test_get_pipeline('NAD83', 'NAVD88', region_name, zone, xx, yy, zz)


if __name__ == '__main__':
    update_vdatum_data_directory()
    test_pipeline()
