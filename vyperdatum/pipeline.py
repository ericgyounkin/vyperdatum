import re
from raster.vyperdatum_raster import LOGGER


UTM_zone = {'ALFLgom': 16,
            'ALmobile': 16,
            'CAmorrob': 10,
            'CAORblan': (10, 9),
            'CAsanfrb': (10, 9),
            'CAsocal': (11, 10),
            'DEdelbay': 18,
            'DEVAemb': 18,
            'FLandrew': 16,
            'FLapalach': (17, 16),
            'FLGAeastbays': 17,
            'FLGAeastshelf': 17,
            'FLjoseph': 16,
            'FLpensac': 16,
            'FLsouth': 17,
            'FLwest': 17,
            'GASCNCsab': (18, 17),
            'LAmobile': (16, 15),
            'LATXwest': 15,
            'MDVAchb': 18,
            'MENHMAgome': 19,
            'NCcoast': (18, 17),
            'NCinner': (18, 17),
            'NJcstemb': 18,
            'NJVAmab': 18,
            'NYgr': 18,
            'NYNJhbr': (19, 18),
            'ORcentr': (10, 9),
            'ORWAcolr': (10, 9),
            'PRVI': (20, 19),
            'RICTbis': (19, 18),
            'TXlaggal': (15, 14),
            'TXlagmat': 14,
            'TXshlgal': (15, 14),
            'TXshlmat': 14,
            'WAjdfuca': (10, 9),
            'WApugets': 10}


def get_pipeline(from_datum, to_datum, region_name, input_utm=True, output_utm=True):

    region = re.match(r'^(\D+).*', region_name).group(1)
    if input_utm or output_utm:
        zone = UTM_zone.get(region)
        if not isinstance(zone, int):
            msg = f'Failed to get UTM zone for region name {region_name}'
            LOGGER.error(msg)
            raise ValueError(msg)

    utm = ''
    if output_utm:
        utm = f'step proj=utm zone={zone}'
    inv_utm = ''
    if input_utm:
        inv_utm = f'step inv proj=utm zone={zone}'
    mllw = f'step proj=vgridshift grids={region_name}\\mllw.gtx'
    inv_mllw = f'step inv proj=vgridshift grids={region_name}\\mllw.gtx'
    tss = f'step proj=vgridshift grids={region_name}\\tss.gtx'
    inv_tss = f'step inv proj=vgridshift grids={region_name}\\tss.gtx'
    geoid12b = f'step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx'
    inv_geoid12b = f'step inv proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx'


    pipeline_dict = {'NAVD88_MLLW':  f'proj=pipeline {inv_utm} {mllw} {inv_tss} {utm}',

                     'MLLW_NAVD88':  f'proj=pipeline {inv_utm} {inv_mllw} {tss} {utm}',

                     'MLLW_NAD83':   f'proj=pipeline {inv_utm} {inv_mllw} {tss} {inv_geoid12b} {utm}',

                     'NAD83_MLLW':   f'proj=pipeline {inv_utm} {mllw} {inv_tss} {geoid12b} {utm}',

                     'NAD83_NAVD88': f'proj=pipeline {inv_utm} {geoid12b} {utm}',

                     'NAVD88_NAD83': f'proj=pipeline {inv_utm} {inv_geoid12b} {utm}'}


    try:
        pipeline = pipeline_dict[f'{from_datum.upper()}_{to_datum.upper()}']
    except:
        msg = f'Failed to get pipeline from {from_datum} to {to_datum}'
        LOGGER.error(msg)
        raise ValueError(msg)

    return pipeline