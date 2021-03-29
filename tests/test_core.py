from vyperdatum.core import *


def test_core_setup():
    # these tests assume you have the vdatum path setup in VyperCore
    # first time, you need to run it with the path to the vdatum folder, vc = VyperCore('path\to\vdatum')
    vc = VyperCore()
    assert os.path.exists(vc.vdatum.vdatum_path)
    assert vc.vdatum.grid_files
    assert vc.vdatum.polygon_files


def test_regions():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    assert vc.regions == ['NCcoast11_8301', 'NCinner11_8301']


def test_is_alaska():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    assert not vc.is_alaska()
    vc.set_region_by_bounds(-136.56527, 56.21873, -135.07113, 56.77662)
    assert vc.is_alaska()


def test_out_of_bounds():
    vc = VyperCore()
    vc.set_region_by_bounds(-155.29119, 57.12611, -154.56609, 57.67068)

    assert vc.regions == []
    try:
        vc.is_alaska()
    except ValueError:  # no regions, so this will fail with valueerror exception
        assert True


def test_set_input_datum():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    vc.set_input_datum('mllw')

    assert vc.in_crs.datum_name == 'mllw'
    assert vc.in_crs.pipeline_string == 'proj=pipeline step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx ' \
                                        'step proj=vgridshift grids=NCcoast11_8301\\tss.gtx ' \
                                        'step proj=vgridshift grids=NCcoast11_8301\\mllw.gtx'
    assert vc.in_crs.regions == ['NCcoast11_8301', 'NCinner11_8301']


def test_set_output_datum():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    vc.set_output_datum('geoid12b')

    assert vc.out_crs.datum_name == 'geoid12b'
    assert vc.out_crs.pipeline_string == 'proj=pipeline step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx'
    assert vc.out_crs.regions == ['NCcoast11_8301', 'NCinner11_8301']


def test_transform_dataset():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    vc.set_input_datum('nad83')
    vc.set_output_datum('mllw')
    x = np.array([-75.79180, -75.79190, -75.79200])
    y = np.array([36.01570, 36.01560, 36.01550])
    z = np.array([10.5, 11.0, 11.5])
    newx, newy, newz, _ = vc.transform_dataset(x, y, z, include_vdatum_uncertainty=False)

    assert (x == newx).all()
    assert (y == newy).all()
    assert (newz == np.array([49.490, 49.990, 50.490])).all()

    assert vc.out_crs.to_wkt() == 'VERTCRS["mllw",VDATUM["mllw"],' \
                                  'CS[vertical,1],AXIS["gravity-related height (H)",up],LENGTHUNIT["metre",1],' \
                                  'REMARK["regions=[NCcoast11_8301,NCinner11_8301],' \
                                  'pipeline=proj=pipeline step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx ' \
                                  'step proj=vgridshift grids=NCcoast11_8301\\tss.gtx ' \
                                  'step proj=vgridshift grids=NCcoast11_8301\\mllw.gtx"]]'


def test_transform_dataset_inv():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    vc.set_input_datum('mllw')
    vc.set_output_datum('nad83')
    x = np.array([-75.79180, -75.79190, -75.79200])
    y = np.array([36.01570, 36.01560, 36.01550])
    z = np.array([49.490, 49.990, 50.490])
    newx, newy, newz, _ = vc.transform_dataset(x, y, z, include_vdatum_uncertainty=False)

    assert (x == newx).all()
    assert (y == newy).all()
    assert (newz == np.array([10.5, 11.0, 11.5])).all()

    assert vc.out_crs.to_wkt() == 'VERTCRS["nad83",VDATUM["nad83"],' \
                                  'CS[vertical,1],AXIS["gravity-related height (H)",up],LENGTHUNIT["metre",1],' \
                                  'REMARK["regions=[NCcoast11_8301,NCinner11_8301],pipeline=None"]]'


def test_transform_dataset_unc():
    vc = VyperCore()
    vc.set_region_by_bounds(-75.79179, 35.80674, -75.3853, 36.01585)
    vc.set_input_datum('nad83')
    vc.set_output_datum('mllw')
    x = np.array([-75.79180, -75.79190, -75.79200])
    y = np.array([36.01570, 36.01560, 36.01550])
    z = np.array([10.5, 11.0, 11.5])
    newx, newy, newz, newunc = vc.transform_dataset(x, y, z)

    assert (x == newx).all()
    assert (y == newy).all()
    assert (newz == np.array([49.490, 49.990, 50.490])).all()
    assert (newunc == np.array([6.5, 6.5, 6.5])).all()  # ncinner.mllw=1.5, conus.navd88=5.0
