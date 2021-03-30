from vyperdatum.points import *


def test_points_setup():
    # these tests assume you have the vdatum path setup in VyperCore
    # first time, you need to run it with the path to the vdatum folder, vp = VyperPoints('path\to\vdatum')
    vp = VyperPoints()
    assert os.path.exists(vp.vdatum.vdatum_path)
    assert vp.vdatum.grid_files
    assert vp.vdatum.polygon_files


def test_transform_dataset():
    vp = VyperPoints()
    x = np.array([-75.79180, -75.79190, -75.79200])
    y = np.array([36.01570, 36.01560, 36.01550])
    z = np.array([10.5, 11.0, 11.5])
    vp.transform_points('nad83', 'mllw', x, y, z=z, include_vdatum_uncertainty=False)

    assert (x == vp.x).all()
    assert (y == vp.y).all()
    assert (vp.z == np.array([49.490, 49.990, 50.490])).all()

    assert vp.out_crs.to_wkt() == 'VERTCRS["mllw",VDATUM["mllw"],' \
                                  'CS[vertical,1],AXIS["gravity-related height (H)",up],LENGTHUNIT["metre",1],' \
                                  'REMARK["regions=[NCinner11_8301],' \
                                  'pipeline=proj=pipeline step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx ' \
                                  'step proj=vgridshift grids=NCinner11_8301\\tss.gtx ' \
                                  'step proj=vgridshift grids=NCinner11_8301\\mllw.gtx"]]'
