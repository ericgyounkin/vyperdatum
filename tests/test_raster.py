from pytest import approx

from vyperdatum.raster import *


test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tiff', 'test.tiff')


def test_core_setup():
    # these tests assume you have the vdatum path setup in VyperCore
    # first time, you need to run it with the path to the vdatum folder, vc = VyperCore('path\to\vdatum')
    vc = VyperCore()
    assert os.path.exists(vc.vdatum.vdatum_path)
    assert vc.vdatum.grid_files
    assert vc.vdatum.polygon_files


def test_find_testdata():
    assert os.path.exists(test_file)


def test_raster_initialize():
    vr_one = VyperRaster()
    vr_one.initialize(test_file)
    vr_two = VyperRaster(test_file)

    assert vr_one.input_file == test_file
    assert vr_one.geotransform == (339262.0, 4.0, 0.0, 4693254.0, 0.0, -4.0)

    assert vr_one.min_x == 339262.0
    assert vr_one.min_y == 4684786.0
    assert vr_one.max_x == 345630.0
    assert vr_one.max_y == 4693254.0

    assert vr_one.geographic_min_x == approx(-70.94997811389081, 0.0000001)
    assert vr_one.geographic_min_y == approx(42.29873069934964, 0.0000001)
    assert vr_one.geographic_max_x == approx(-70.87503006957049, 0.0000001)
    assert vr_one.geographic_max_y == approx(42.37624115875231, 0.0000001)

    assert vr_one.input_file == vr_two.input_file
    assert vr_one.geotransform == vr_two.geotransform

    assert vr_one.min_x == vr_two.min_x
    assert vr_one.min_y == vr_two.min_y
    assert vr_one.max_x == vr_two.max_x
    assert vr_one.max_y == vr_two.max_y

    assert vr_one.geographic_min_x == vr_two.geographic_min_x
    assert vr_one.geographic_min_y == vr_two.geographic_min_y
    assert vr_one.geographic_max_x == vr_two.geographic_max_x
    assert vr_one.geographic_max_y == vr_two.geographic_max_y


def test_raster_data():
    vr = VyperRaster()
    vr.initialize(test_file)
    assert vr.layernames == ['Elevation', 'Uncertainty', 'Contributor']

    elev_idx = vr._get_elevation_layer_index()
    unc_idx = vr._get_uncertainty_layer_index()
    cont_idx = vr._get_contributor_layer_index()

    assert elev_idx == 0
    assert unc_idx == 1
    assert cont_idx == 2

    elev_layer = vr.layers[elev_idx]
    unc_layer = vr.layers[unc_idx]
    cont_layer = vr.layers[cont_idx]

    assert np.isnan(vr.layers[0][0][0])
    assert vr.layers[0][100][100] == approx(-10.61, 0.001)
    assert vr.layers[0][1050][100] == approx(-21.3, 0.001)
    assert vr.layers[0][400][400] == approx(-10.560385, 0.001)

    assert np.isnan(vr.layers[1][0][0])
    assert vr.layers[1][100][100] == approx(1.21, 0.001)
    assert vr.layers[1][1050][100] == approx(1.43, 0.001)
    assert vr.layers[1][400][400] == approx(12.316812, 0.001)

    assert np.isnan(vr.layers[2][0][0])
    assert vr.layers[2][100][100] == 124.0
    assert vr.layers[2][1050][100] == 214.0
    assert vr.layers[2][400][400] == 396.0


def test_raster_set_input_datum():
    vr = VyperRaster(test_file)
    base_input_datum = vr.transformed_from  # should be 26919, as read from the file provided
    base_post_initial_transform = vr.in_crs.to_wkt()  # the in_crs should always be nad83 wkt string

    vr.set_input_datum(26919)
    assert base_input_datum == 26919
    assert base_post_initial_transform == vr.in_crs.to_wkt()


def test_raster_set_output_datum():
    vr = VyperRaster(test_file)
    assert vr.out_crs is None
    vr.set_output_datum('mllw')
    assert vr.out_crs.to_wkt() == 'VERTCRS["mllw",VDATUM["mllw"],' \
                                  'CS[vertical,1],AXIS["gravity-related height (H)",up],LENGTHUNIT["metre",1],' \
                                  'REMARK["regions=[MENHMAgome23_8301],' \
                                  'pipeline=proj=pipeline step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx ' \
                                  'step proj=vgridshift grids=MENHMAgome23_8301\\tss.gtx ' \
                                  'step proj=vgridshift grids=MENHMAgome23_8301\\mllw.gtx"]]'

