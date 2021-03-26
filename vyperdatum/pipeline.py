"""
pipeline.py

The proj pipeline string generator.

The pipelines are currently stored as a dictionary, but a lightweight database
likely makes more sense so that specific datum and epochs (and versions?) can
be queried.

TODO: We need a WKT to datum definition lookup generator, unless the WKT 
    contains the information to specify specific tranformation layers.

TODO: Perhaps we should not specify a geoid file for a datum, but,
    like the VDatum regions, find which ones intersect a specific dataset and
    specify a heirarcy for which one to use.  Currently a specific file is
    specified.

TODO: g2012bu0 for CONUS and g2012ba0 for Alaska, need to make that distinction

The basic steps outlined here are:
    1) Find the datum definitions for the input and output datums.
    2) Compare the definitions starting at the base (ellipsoid).  When the
       definitions no long agree, stop.
    3) Reverse the remaining input datum layers and prepend 'inv' to each layer.
    4) String the input and output layers together.
"""

# All datum definitions are defined relative to the same 'pivot' ellipsoid.
datum_definition = {
    'nad83'    : [],
    'geoid12b' : ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx'],
    'navd88'   : ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx'],
    'tss'      : ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx',
                  'proj=vgridshift grids={region_name}\\tss.gtx'],
    'mllw'     : ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx',
                  'proj=vgridshift grids={region_name}\\tss.gtx',
                  'proj=vgridshift grids={region_name}\\mllw.gtx'],
    'noaa chart datum': ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx',
                         'proj=vgridshift grids={region_name}\\tss.gtx',
                         'proj=vgridshift grids={region_name}\\mllw.gtx'],
    'mhw'     : ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx',
                 'proj=vgridshift grids={region_name}\\tss.gtx',
                 'proj=vgridshift grids={region_name}\\mhw.gtx']
    'noaa chart height': ['proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx',
                          'proj=vgridshift grids={region_name}\\tss.gtx',
                          'proj=vgridshift grids={region_name}\\mhw.gtx']
    }


def get_regional_pipeline(from_datum: str, to_datum: str, region_name: str) -> [str]:
    """
    Return a string describing the pipeline to use to convert between the
    provided datums.

    Parameters
    ----------
    from_datum : str
        A string corresponding to one of the stored datums.
    to_datum : str
        A string corresponding to one of the stored datums.
    region_name: str
        A region name corrisponding to a VDatum subfolder name.

    Raises
    ------
    ValueError
        If an input string is not found in the datum definition database a
        value error is returned.

    Returns
    -------
    regional_pipeline : str
        A string describing the pipeline to use to convert between the
        provided datums.

    """
    from_datum = from_datum.lower()
    to_datum = to_datum.lower()
    if from_datum == to_datum:
        return None

    _validate_datum_names(from_datum, to_datum)
    input_datum_def = datum_definition[from_datum].copy()
    output_datum_def = datum_definition[to_datum].copy()
    input_datum_def, output_datum_def = compare_datums(input_datum_def, output_datum_def)
    reversed_input_def = inverse_datum_def(input_datum_def)
    transformation_def = ['proj=pipeline', *reversed_input_def, *output_datum_def]
    pipeline = ' step '.join(transformation_def)
    regional_pipeline = pipeline.replace('{region_name}', region_name)
    return regional_pipeline


def _validate_datum_names(from_datum: str, to_datum: str):
    """
    Raise an error if the provided datum names are not found in the datum
    definition dictionary.

    Parameters
    ----------
    from_datum : str
        DESCRIPTION.
    to_datum : str
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if from_datum not in datum_definition:
        raise ValueError(f'Input datum {from_datum} not found in datum definitions.')
    if to_datum not in datum_definition:
        raise ValueError(f'Output datum {to_datum} not found in datum definitions.')


def compare_datums(in_datum_def: [str], out_datum_def: [str]) -> [[str], [str]]:
    """
    Compare two lists describing the datums.  Remove common parts of the
    definition starting from the first entry.  Stop when they do not agree.

    Parameters
    ----------
    in_datum_def : [str]
        The datum definition as described in the datum defition database.
    out_datum_def : [str]
        The datum definition as described in the datum defition database.

    Returns
    -------
    [[str],[str]]
        A reduced list of the input datum and output datum layers.

    """
    num_to_compare = min(len(in_datum_def), len(out_datum_def))
    remove_these = []
    for n in range(num_to_compare):
        if in_datum_def[n] == out_datum_def[n]:
            remove_these.append(in_datum_def[n])
    for rmve in remove_these:
        in_datum_def.remove(rmve)
        out_datum_def.remove(rmve)
    return [in_datum_def, out_datum_def]


def inverse_datum_def(datum_def: [str]) -> [str]:
    """
    Reverse the order of the datum definition list and prepend 'inv' to each
    layer.

    Parameters
    ----------
    datum_def : [str]
        A list describing the layers of a datum definition.

    Returns
    -------
    [str]
        The provided list reversed with 'inv' prepended to each layer.

    """
    inverse = []
    for layer in datum_def[::-1]:
        inverse.append(' '.join(['inv', layer]))
    return inverse
