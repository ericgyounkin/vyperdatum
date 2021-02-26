"""
pipeline.py

The proj pipeline string generator.

The pipelines are currently stored as a dictionary, but a lightweight database
likely makes more sense so that specific datum and epochs (and versions?) can
be queried.

We need a WKT to lookup generator, unless the WKT contains the information to
specify specific tranformation layers.

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
    }


def get_regional_vertical_pipelines(from_datum: str, to_datum: str, region_names: [str]) -> [str]:
    """
    Return a string describing the pipeline to use to convert between the
    provided datums. A placeholder for the regions is returned within the
    string as {region_name}.

    Parameters
    ----------
    from_datum : str
        A string corresponding to one of the stored datums.
    to_datum : str
        A string corresponding to one of the stored datums.
    region_names: [str]
        A list of region names corrisponding to VDatum subfolder names.

    Raises
    ------
    ValueError
        If an input string is not found in the datum definition database a
        value error is returned.

    Returns
    -------
    regional_pipelines : [str]
        A string describing the pipeline to use to convert between the
        provided datums. A placeholder for the regions is returned within the
        string as {region_name}.

    """

    _validate_datum_names(from_datum, to_datum)
    input_datum_def = datum_definition[from_datum]
    output_datum_def = datum_definition[to_datum]
    input_datum_def, output_datum_def = compare_datums(input_datum_def, output_datum_def)
    reversed_input_def = inverse_datum_def(input_datum_def)
    transformation_def = ['proj=pipeline', *reversed_input_def, *output_datum_def]
    pipeline = ' step '.join(transformation_def)
    regional_pipelines = []
    for region_name in region_names:
        regional_pipelines.append(pipeline.replace('{region_name}', region_name))
    return regional_pipelines


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
    for n in range(num_to_compare):
        if in_datum_def[n] == out_datum_def[n]:
            in_datum_def.pop(n)
            out_datum_def.pop(n)
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
