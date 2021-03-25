"""
WKT spec http://docs.opengeospatial.org/is/12-063r5/12-063r5.html#93

Need something to build wkt for our custom datum transformations

Just do the vertical since there aren't any custom horiz transforms that we do.  We can do a compound CRS with the known
horiz later.  How do we handle these custom vert transformations though?  We have no EPSG registered datum, so it won't be
something we can directly use in pyproj.  We have to build our own pipeline from a source custom vert to a source custom
vert.  Use this as a starting point:

VERTCRS["NOAA Chart Datum",
        BASEVERTCRS["NAD83(2011) Height",
                    VDATUM["NAD83(2011) Height"],
                    ID["EPSG",6319]],
        DERIVINGCONVERSION["NAD83(2011) Height to NOAA Mean Lower Low Water",
                           METHOD["VDatum_VXXX gtx grid transformation",
                             ID["EPSG",1084]]],
        VDATUM["NOAA Chart Datum"],
        CS[vertical,1],
          AXIS["gravity-related height (H)",up,
          LENGTHUNIT["metre",1]]]



VERTCRS[“NAVD88”,
        VDATUM[“North American Vertical Datum 1983”],
        CS[vertical,1],
        AXIS["gravity-related height (H)",up],
        LENGTHUNIT[“metre”,1]]]


CS[vertical,1],
                 AXIS["depth (D)",down,
                   LENGTHUNIT[“metre”,1.0]]


VERTCRS["Bonneville Reservoir height",
        BASEVERTCRS["MSL height",
                    VDATUM["Mean Sea Level"],
                    ID["EPSG",5714]],
        DERIVINGCONVERSION["Bonneville Reservoir normal pool level = MSL + 72 feet",
                           METHOD["Vertical Offset",ID["EPSG",9616]],
                           PARAMETER["dH",21.946,LENGTHUNIT["metre",1]]],
        VDATUM["Bonneville Reservoir height"],
        CS[vertical,1],
        AXIS["gravity-related height (H)",up,LENGTHUNIT["metre",1]],
        USAGE[SCOPE["NOAA ENC US5OR30M"],
              AREA["Bonneville Dam to Dalles Dam"],
              BBOX[45.59823,-121.94967,45.73293,-121.11338]],
]
"""


class CoordinateSystem:
    def __init__(self):
        pass


class BaseCoordinateSystem:
    def __init__(self):
        pass


class VerticalCoordinateSystem:
    def __init__(self):
        pass
