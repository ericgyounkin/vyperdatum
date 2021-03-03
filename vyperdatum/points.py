# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:12:21 2021

@author: grice
"""

'''
Read points
Write points
Convert points

'''
import os
import numpy as np
import gdal
gdal.UseExceptions()

class VyperPoints:
    def tranform_dataset(self, xy_dataset: np.ndarray):
        # get out vertical and horizontal datums
        input_datum = ''
        # get x and y as an array
        x = []
        y = []
        z = np.full(len(y), np.nan)
        r = np.full(len(y), 'NoRegion')
        # get bounds
        # get regions
        regions = []
        # get the pipeline
        for region in regions:
            pipeline = get_regional_pipeline(input_datum, self.output_datum, region)
            sep = self.run_pipeline(x, y, pipeline)
            # find the non-nodata values and move into z
            # populate these same indicies in the r array with the same values
        return z, r