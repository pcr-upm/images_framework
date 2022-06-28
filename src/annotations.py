#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@geoaitech.com'

import uuid
import numpy as np
from .categories import ObjComposite


class SatelliteObject:
    """
    Object localised within the image.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.obb = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.multipolygon = [np.array([[[-1, -1]], [[-1, -1]], [[-1, -1]]])]
        self.categories = ObjComposite()


class SatelliteImage:
    """
    Satellite image.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.gsd = -1
        self.nadir_angle = -1
        self.timestamp = ''
        self.objects = list([])

    def add_object(self, obj):
        self.objects.append(obj)


class SatelliteSequence:
    """
    Set of satellite imagery in the same location (time-lapse).
    """
    def __init__(self):
        self.images = list([])

    def add_image(self, img):
        self.images.append(img)

    def clear(self):
        self.images.clear()
