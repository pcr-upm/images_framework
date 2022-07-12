#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import uuid
import numpy as np
from .categories import Category


class GenericCategory:
    """
    Generic category data.
    """
    def __init__(self, label: Category, score=-1):
        self.label = label
        self.score = score


class GenericObject:
    """
    Generic object data.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.obb = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.multipolygon = [np.array([[[-1, -1]], [[-1, -1]], [[-1, -1]]])]
        self.categories = list([])

    def add_category(self, category: GenericCategory):
        self.categories.append(category)


class GenericImage:
    """
    Generic image data.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.timestamp = ''
        self.objects = list([])

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)


class AerialImage(GenericImage):
    """
    Aerial image inherits from the generic image class.
    """
    def __init__(self, filename):
        super().__init__(filename)
        self.gsd = -1
        self.nadir_angle = -1


class GenericGroup:
    """
    Set of images or sequence of frames from a video.
    """
    def __init__(self):
        self.images = list([])

    def add_image(self, img: GenericImage):
        self.images.append(img)

    def clear(self):
        self.images.clear()
