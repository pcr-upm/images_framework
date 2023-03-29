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


class FaceAttribute:
    """
    Face attribute data.
    """
    def __init__(self, label, value, confidence=-1):
        self.label = label
        self.value = value
        self.confidence = confidence


class FaceLandmark:
    """
    Face landmark data.
    """
    def __init__(self, label, pos, visible, confidence=-1):
        self.label = label
        self.pos = pos  # (pt_x, pt_y)
        self.visible = visible
        self.confidence = confidence


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

    def clear(self):
        self.categories.clear()


class FaceObject(GenericObject):
    """
    Face object inherits from the generic object class.
    """
    def __init__(self):
        super().__init__()
        self.headpose = np.array([-1, -1, -1])  # np.matrix([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        self.attributes = list([])
        self.landmarks = list([])

    def add_attribute(self, att: FaceAttribute):
        self.attributes.append(att)

    def add_landmark(self, lnd: FaceLandmark):
        self.landmarks.append(lnd)

    def clear(self):
        self.attributes.clear()
        self.landmarks.clear()


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

    def clear(self):
        self.objects.clear()


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