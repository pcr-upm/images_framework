#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import uuid
import numpy as np
from enum import Enum
from .categories import Name
from images_framework.alignment.landmarks import PersonLandmarkPart as Pl


class GenericCategory:
    """
    Generic category data.
    """
    def __init__(self, label: Name, score=-1):
        self.label = label
        self.score = score


class GenericAttribute:
    """
    Generic attribute data.
    """
    def __init__(self, label, value, confidence=-1):
        self.label = label
        self.value = value
        self.confidence = confidence


class GenericLandmark:
    """
    Generic landmark data.
    """
    def __init__(self, label: int, part: Enum, pos: tuple, visible: bool, confidence=-1):
        self.label = label
        self.part = part
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


class PersonObject(GenericObject):
    """
    Person object inherits from the generic object class.
    """
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf, HandLandmarkPart as Ph, BodyLandmarkPart as Pb
        super().__init__()
        self.headpose = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        self.attributes = list([])
        self.landmarks = {part.value: {} for part in Pl}
        self.landmarks[Pl.FACE.value] = {part.value: list([]) for part in Pf}
        self.landmarks[Pl.HAND.value] = {part.value: list([]) for part in Ph}
        self.landmarks[Pl.BODY.value] = {part.value: list([]) for part in Pb}

    def add_attribute(self, att: GenericAttribute):
        self.attributes.append(att)

    def add_landmark(self, lnd: GenericLandmark, part: Pl):
        self.landmarks[part.value].setdefault(lnd.part.value, list([])).append(lnd)

    def clear(self):
        self.attributes.clear()
        self.landmarks[Pl.FACE.value].clear()
        self.landmarks[Pl.HAND.value].clear()
        self.landmarks[Pl.BODY.value].clear()


class DiffusionObject(PersonObject):
    """
    Diffusion object inherits from the person object class.
    """
    def __init__(self):
        super().__init__()
        self.control = ''
        self.prompt = ''


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
