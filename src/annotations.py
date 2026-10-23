#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import uuid
import numpy as np
from enum import Enum
from typing import Any, NamedTuple
from scipy.spatial.transform import Rotation
from pcr_framework.regression.alignment.landmarks import PersonLandmarkPart as Pl


class Prediction(NamedTuple):
    """
    Predicted value with its confidence (-1 if not evaluated).
    """
    name: Any
    confidence: float = -1


class GenericCategory:
    """
    Generic category data.
    """
    def __init__(self, label: str, label_confidence=-1, fake=False, fake_confidence=-1):
        self.label = Prediction(label, label_confidence)
        self.fake = Prediction(fake, fake_confidence)


class TemporalCategory(GenericCategory):
    """
    Temporal category inherits from the generic category class.
    """
    def __init__(self, segment: tuple, label: str, label_confidence=-1, fake=False, fake_confidence=-1):
        super().__init__(label, label_confidence, fake, fake_confidence)
        self.segment = segment


class GenericAnnotation:
    """
    Mixin for any element (video, image, object) that can have one or several categories.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.categories = list([])

    def add_category(self, category: GenericCategory):
        self.categories.append(category)

    def has_category(self, label: str):
        return any(category.label.name == label for category in self.categories)

    def clear(self):
        self.categories.clear()


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


class GenericObject(GenericAnnotation):
    """
    Generic object data.
    """
    def __init__(self):
        super().__init__()
        self.bb = (-1, -1, -1, -1)
        self.obb = (-1, -1, -1, -1, -1, -1, -1, -1)
        self.multipolygon = [np.array([[[-1, -1]], [[-1, -1]], [[-1, -1]]])]


class PersonObject(GenericObject):
    """
    Person object inherits from the generic object class.
    """
    def __init__(self):
        from pcr_framework.regression.alignment.landmarks import FaceLandmarkPart as Pf, HandLandmarkPart as Ph, BodyLandmarkPart as Pb
        super().__init__()
        self.headpose = Rotation.from_euler('YXZ', [-1, -1, -1], degrees=True).as_matrix()
        self.landmarks = {part.value: {} for part in Pl}
        self.landmarks[Pl.FACE.value] = {part.value: list([]) for part in Pf}
        self.landmarks[Pl.HAND.value] = {part.value: list([]) for part in Ph}
        self.landmarks[Pl.BODY.value] = {part.value: list([]) for part in Pb}

    def add_landmark(self, lnd: GenericLandmark, part: Pl):
        self.landmarks[part.value].setdefault(lnd.part.value, list([])).append(lnd)

    def clear(self):
        super().clear()
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


class GenericImage(GenericAnnotation):
    """
    Generic image data.
    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.timestamp = ''
        self.objects = list([])

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)

    def clear(self):
        super().clear()
        self.objects.clear()


class AerialImage(GenericImage):
    """
    Aerial image inherits from the generic image class.
    """
    def __init__(self, filename):
        super().__init__(filename)
        self.gsd = -1
        self.nadir_angle = -1


class GenericVideo(GenericAnnotation):
    """
    Generic video data or sequence of frames.
    """
    def __init__(self, filename=None):
        super().__init__()
        self.filename = filename
        self.frames = -1
        self.duration = -1
        self.images = list([])
        self.actions = list([])

    def add_image(self, img: GenericImage):
        self.images.append(img)

    def add_action(self, action: TemporalCategory):
        self.actions.append(action)

    def clear(self):
        super().clear()
        self.images.clear()
        self.actions.clear()
