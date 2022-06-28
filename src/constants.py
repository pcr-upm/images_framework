#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@geoaitech.com'

from enum import Enum, unique


@unique
class Modes(Enum):
    """
    Different types of experiments.
    """
    TRAIN = 'train'
    TEST = 'test'


@unique
class Parts(Enum):
    """
    Different types of computer vision modules to use.
    """
    DETECTION = 'detection'
    SEGMENTATION = 'segmentation'
    RECOGNITION = 'recognition'
    ALL = 'all'


@unique
class Detectors(Enum):
    """
    Different object detectors algorithms.
    """
    RETINANET17_DETECTION = 'retinanet17_detection'
    SCRDET19_DETECTION = 'scrdet19_detection'


@unique
class Segmentators(Enum):
    """
    Different object segmentation algorithms.
    """
    UNET15_SEGMENTATION = 'unet15_segmentation'
    SEGNET17_SEGMENTATION = 'segnet17_segmentation'
    OCR20_SEGMENTATION = 'ocr20_segmentation'


@unique
class Recognitors(Enum):
    """
    Different object recognition algorithms.
    """
    RESNET15_RECOGNITION = 'resnet15_recognition'
    HU19_RECOGNITION = 'hu19_recognition'
