#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

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
    ALIGNMENT = 'alignment'
    RECOGNITION = 'recognition'
    GENERATION = 'generation'
    ALL = 'all'


@unique
class Detectors(Enum):
    """
    Different object detectors algorithms.
    """
    SSD16_DETECTION = 'ssd16_detection'
    RETINANET17_DETECTION = 'retinanet17_detection'
    SCRDET19_DETECTION = 'scrdet19_detection'
    ORP22_DETECTION = 'orp22_detection'
    ARCKPADEL_DETECTION = 'arckpadel_detection'


@unique
class Segmentators(Enum):
    """
    Different object segmentation algorithms.
    """
    UNET15_SEGMENTATION = 'unet15_segmentation'
    OCR20_SEGMENTATION = 'ocr20_segmentation'


@unique
class Aligners(Enum):
    """
    Different object alignment algorithms.
    """
    CIARP17_HEADPOSE = 'ciarp17_headpose'
    OPAL23_HEADPOSE = 'opal23_headpose'
    STUDENTS_HEADPOSE = 'students_headpose'
    ACCESS25_HEADPOSE = 'access25_headpose'
    STUDENTS_LANDMARKS = 'students_landmarks'
    KAZEMI14_LANDMARKS = 'kazemi14_landmarks'
    DAD22_LANDMARKS = 'dad22_landmarks'
    ICITS24_LANDMARKS = 'icits24_landmarks'
    WDANN_LANDMARKS = 'wdann_landmarks'
    OPENPOSE_LANDMARKS = 'openpose_landmarks'
    PAMI20_RECONSTRUCTION = 'pami20_reconstruction'
    EFFICIENTNET21_RECONSTRUCTION = 'efficientnet21_reconstruction'


@unique
class Recognitors(Enum):
    """
    Different object recognition algorithms.
    """
    RESNET15_RECOGNITION = 'resnet15_recognition'
    DANN16_RECOGNITION = 'dann16_recognition'


@unique
class Generators(Enum):
    """
    Different image generation algorithms.
    """
    CONTROLNET23_GENERATION = 'controlnet23_generation'
