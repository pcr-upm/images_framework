#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from enum import Enum, unique


@unique
class FaceLandmarkPart(Enum):
    """
    Face landmark part label.
    """
    LEYEBROW = 'leyebrow'
    REYEBROW = 'reyebrow'
    LEYE = 'leye'
    REYE = 'reye'
    NOSE = 'nose'
    TMOUTH = 'tmouth'
    BMOUTH = 'bmouth'
    LEAR = 'lear'
    REAR = 'rear'
    CHIN = 'chin'
    FOREHEAD = 'forehead'


@unique
class PersonLandmarkPart(Enum):
    """
    Person landmark part label.
    """
    NOSE = 'nose'
    LEYE = 'leye'
    REYE = 'reye'
    LEAR = 'lear'
    REAR = 'rear'
    LSHOULDER = 'lshoulder'
    RSHOULDER = 'rshoulder'
    LELBOW = 'lelbow'
    RELBOW = 'relbow'
    LWRIST = 'lwrist'
    RWRIST = 'rwrist'
    LHIP = 'lhip'
    RHIP = 'rhip'
    LKNEE = 'lknee'
    RKNEE = 'rknee'
    LANKLE = 'lankle'
    RANKLE = 'rankle'
