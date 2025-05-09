#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from enum import Enum, unique


@unique
class PersonLandmarkPart(Enum):
    """
    Person landmark part label.
    """
    FACE = 'face'
    HAND = 'hand'
    BODY = 'body'


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
class HandLandmarkPart(Enum):
    """
    Hand landmark part label.
    """
    RWRIST = 'rwrist'
    RTHUMB = 'rthumb'
    RINDEX = 'rindex'
    RMIDDLE = 'rmiddle'
    RRING = 'rring'
    RPINKY = 'rpinky'
    LWRIST = 'lwrist'
    LTHUMB = 'lthumb'
    LINDEX = 'lindex'
    LMIDDLE = 'lmiddle'
    LRING = 'lring'
    LPINKY = 'lpinky'


@unique
class BodyLandmarkPart(Enum):
    """
    Body landmark part label.
    """
    NOSE = 'nose'
    LEYE = 'leye'
    REYE = 'reye'
    LEAR = 'lear'
    REAR = 'rear'
    NECK = 'neck'
    CHEST = 'chest'
    ABDOMEN = 'abdomen'
    RWRIST = 'rwrist'
    LWRIST = 'lwrist'
    LSHOULDER = 'lshoulder'
    RSHOULDER = 'rshoulder'
    LELBOW = 'lelbow'
    RELBOW = 'relbow'
    LHIP = 'lhip'
    RHIP = 'rhip'
    LKNEE = 'lknee'
    RKNEE = 'rknee'
    LANKLE = 'lankle'
    RANKLE = 'rankle'
    LTOE = 'ltoe'
    RTOE = 'rtoe'


lps = {FaceLandmarkPart: PersonLandmarkPart.FACE, HandLandmarkPart: PersonLandmarkPart.HAND, BodyLandmarkPart: PersonLandmarkPart.BODY}
