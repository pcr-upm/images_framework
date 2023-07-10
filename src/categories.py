#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
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
class PersonLandmarkPart(FaceLandmarkPart):
    """
    Person landmark part label.
    """
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


class Name:
    def __init__(self, name):
        self.name = name


class Category(object):
    """
    Category label.
    """
    __metaclass__ = abc.ABCMeta
    BACKGROUND = Name('Background')
    PERSON = Name('Person')
    ANIMAL = Name('Animal')
    FACE = Name('Face')
    BUILDING = Name('Building')
    SOLAR_PANEL = Name('Solar panel')
    VEHICLE = Name('Vehicle')
    # COCO
    ACCESSORY = Name('Accessory')
    INDOOR = Name('Indoor')
    OUTDOOR = Name('Outdoor')
    SPORTS = Name('Sports')
    KITCHEN = Name('Kitchen')
    FOOD = Name('Food')
    FURNITURE = Name('Furniture')
    ELECTRONIC = Name('Electronic')
    APPLIANCE = Name('Appliance')
    # DOTA
    STORAGE_TANK = Name('Storage tank')
    BASEBALL_DIAMOND = Name('Baseball diamond')
    TENNIS_COURT = Name('Tennis court')
    BASKETBALL_COURT = Name('Basketball court')
    GROUND_TRACK_FIELD = Name('Ground track field')
    HARBOR = Name('Harbor')
    BRIDGE = Name('Bridge')
    SWIMMING_POOL = Name('Swimming pool')
    ROUNDABOUT = Name('Roundabout')
    SOCCER_BALL_FIELD = Name('Soccer ball field')
    AIRPORT = Name('Airport')
    HELIPAD = Name('Helipad')
    # XView
    SHIPPING_CONTAINER = Name('Shipping container')
    SHIPPING_CONTAINER_LOT = Name('Shipping container lot')
    PYLON = Name('Pylon')
    TOWER_STRUCTURE = Name('Tower structure')
    CONSTRUCTION_SITE = Name('Construction site')
