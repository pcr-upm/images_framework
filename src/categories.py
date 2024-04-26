#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc


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
    CHARACTER = Name('Character')
    BALL = Name('Ball')
    RACKET = Name('Racket')
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
