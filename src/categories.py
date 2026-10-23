#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc


class Category(object):
    """
    Category label.
    """
    __metaclass__ = abc.ABCMeta
    BACKGROUND = 'Background'
    PERSON = 'Person'
    ACTION = 'Action'
    ANIMAL = 'Animal'
    FACE = 'Face'
    BUILDING = 'Building'
    SOLAR_PANEL = 'Solar panel'
    VEHICLE = 'Vehicle'
    CHARACTER = 'Character'
    BALL = 'Ball'
    RACKET = 'Racket'
    # COCO
    ACCESSORY = 'Accessory'
    INDOOR = 'Indoor'
    OUTDOOR = 'Outdoor'
    SPORTS = 'Sports'
    KITCHEN = 'Kitchen'
    FOOD = 'Food'
    FURNITURE = 'Furniture'
    ELECTRONIC = 'Electronic'
    APPLIANCE = 'Appliance'
    # DOTA
    STORAGE_TANK = 'Storage tank'
    BASEBALL_DIAMOND = 'Baseball diamond'
    TENNIS_COURT = 'Tennis court'
    BASKETBALL_COURT = 'Basketball court'
    GROUND_TRACK_FIELD = 'Ground track field'
    HARBOR = 'Harbor'
    BRIDGE = 'Bridge'
    SWIMMING_POOL = 'Swimming pool'
    ROUNDABOUT = 'Roundabout'
    SOCCER_BALL_FIELD = 'Soccer ball field'
    AIRPORT = 'Airport'
    HELIPAD = 'Helipad'
    # XView
    SHIPPING_CONTAINER = 'Shipping container'
    SHIPPING_CONTAINER_LOT = 'Shipping container lot'
    PYLON = 'Pylon'
    TOWER_STRUCTURE = 'Tower structure'
    CONSTRUCTION_SITE = 'Construction site'
