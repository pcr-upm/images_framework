#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'


class Label(str):
    """
    Category label: a string that can contain subcategories (e.g. Category.FACE.ANGER == 'Anger').
    """
    def __setattr__(self, key, value):
        super().__setattr__(key, Label(value) if isinstance(value, str) else value)


class Taxonomy(type):
    """
    Metaclass that turns every uppercase string attribute into a Label.
    """
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        for key, value in namespace.items():
            if key.isupper() and isinstance(value, str):
                setattr(cls, key, Label(value))


class Category(metaclass=Taxonomy):
    """
    Category label.
    """
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
