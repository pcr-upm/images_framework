#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Building(Oi):
    # XView2
    Oi.BUILDING.UNCLASSIFIED = Name('Unclassified')
    Oi.BUILDING.NO_DAMAGE = Name('No damage')
    Oi.BUILDING.MINOR_DAMAGE = Name('Minor damage')
    Oi.BUILDING.MAJOR_DAMAGE = Name('Major damage')
    Oi.BUILDING.DESTROYED = Name('Destroyed')
    # Xview
    Oi.BUILDING.HUT_TENT = Name('Hut/tent')
    Oi.BUILDING.SHED = Name('Shed')
    Oi.BUILDING.AIRCRAFT_HANGAR = Name('Aircraft hangar')
    Oi.BUILDING.DAMAGED_BUILDING = Name('Damaged building')
    Oi.BUILDING.FACILITY = Name('Facility')
