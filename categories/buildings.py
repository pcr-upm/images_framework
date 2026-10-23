#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from pcr_framework.src.categories import Category as Oi


class Building(Oi):
    # XView2
    Oi.BUILDING.UNCLASSIFIED = 'Unclassified'
    Oi.BUILDING.NO_DAMAGE = 'No damage'
    Oi.BUILDING.MINOR_DAMAGE = 'Minor damage'
    Oi.BUILDING.MAJOR_DAMAGE = 'Major damage'
    Oi.BUILDING.DESTROYED = 'Destroyed'
    # Xview
    Oi.BUILDING.HUT_TENT = 'Hut/tent'
    Oi.BUILDING.SHED = 'Shed'
    Oi.BUILDING.AIRCRAFT_HANGAR = 'Aircraft hangar'
    Oi.BUILDING.DAMAGED_BUILDING = 'Damaged building'
    Oi.BUILDING.FACILITY = 'Facility'
