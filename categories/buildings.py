#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Building(Oi):
    Oi.BUILDING.UNCLASSIFIED = Name('Unclassified')
    Oi.BUILDING.NO_DAMAGE = Name('No damage')
    Oi.BUILDING.MINOR_DAMAGE = Name('Minor damage')
    Oi.BUILDING.MAJOR_DAMAGE = Name('Major damage')
    Oi.BUILDING.DESTROYED = Name('Destroyed')
