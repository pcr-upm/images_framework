#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Furniture(Oi):
    # COCO
    Oi.FURNITURE.CHAIR = Name('Chair')
    Oi.FURNITURE.COUCH = Name('Couch')
    Oi.FURNITURE.POTTED_PLANT = Name('Potted plant')
    Oi.FURNITURE.BED = Name('Bed')
    Oi.FURNITURE.MIRROR = Name('Mirror')
    Oi.FURNITURE.DINING_TABLE = Name('Dining table')
    Oi.FURNITURE.WINDOW = Name('Window')
    Oi.FURNITURE.DESK = Name('Desk')
    Oi.FURNITURE.TOILET = Name('Toilet')
    Oi.FURNITURE.DOOR = Name('Door')
