#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Kitchen(Oi):
    # COCO
    Oi.KITCHEN.BOTTLE = Name('Bottle')
    Oi.KITCHEN.PLATE = Name('Plate')
    Oi.KITCHEN.WINE_GLASS = Name('Wine glass')
    Oi.KITCHEN.CUP = Name('Cup')
    Oi.KITCHEN.FORK = Name('Fork')
    Oi.KITCHEN.KNIFE = Name('Knife')
    Oi.KITCHEN.SPOON = Name('Spoon')
    Oi.KITCHEN.BOWL = Name('Bowl')
