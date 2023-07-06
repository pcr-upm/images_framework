#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Accessory(Oi):
    # COCO
    Oi.ACCESSORY.HAT = Name('Hat')
    Oi.ACCESSORY.BACKPACK = Name('Backpack')
    Oi.ACCESSORY.UMBRELLA = Name('Umbrella')
    Oi.ACCESSORY.SHOE = Name('Shoe')
    Oi.ACCESSORY.EYE_GLASSES = Name('Eye glasses')
    Oi.ACCESSORY.HANDBAG = Name('Handbag')
    Oi.ACCESSORY.TIE = Name('Tie')
    Oi.ACCESSORY.SUITCASE = Name('Suitcase')
