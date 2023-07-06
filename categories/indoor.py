#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Indoor(Oi):
    # COCO
    Oi.INDOOR.BOOK = Name('Book')
    Oi.INDOOR.CLOCK = Name('Clock')
    Oi.INDOOR.VASE = Name('Vase')
    Oi.INDOOR.SCISSORS = Name('Scissors')
    Oi.INDOOR.TEDDY_BEAR = Name('Teddy bear')
    Oi.INDOOR.HAIR_DRIER = Name('Hair drier')
    Oi.INDOOR.TOOTHBRUSH = Name('Toothbrush')
    Oi.INDOOR.HAIR_BRUSH = Name('Hair brush')
