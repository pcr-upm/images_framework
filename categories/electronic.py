#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Electronic(Oi):
    # COCO
    Oi.ELECTRONIC.TV = Name('Tv')
    Oi.ELECTRONIC.LAPTOP = Name('Laptop')
    Oi.ELECTRONIC.MOUSE = Name('Mouse')
    Oi.ELECTRONIC.REMOTE = Name('Remote')
    Oi.ELECTRONIC.KEYBOARD = Name('Keyboard')
    Oi.ELECTRONIC.CELL_PHONE = Name('Cell phone')
