#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Appliance(Oi):
    # COCO
    Oi.APPLIANCE.MICROWAVE = Name('Microwave')
    Oi.APPLIANCE.OVEN = Name('Oven')
    Oi.APPLIANCE.TOASTER = Name('Toaster')
    Oi.APPLIANCE.SINK = Name('Sink')
    Oi.APPLIANCE.REFRIGERATOR = Name('Refrigerator')
    Oi.APPLIANCE.BLENDER = Name('Blender')
