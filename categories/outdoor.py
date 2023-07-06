#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Outdoor(Oi):
    # COCO
    Oi.OUTDOOR.TRAFFIC_LIGHT = Name('Traffic light')
    Oi.OUTDOOR.FIRE_HYDRANT = Name('Fire hydrant')
    Oi.OUTDOOR.STREET_SIGN = Name('Street sign')
    Oi.OUTDOOR.STOP_SIGN = Name('Stop sign')
    Oi.OUTDOOR.PARKING_METER = Name('Parking meter')
    Oi.OUTDOOR.BENCH = Name('Bench')
