#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from pcr_framework.src.categories import Category as Oi


class Outdoor(Oi):
    # COCO
    Oi.OUTDOOR.TRAFFIC_LIGHT = 'Traffic light'
    Oi.OUTDOOR.FIRE_HYDRANT = 'Fire hydrant'
    Oi.OUTDOOR.STREET_SIGN = 'Street sign'
    Oi.OUTDOOR.STOP_SIGN = 'Stop sign'
    Oi.OUTDOOR.PARKING_METER = 'Parking meter'
    Oi.OUTDOOR.BENCH = 'Bench'
