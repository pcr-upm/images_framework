#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Sport(Oi):
    # COCO
    Oi.SPORTS.FRISBEE = Name('Frisbee')
    Oi.SPORTS.SKIS = Name('Skis')
    Oi.SPORTS.SNOWBOARD = Name('Snowboard')
    Oi.SPORTS.SPORTS_BALL = Name('Sports ball')
    Oi.SPORTS.KITE = Name('Kite')
    Oi.SPORTS.BASEBALL_BAT = Name('Baseball bat')
    Oi.SPORTS.BASEBALL_GLOVE = Name('Baseball glove')
    Oi.SPORTS.SKATEBOARD = Name('Skateboard')
    Oi.SPORTS.SURFBOARD = Name('Surfboard')
    Oi.SPORTS.TENNIS_RACKET = Name('Tennis racket')
