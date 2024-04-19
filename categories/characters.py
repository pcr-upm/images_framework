#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Character(Oi):
    Oi.CHARACTER.ZERO = Name('0')
    Oi.CHARACTER.ONE = Name('1')
    Oi.CHARACTER.TWO = Name('2')
    Oi.CHARACTER.THREE = Name('3')
    Oi.CHARACTER.FOUR = Name('4')
    Oi.CHARACTER.FIVE = Name('5')
    Oi.CHARACTER.SIX = Name('6')
    Oi.CHARACTER.SEVEN = Name('7')
    Oi.CHARACTER.EIGHT = Name('8')
    Oi.CHARACTER.NINE = Name('9')
