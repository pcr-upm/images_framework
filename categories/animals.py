#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Animal(Oi):
    # COCO
    Oi.ANIMAL.BIRD = Name('Bird')
    Oi.ANIMAL.CAT = Name('Cat')
    Oi.ANIMAL.DOG = Name('Dog')
    Oi.ANIMAL.HORSE = Name('Horse')
    Oi.ANIMAL.SHEEP = Name('Sheep')
    Oi.ANIMAL.COW = Name('Cow')
    Oi.ANIMAL.ELEPHANT = Name('Elephant')
    Oi.ANIMAL.BEAR = Name('Bear')
    Oi.ANIMAL.ZEBRA = Name('Zebra')
    Oi.ANIMAL.GIRAFFE = Name('Giraffe')
