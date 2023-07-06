#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Food(Oi):
    # COCO
    Oi.FOOD.BANANA = Name('Banana')
    Oi.FOOD.APPLE = Name('Apple')
    Oi.FOOD.SANDWICH = Name('Sandwich')
    Oi.FOOD.ORANGE = Name('Orange')
    Oi.FOOD.BROCCOLI = Name('Broccoli')
    Oi.FOOD.CARROT = Name('Carrot')
    Oi.FOOD.HOT_DOG = Name('Hot dog')
    Oi.FOOD.PIZZA = Name('Pizza')
    Oi.FOOD.DONUT = Name('Donut')
    Oi.FOOD.CAKE = Name('Cake')
