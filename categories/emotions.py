#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Emotion(Oi):
    # FER-2013
    Oi.FACE.ANGER = Name('Anger')
    Oi.FACE.CONTEMPT = Name('Contempt')
    Oi.FACE.DISGUST = Name('Disgust')
    Oi.FACE.FEAR = Name('Fear')
    Oi.FACE.HAPPINESS = Name('Happiness')
    Oi.FACE.NEUTRAL = Name('Neutral')
    Oi.FACE.SADNESS = Name('Sadness')
    Oi.FACE.SURPRISE = Name('Surprise')
