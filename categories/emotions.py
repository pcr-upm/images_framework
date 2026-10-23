#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from pcr_framework.src.categories import Category as Oi


class Emotion(Oi):
    # FER-2013
    Oi.FACE.ANGER = 'Anger'
    Oi.FACE.CONTEMPT = 'Contempt'
    Oi.FACE.DISGUST = 'Disgust'
    Oi.FACE.FEAR = 'Fear'
    Oi.FACE.HAPPINESS = 'Happiness'
    Oi.FACE.NEUTRAL = 'Neutral'
    Oi.FACE.SADNESS = 'Sadness'
    Oi.FACE.SURPRISE = 'Surprise'
    Oi.FACE.OTHER = 'Other'
