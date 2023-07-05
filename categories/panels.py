#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from images_framework.src.categories import Name, Category as Oi


class Panel(Oi):
    Oi.SOLAR_PANEL.UNCLASSIFIED = Name('Unclassified')
    Oi.SOLAR_PANEL.NO_DAMAGE = Name('No damage')
    Oi.SOLAR_PANEL.HOT_CELL = Name('Hot cell')
    Oi.SOLAR_PANEL.HOT_CELL_CHAIN = Name('Hot cell chain')
    Oi.SOLAR_PANEL.SEVERAL_HOT_CELLS = Name('Several hot cells')
    Oi.SOLAR_PANEL.HOT_SPOT = Name('Hot spot')
    Oi.SOLAR_PANEL.SEVERAL_HOT_SPOTS = Name('Several hot spots')
    Oi.SOLAR_PANEL.POTENTIAL_INDUCED_DEGRADATION = Name('Potential induced degradation')
    Oi.SOLAR_PANEL.DIRTY_PANEL = Name('Dirty panel')
    Oi.SOLAR_PANEL.BROKEN_PANEL = Name('Broken panel')
    Oi.SOLAR_PANEL.DISCONNECTED_PANEL = Name('Disconnected panel')
    Oi.SOLAR_PANEL.SHADES = Name('Shades')
    Oi.SOLAR_PANEL.SHADES_HOT_CELL_CHAIN = Name('Shades + Hot cell chain')
    Oi.SOLAR_PANEL.MELTED_FUSES = Name('Melted fuses')
