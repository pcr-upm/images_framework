#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from pcr_framework.src.categories import Category as Oi


class Panel(Oi):
    Oi.SOLAR_PANEL.UNCLASSIFIED = 'Unclassified'
    Oi.SOLAR_PANEL.NO_DAMAGE = 'No damage'
    Oi.SOLAR_PANEL.HOT_CELL = 'Hot cell'
    Oi.SOLAR_PANEL.HOT_CELL_CHAIN = 'Hot cell chain'
    Oi.SOLAR_PANEL.SEVERAL_HOT_CELLS = 'Several hot cells'
    Oi.SOLAR_PANEL.HOT_SPOT = 'Hot spot'
    Oi.SOLAR_PANEL.SEVERAL_HOT_SPOTS = 'Several hot spots'
    Oi.SOLAR_PANEL.POTENTIAL_INDUCED_DEGRADATION = 'Potential induced degradation'
    Oi.SOLAR_PANEL.DIRTY_PANEL = 'Dirty panel'
    Oi.SOLAR_PANEL.BROKEN_PANEL = 'Broken panel'
    Oi.SOLAR_PANEL.DISCONNECTED_PANEL = 'Disconnected panel'
    Oi.SOLAR_PANEL.SHADES = 'Shades'
    Oi.SOLAR_PANEL.SHADES_HOT_CELL_CHAIN = 'Shades + Hot cell chain'
    Oi.SOLAR_PANEL.MELTED_FUSES = 'Melted fuses'
