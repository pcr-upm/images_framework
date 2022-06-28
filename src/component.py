#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@geoaitech.com'

import abc


class SatelliteComponent(object):
    """
    Declare the interface for objects in the composition.
    Implement default behavior for the interface common to all classes, as appropriate.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, part):
        self._part = part

    def get_component_class(self):
        return self._part

    @abc.abstractmethod
    def parse_options(self, params):
        pass

    @abc.abstractmethod
    def train(self, anns_train, anns_valid):
        pass

    @abc.abstractmethod
    def load(self, mode):
        pass

    @abc.abstractmethod
    def process(self, ann, pred):
        pass

    @abc.abstractmethod
    def show(self, viewer, ann, pred):
        pass

    @abc.abstractmethod
    def evaluate(self, fs, ann, pred):
        pass

    @abc.abstractmethod
    def save(self, dirname, pred):
        pass
