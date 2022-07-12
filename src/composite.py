#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

from .component import Component


class Composite(Component):
    """
    Store child components. Implement child-related operations in the Component interface.
    """
    def __init__(self):
        super().__init__(0)
        self._children = list([])

    def parse_options(self, params):
        for child in self._children:
            child.parse_options(params)

    def train(self, anns_train, anns_valid):
        for child in self._children:
            child.train(anns_train, anns_valid)

    def load(self, mode):
        for child in self._children:
            child.load(mode)

    def process(self, ann, pred):
        for child in self._children:
            child.process(ann, pred)

    def show(self, viewer, ann, pred):
        for child in self._children:
            child.show(viewer, ann, pred)

    def evaluate(self, fs, ann, pred):
        for child in self._children:
            child.evaluate(fs, ann, pred)

    def save(self, dirname, pred):
        for child in self._children:
            child.save(dirname, pred)

    def add(self, component):
        self._children.append(component)

    def contains_part(self, part):
        for child in self._children:
            if child.get_component_class() == part:
                return True
        return False
