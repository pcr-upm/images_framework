#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
from .component import Component
from .datasets import Database


class Generation(Component):
    """
    Represent generation instances in the composition.
    """
    def __init__(self):
        super().__init__(5)
        self.database = None

    def parse_options(self, params):
        import argparse
        import itertools
        choices = list(itertools.chain.from_iterable([db().get_names() for db in Database.__subclasses__()]))
        choices.append('all')
        parser = argparse.ArgumentParser(prog='Generation', add_help=False)
        parser.add_argument('--database', required=True, choices=choices,
                            help='Select database model.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.database = args.database
        return unknown

    @abc.abstractmethod
    def train(self, anns_train, anns_valid):
        pass

    @abc.abstractmethod
    def load(self, mode):
        pass

    @abc.abstractmethod
    def process(self, ann, pred):
        pass

    def show(self, viewer, ann, pred):
        return

    def evaluate(self, fs, ann, pred):
        return

    def save(self, dirname, pred):
        return
