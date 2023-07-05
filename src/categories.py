#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
from enum import Enum, unique


@unique
class FaceLandmarkPart(Enum):
    """
    Landmark part label.
    """
    LEYEBROW = 'leyebrow'
    REYEBROW = 'reyebrow'
    LEYE = 'leye'
    REYE = 'reye'
    NOSE = 'nose'
    TMOUTH = 'tmouth'
    BMOUTH = 'bmouth'
    LEAR = 'lear'
    REAR = 'rear'
    CHIN = 'chin'
    FOREHEAD = 'forehead'


class Name:
    def __init__(self, name):
        self.name = name


class Category(object):
    """
    Category label.
    """
    __metaclass__ = abc.ABCMeta
    BACKGROUND = Name('Background')
    FACE = Name('Face')
    # XView
    FIXED_WING_AIRCRAFT = Name('Fixed wing aircraft')
    FIXED_WING_AIRCRAFT.SMALL_AIRCRAFT = Name('Small aircraft')
    FIXED_WING_AIRCRAFT.CARGO_PLANE = Name('Cargo plane')
    PASSENGER_VEHICLE = Name('Passenger vehicle')
    PASSENGER_VEHICLE.SMALL_CAR = Name('Small car')
    PASSENGER_VEHICLE.BUS = Name('Bus')
    TRUCK = Name('Truck')
    TRUCK.PICKUP_TRUCK = Name('Pickup truck')
    TRUCK.UTILITY_TRUCK = Name('Utility truck')
    TRUCK.CARGO_TRUCK = Name('Cargo truck')
    TRUCK.TRUCK_BOX = Name('Truck box')
    TRUCK.TRUCK_TRACTOR = Name('Truck tractor')
    TRUCK.TRAILER = Name('Trailer')
    TRUCK.TRUCK_FLATBED = Name('Truck flatbed')
    TRUCK.TRUCK_LIQUID = Name('Truck liquid')
    RAILWAY_VEHICLE = Name('Railway vehicle')
    RAILWAY_VEHICLE.PASSENGER_CAR = Name('Passenger car')
    RAILWAY_VEHICLE.CARGO_CAR = Name('Cargo car')
    RAILWAY_VEHICLE.FLAT_CAR = Name('Flat car')
    RAILWAY_VEHICLE.TANK_CAR = Name('Tank car')
    RAILWAY_VEHICLE.LOCOMOTIVE = Name('Locomotive')
    MARITIME_VESSEL = Name('Maritime vessel')
    MARITIME_VESSEL.MOTORBOAT = Name('Motorboat')
    MARITIME_VESSEL.SAILBOAT = Name('Sailboat')
    MARITIME_VESSEL.TUGBOAT = Name('Tugboat')
    MARITIME_VESSEL.BARGE = Name('Barge')
    MARITIME_VESSEL.FISHING_VESSEL = Name('Fishing vessel')
    MARITIME_VESSEL.FERRY = Name('Ferry')
    MARITIME_VESSEL.YATCH = Name('Yatch')
    MARITIME_VESSEL.CONTAINER_SHIP = Name('Container ship')
    MARITIME_VESSEL.OIL_TANKER = Name('Oil tanker')
    ENGINEERING_VEHICLE = Name('Engineering vehicle')
    ENGINEERING_VEHICLE.TOWER_CRANE = Name('Tower crane')
    ENGINEERING_VEHICLE.CONTAINER_CRANE = Name('Container crane')
    ENGINEERING_VEHICLE.REACH_STACKER = Name('Reach stacker')
    ENGINEERING_VEHICLE.STRADDLE_CARRIER = Name('Straddle carrier')
    ENGINEERING_VEHICLE.MOBILE_CRANE = Name('Mobile crane')
    ENGINEERING_VEHICLE.DUMP_TRUCK = Name('Dump truck')
    ENGINEERING_VEHICLE.HAUL_TRUCK = Name('Haul truck')
    ENGINEERING_VEHICLE.SCRAPER_TRACTOR = Name('Scrapper/tractor')
    ENGINEERING_VEHICLE.FRONT_LOADER = Name('Front loader')
    ENGINEERING_VEHICLE.EXCAVATOR = Name('Excavator')
    ENGINEERING_VEHICLE.CEMENT_MIXER = Name('Cement mixer')
    ENGINEERING_VEHICLE.GROUND_GRADER = Name('Ground grader')
    ENGINEERING_VEHICLE.CRANE_TRUCK = Name('Crane truck')
    BUILDING = Name('Building')
    BUILDING.HUT_TENT = Name('Hut/tent')
    BUILDING.SHED = Name('Shed')
    BUILDING.AIRCRAFT_HANGAR = Name('Aircraft hangar')
    BUILDING.DAMAGED_BUILDING = Name('Damaged building')
    BUILDING.FACILITY = Name('Facility')
    AIRPORT = Name('Airport')
    HELIPAD = Name('Helipad')
    PYLON = Name('Pylon')
    SHIPPING_CONTAINER = Name('Shipping container')
    SHIPPING_CONTAINER_LOT = Name('Shipping container lot')
    STORAGE_TANK = Name('Storage tank')
    VEHICLE_LOT = Name('Vehicle lot')
    CONSTRUCTION_SITE = Name('Construction site')
    TOWER_STRUCTURE = Name('Tower structure')
    HELICOPTER = Name('Helicopter')
    # XView2
    # BUILDING = Name('Building')
    # DOTA
    SHIP = Name('Ship')
    # STORAGE_TANK = Name('Storage tank')
    BASEBALL_DIAMOND = Name('Baseball diamond')
    TENNIS_COURT = Name('Tennis court')
    BASKETBALL_COURT = Name('Basketball court')
    GROUND_TRACK_FIELD = Name('Ground track field')
    BRIDGE = Name('Bridge')
    LARGE_VEHICLE = Name('Large vehicle')
    SMALL_VEHICLE = Name('Small vehicle')
    # HELICOPTER = Name('Helicopter')
    SWIMMING_POOL = Name('Swimming pool')
    ROUNDABOUT = Name('Roundabout')
    SOCCER_BALL_FIELD = Name('Soccer ball field')
    PLANE = Name('Plane')
    HARBOR = Name('Harbor')
    # ENGINEERING_VEHICLE.CONTAINER_CRANE = Name('Container crane')
    # PV Solar Panels
    SOLAR_PANEL = Name('Solar panel')
    # Stanford
    VEHICLE = Name('Vehicle')
    VEHICLE.CAR = Name('Car')
