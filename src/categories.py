#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'


class Name:
    def __init__(self, name):
        self.name = name


class Category:
    """
    Category label.
    """
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
    BUILDING.UNCLASSIFIED = Name('Unclassified')
    BUILDING.NO_DAMAGE = Name('No damage')
    BUILDING.MINOR_DAMAGE = Name('Minor damage')
    BUILDING.MAJOR_DAMAGE = Name('Major damage')
    BUILDING.DESTROYED = Name('Destroyed')
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
    SOLAR_PANEL.UNCLASSIFIED = Name('Unclassified')
    SOLAR_PANEL.NO_DAMAGE = Name('No damage')
    SOLAR_PANEL.HOT_CELL = Name('Hot cell')
    SOLAR_PANEL.HOT_CELL_CHAIN = Name('Hot cell chain')
    SOLAR_PANEL.SEVERAL_HOT_CELLS = Name('Several hot cells')
    SOLAR_PANEL.HOT_SPOT = Name('Hot spot')
    SOLAR_PANEL.SEVERAL_HOT_SPOTS = Name('Several hot spots')
    SOLAR_PANEL.POTENTIAL_INDUCED_DEGRADATION = Name('Potential induced degradation')
    SOLAR_PANEL.DIRTY_PANEL = Name('Dirty panel')
    SOLAR_PANEL.BROKEN_PANEL = Name('Broken panel')
    SOLAR_PANEL.DISCONNECTED_PANEL = Name('Disconnected panel')
    SOLAR_PANEL.SHADES = Name('Shades')
    SOLAR_PANEL.SHADES_HOT_CELL_CHAIN = Name('Shades + Hot cell chain')
    SOLAR_PANEL.MELTED_FUSES = Name('Melted fuses')
    # Stanford
    CAR = Name('Car')
    CAR.AMGENERAL_HUMMER_SUV_2000 = Name('AM General Hummer SUV 2000')
    CAR.ACURA_RL_SEDAN_2012 = Name('Acura RL Sedan 2012')
    CAR.ACURA_TL_SEDAN_2012 = Name('Acura TL Sedan 2012')
    CAR.ACURA_TL_TYPES_2008 = Name('Acura TL Type-S 2008')
    CAR.ACURA_TSX_SEDAN_2012 = Name('Acura TSX Sedan 2012')
    CAR.ACURA_INTEGRA_TYPER_2001 = Name('Acura Integra Type R 2001')
    CAR.ACURA_ZDX_HATCHBACK_2012 = Name('Acura ZDX Hatchback 2012')
    CAR.ASTONMARTIN_V8VANTAGE_CONVERTIBLE_2012 = Name('Aston Martin V8 Vantage Convertible 2012')
    CAR.ASTONMARTIN_V8VANTAGE_COUPE_2012 = Name('Aston Martin V8 Vantage Coupe 2012')
    CAR.ASTONMARTIN_VIRAGE_CONVERTIBLE_2012 = Name('Aston Martin Virage Convertible 2012')
    CAR.ASTONMARTIN_VIRAGE_COUPE_2012 = Name('Aston Martin Virage Coupe 2012')
    CAR.AUDI_RS4_CONVERTIBLE_2008 = Name('Audi RS 4 Convertible 2008')
    CAR.AUDI_A5_COUPE_2012 = Name('Audi A5 Coupe 2012')
    CAR.AUDI_TTS_COUPE_2012 = Name('Audi TTS Coupe 2012')
    CAR.AUDI_R8_COUPE_2012 = Name('Audi R8 Coupe 2012')
    CAR.AUDI_V8_SEDAN_1994 = Name('Audi V8 Sedan 1994')
    CAR.AUDI_100_SEDAN_1994 = Name('Audi 100 Sedan 1994')
    CAR.AUDI_100_WAGON_1994 = Name('Audi 100 Wagon 1994')
    CAR.AUDI_TT_HATCHBACK_2011 = Name('Audi TT Hatchback 2011')
    CAR.AUDI_S6_SEDAN_2011 = Name('Audi S6 Sedan 2011')
    CAR.AUDI_S5_CONVERTIBLE_2012 = Name('Audi S5 Convertible 2012')
    CAR.AUDI_S5_COUPE_2012 = Name('Audi S5 Coupe 2012')
    CAR.AUDI_S4_SEDAN_2012 = Name('Audi S4 Sedan 2012')
    CAR.AUDI_S4_SEDAN_2007 = Name('Audi S4 Sedan 2007')
    CAR.AUDI_TTRS_COUPE_2012 = Name('Audi TT RS Coupe 2012')
    CAR.BMW_ACTIVEHYBRID5_SEDAN_2012 = Name('BMW ActiveHybrid 5 Sedan 2012')
    CAR.BMW_SERIES1_CONVERTIBLE_2012 = Name('BMW 1 Series Convertible 2012')
    CAR.BMW_SERIES1_COUPE_2012 = Name('BMW 1 Series Coupe 2012')
    CAR.BMW_SERIES3_SEDAN_2012 = Name('BMW 3 Series Sedan 2012')
    CAR.BMW_SERIES3_WAGON_2012 = Name('BMW 3 Series Wagon 2012')
    CAR.BMW_SERIES6_CONVERTIBLE_2007 = Name('BMW 6 Series Convertible 2007')
    CAR.BMW_X5_SUV_2007 = Name('BMW X5 SUV 2007')
    CAR.BMW_X6_SUV_2012 = Name('BMW X6 SUV 2012')
    CAR.BMW_M3_COUPE_2012 = Name('BMW M3 Coupe 2012')
    CAR.BMW_M5_SEDAN_2010 = Name('BMW M5 Sedan 2010')
    CAR.BMW_M6_CONVERTIBLE_2010 = Name('BMW M6 Convertible 2010')
    CAR.BMW_X3_SUV_2012 = Name('BMW X3 SUV 2012')
    CAR.BMW_Z4_CONVERTIBLE_2012 = Name('BMW Z4 Convertible 2012')
    CAR.BENTLEY_CONTINENTAL_CONVERTIBLE_2012 = Name('Bentley Continental Supersports Conv. Convertible 2012')
    CAR.BENTLEY_ARNAGE_SEDAN_2009 = Name('Bentley Arnage Sedan 2009')
    CAR.BENTLEY_MULSANNE_SEDAN_2011 = Name('Bentley Mulsanne Sedan 2011')
    CAR.BENTLEY_CONTINENTAL_COUPE_2012 = Name('Bentley Continental GT Coupe 2012')
    CAR.BENTLEY_CONTINENTAL_COUPE_2007 = Name('Bentley Continental GT Coupe 2007')
    CAR.BENTLEY_CONTINENTAL_SEDAN_2007 = Name('Bentley Continental Flying Spur Sedan 2007')
    CAR.BUGATTI_VEYRON_CONVERTIBLE_2009 = Name('Bugatti Veyron 16.4 Convertible 2009')
    CAR.BUGATTI_VEYRON_COUPE_2009 = Name('Bugatti Veyron 16.4 Coupe 2009')
    CAR.BUICK_REGAL_GS_2012 = Name('Buick Regal GS 2012')
    CAR.BUICK_RAINIER_SUV_2007 = Name('Buick Rainier SUV 2007')
    CAR.BUICK_VERANO_SEDAN_2012 = Name('Buick Verano Sedan 2012')
    CAR.BUICK_ENCLAVE_SUV_2012 = Name('Buick Enclave SUV 2012')
    CAR.CADILLAC_CTSV_SEDAN_2012 = Name('Cadillac CTS-V Sedan 2012')
    CAR.CADILLAC_SRX_SUV_2012 = Name('Cadillac SRX SUV 2012')
    CAR.CADILLAC_ESCALADE_CREWCAB_2007 = Name('Cadillac Escalade EXT Crew Cab 2007')
    CAR.CHEVROLET_SILVERADO1500HYBRID_CREWCAB_2012 = Name('Chevrolet Silverado 1500 Hybrid Crew Cab 2012')
    CAR.CHEVROLET_CORVETTE_CONVERTIBLE_2012 = Name('Chevrolet Corvette Convertible 2012')
    CAR.CHEVROLET_CORVETTE_ZR1_2012 = Name('Chevrolet Corvette ZR1 2012')
    CAR.CHEVROLET_CORVETTE_Z06_2007 = Name('Chevrolet Corvette Ron Fellows Edition Z06 2007')
    CAR.CHEVROLET_TRAVERSE_SUV_2012 = Name('Chevrolet Traverse SUV 2012')
    CAR.CHEVROLET_CAMARO_CONVERTIBLE_2012 = Name('Chevrolet Camaro Convertible 2012')
    CAR.CHEVROLET_HHR_SS_2010 = Name('Chevrolet HHR SS 2010')
    CAR.CHEVROLET_IMPALA_SEDAN_2007 = Name('Chevrolet Impala Sedan 2007')
    CAR.CHEVROLET_TAHOEHYBRID_SUV_2012 = Name('Chevrolet Tahoe Hybrid SUV 2012')
    CAR.CHEVROLET_SONIC_SEDAN_2012 = Name('Chevrolet Sonic Sedan 2012')
    CAR.CHEVROLET_EXPRESS_CARGOVAN_2007 = Name('Chevrolet Express Cargo Van 2007')
    CAR.CHEVROLET_AVALANCHE_CREWCAB_2012 = Name('Chevrolet Avalanche Crew Cab 2012')
    CAR.CHEVROLET_COBALT_SS_2010 = Name('Chevrolet Cobalt SS 2010')
    CAR.CHEVROLET_MALIBUHYBRID_SEDAN_2010 = Name('Chevrolet Malibu Hybrid Sedan 2010')
    CAR.CHEVROLET_TRAINBLAZER_SS_2009 = Name('Chevrolet TrailBlazer SS 2009')
    CAR.CHEVROLET_SILVERADO2500HD_REGULARCAB_2012 = Name('Chevrolet Silverado 2500HD Regular Cab 2012')
    CAR.CHEVROLET_SILVERADO1500CLASSIC_EXTENDEDCAB_2007 = Name('Chevrolet Silverado 1500 Classic Extended Cab 2007')
    CAR.CHEVROLET_EXPRESS_VAN_2007 = Name('Chevrolet Express Van 2007')
    CAR.CHEVROLET_MONTECARLO_COUPE_2007 = Name('Chevrolet Monte Carlo Coupe 2007')
    CAR.CHEVROLET_MALIBU_SEDAN_2007 = Name('Chevrolet Malibu Sedan 2007')
    CAR.CHEVROLET_SILVERADO1500_EXTENDEDCAB_2012 = Name('Chevrolet Silverado 1500 Extended Cab 2012')
    CAR.CHEVROLET_SILVERADO1500_REGULARCAB_2012 = Name('Chevrolet Silverado 1500 Regular Cab 2012')
    CAR.CHRYSLER_ASPEN_SUV_2009 = Name('Chrysler Aspen SUV 2009')
    CAR.CHRYSLER_SEBRING_CONVERTIBLE_2010 = Name('Chrysler Sebring Convertible 2010')
    CAR.CHRYSLER_TOWN_MINIVAN_2012 = Name('Chrysler Town and Country Minivan 2012')
    CAR.CHRYSLER_300_STR8_2010 = Name('Chrysler 300 SRT-8 2010')
    CAR.CHRYSLER_CROSSFIRE_CONVERTIBLE_2008 = Name('Chrysler Crossfire Convertible 2008')
    CAR.CHRYSLER_PTCRUISER_CONVERTIBLE_2008 = Name('Chrysler PT Cruiser Convertible 2008')
    CAR.DAEWOO_NUBIRA_WAGON_2002 = Name('Daewoo Nubira Wagon 2002')
    CAR.DODGE_CALIBER_WAGON_2012 = Name('Dodge Caliber Wagon 2012')
    CAR.DODGE_CALIBER_WAGON_2007 = Name('Dodge Caliber Wagon 2007')
    CAR.DODGE_CARAVAN_MINIVAN_1997 = Name('Dodge Caravan Minivan 1997')
    CAR.DODGE_RAM_CREWCAB_2010 = Name('Dodge Ram Pickup 3500 Crew Cab 2010')
    CAR.DODGE_RAM_QUADCAB_2009 = Name('Dodge Ram Pickup 3500 Quad Cab 2009')
    CAR.DODGE_SPRINTER_CARGOVAN_2009 = Name('Dodge Sprinter Cargo Van 2009')
    CAR.DODGE_JOURNEY_SUV_2012 = Name('Dodge Journey SUV 2012')
    CAR.DODGE_DAKOTA_CREWCAB_2010 = Name('Dodge Dakota Crew Cab 2010')
    CAR.DODGE_DAKOTA_CLUBCAB_2007 = Name('Dodge Dakota Club Cab 2007')
    CAR.DODGE_MAGNUM_WAGON_2008 = Name('Dodge Magnum Wagon 2008')
    CAR.DODGE_CHALLENGER_SRT8_2011 = Name('Dodge Challenger SRT8 2011')
    CAR.DODGE_DURANGO_SUV_2012 = Name('Dodge Durango SUV 2012')
    CAR.DODGE_DURANGO_SUV_2007 = Name('Dodge Durango SUV 2007')
    CAR.DODGE_CHARGER_SEDAN_2012 = Name('Dodge Charger Sedan 2012')
    CAR.DODGE_CHARGER_SRT8_2009 = Name('Dodge Charger SRT-8 2009')
    CAR.EAGLE_TALON_HATCHBACK_1998 = Name('Eagle Talon Hatchback 1998')
    CAR.FIAT_500_ABARTH_2012 = Name('FIAT 500 Abarth 2012')
    CAR.FIAT_500_CONVERTIBLE_2012 = Name('FIAT 500 Convertible 2012')
    CAR.FERRARI_FF_COUPE_2012 = Name('Ferrari FF Coupe 2012')
    CAR.FERRARI_CALIFORNIA_CONVERTIBLE_2012 = Name('Ferrari California Convertible 2012')
    CAR.FERRARI_ITALIA_CONVERTIBLE_2012 = Name('Ferrari 458 Italia Convertible 2012')
    CAR.FERRARI_ITALIA_COUPE_2012 = Name('Ferrari 458 Italia Coupe 2012')
    CAR.FISKER_KARMA_SEDAN_2012 = Name('Fisker Karma Sedan 2012')
    CAR.FORD_F450_CREWCAB_2012 = Name('Ford F-450 Super Duty Crew Cab 2012')
    CAR.FORD_MUSTANG_CONVERTIBLE_2007 = Name('Ford Mustang Convertible 2007')
    CAR.FORD_FREESTAR_MINIVAN_2007 = Name('Ford Freestar Minivan 2007')
    CAR.FORD_EXPEDITION_SUV_2009 = Name('Ford Expedition EL SUV 2009')
    CAR.FORD_EDGE_SUV_2012 = Name('Ford Edge SUV 2012')
    CAR.FORD_RANGER_SUPERCAB_2011 = Name('Ford Ranger SuperCab 2011')
    CAR.FORD_GT_COUPE_2006 = Name('Ford GT Coupe 2006')
    CAR.FORD_F150_REGULARCAB_2012 = Name('Ford F-150 Regular Cab 2012')
    CAR.FORD_F150_REGULARCAB_2007 = Name('Ford F-150 Regular Cab 2007')
    CAR.FORD_FOCUS_SEDAN_2007 = Name('Ford Focus Sedan 2007')
    CAR.FORD_ESERIES_WAGON_2012 = Name('Ford E-Series Wagon Van 2012')
    CAR.FORD_FIESTA_SEDAN_2012 = Name('Ford Fiesta Sedan 2012')
    CAR.GMC_TERRAIN_SUV_2012 = Name('GMC Terrain SUV 2012')
    CAR.GMC_SAVANA_VAN_2012 = Name('GMC Savana Van 2012')
    CAR.GMC_YUKONHYBRID_SUV_2012 = Name('GMC Yukon Hybrid SUV 2012')
    CAR.GMC_ACADIA_SUV_2012 = Name('GMC Acadia SUV 2012')
    CAR.GMC_CANYON_EXTENDEDCAB_2012 = Name('GMC Canyon Extended Cab 2012')
    CAR.GMC_METRO_CONVERTIBLE_1993 = Name('Geo Metro Convertible 1993')
    CAR.HUMMER_H3T_CREWCAB_2010 = Name('HUMMER H3T Crew Cab 2010')
    CAR.HUMMER_H2SUT_CREWCAB_2009 = Name('HUMMER H2 SUT Crew Cab 2009')
    CAR.HONDA_ODYSSEY_MINIVAN_2012 = Name('Honda Odyssey Minivan 2012')
    CAR.HONDA_ODYSSEY_MINIVAN_2007 = Name('Honda Odyssey Minivan 2007')
    CAR.HONDA_ACCORD_COUPE_2012 = Name('Honda Accord Coupe 2012')
    CAR.HONDA_ACCORD_SEDAN_2012 = Name('Honda Accord Sedan 2012')
    CAR.HYUNDAI_VELOSTER_HATCHBACK_2012 = Name('Hyundai Veloster Hatchback 2012')
    CAR.HYUNDAI_SANTAFE_SUV_2012 = Name('Hyundai Santa Fe SUV 2012')
    CAR.HYUNDAI_TUCSON_SUV_2012 = Name('Hyundai Tucson SUV 2012')
    CAR.HYUNDAI_VERACRUZ_SUV_2012 = Name('Hyundai Veracruz SUV 2012')
    CAR.HYUNDAI_SONATAHYBRID_SEDAN_2012 = Name('Hyundai Sonata Hybrid Sedan 2012')
    CAR.HYUNDAI_ELANTRA_SEDAN_2007 = Name('Hyundai Elantra Sedan 2007')
    CAR.HYUNDAI_ACCENT_SEDAN_2012 = Name('Hyundai Accent Sedan 2012')
    CAR.HYUNDAI_GENESIS_SEDAN_2012 = Name('Hyundai Genesis Sedan 2012')
    CAR.HYUNDAI_SONATA_SEDAN_2012 = Name('Hyundai Sonata Sedan 2012')
    CAR.HYUNDAI_ELANTRA_HATCHBACK_2012 = Name('Hyundai Elantra Touring Hatchback 2012')
    CAR.HYUNDAI_AZERA_SEDAN_2012 = Name('Hyundai Azera Sedan 2012')
    CAR.INFINITI_G_COUPE_2012 = Name('Infiniti G Coupe IPL 2012')
    CAR.INFINITI_QX56_SUV_2011 = Name('Infiniti QX56 SUV 2011')
    CAR.ISUZU_ASCENDER_SUV_2008 = Name('Isuzu Ascender SUV 2008')
    CAR.JAGUAR_XK_XKR_2012 = Name('Jaguar XK XKR 2012')
    CAR.JEEP_PATRIOT_SUV_2012 = Name('Jeep Patriot SUV 2012')
    CAR.JEEP_WRANGLER_SUV_2012 = Name('Jeep Wrangler SUV 2012')
    CAR.JEEP_LIBERTY_SUV_2012 = Name('Jeep Liberty SUV 2012')
    CAR.JEEP_GRANDCHEROKEE_SUV_2012 = Name('Jeep Grand Cherokee SUV 2012')
    CAR.JEEP_COMPASS_SUV_2012 = Name('Jeep Compass SUV 2012')
    CAR.LAMBORGHINI_REVENTON_COUPE_2012 = Name('Lamborghini Reventon Coupe 2008')
    CAR.LAMBORGHINI_AVENTADOR_COUPE_2012 = Name('Lamborghini Aventador Coupe 2012')
    CAR.LAMBORGHINI_GALLARDO_SUPERLEGGERA_2012 = Name('Lamborghini Gallardo LP 570-4 Superleggera 2012')
    CAR.LAMBORGHINI_DIABLO_COUPE_2001 = Name('Lamborghini Diablo Coupe 2001')
    CAR.LANDROVER_RANGEROVER_SUV_2012 = Name('Land Rover Range Rover SUV 2012')
    CAR.LANDROVER_LR2_SUV_2012 = Name('Land Rover LR2 SUV 2012')
    CAR.LINCOLN_TOWN_SEDAN_2011 = Name('Lincoln Town Car Sedan 2011')
    CAR.MINI_COOPER_CONVERTIBLE_2012 = Name('MINI Cooper Roadster Convertible 2012')
    CAR.MAYBACH_LANDAULET_CONVERTIBLE_2012 = Name('Maybach Landaulet Convertible 2012')
    CAR.MAZDA_TRIBUTE_SUV_2011 = Name('Mazda Tribute SUV 2011')
    CAR.MCLAREN_MP4_COUPE_2012 = Name('McLaren MP4-12C Coupe 2012')
    CAR.MERCEDES_300_CONVERTIBLE_1993 = Name('Mercedes-Benz 300-Class Convertible 1993')
    CAR.MERCEDES_C_SEDAN_2012 = Name('Mercedes-Benz C-Class Sedan 2012')
    CAR.MERCEDES_SL_COUPE_2009 = Name('Mercedes-Benz SL-Class Coupe 2009')
    CAR.MERCEDES_E_SEDAN_2012 = Name('Mercedes-Benz E-Class Sedan 2012')
    CAR.MERCEDES_S_SEDAN_2012 = Name('Mercedes-Benz S-Class Sedan 2012')
    CAR.MERCEDES_SPRINTER_VAN_2012 = Name('Mercedes-Benz Sprinter Van 2012')
    CAR.MITSUBISHI_LANCER_SEDAN_2012 = Name('Mitsubishi Lancer Sedan 2012')
    CAR.NISSAN_LEAF_HATCHBACK_2012 = Name('Nissan Leaf Hatchback 2012')
    CAR.NISSAN_NV_VAN_2012 = Name('Nissan NV Passenger Van 2012')
    CAR.NISSAN_JUKE_HATCHBACK_2012 = Name('Nissan Juke Hatchback 2012')
    CAR.NISSAN_240SX_COUPE_1998 = Name('Nissan 240SX Coupe 1998')
    CAR.PLYMOUTH_NEON_COUPE_1999 = Name('Plymouth Neon Coupe 1999')
    CAR.PORSCHE_PANAMERA_SEDAN_2012 = Name('Porsche Panamera Sedan 2012')
    CAR.RAM_CV_MINIVAN_2012 = Name('Ram C/V Cargo Van Minivan 2012')
    CAR.ROLLSROYCE_PHANTOM_COUPE_2012 = Name('Rolls-Royce Phantom Drophead Coupe Convertible 2012')
    CAR.ROLLSROYCE_GHOST_SEDAN_2012 = Name('Rolls-Royce Ghost Sedan 2012')
    CAR.ROLLSROYCE_PHANTOM_SEDAN_2012 = Name('Rolls-Royce Phantom Sedan 2012')
    CAR.SCION_XD_HATCHBACK_2012 = Name('Scion xD Hatchback 2012')
    CAR.SPYKER_C8_CONVERTIBLE_2009 = Name('Spyker C8 Convertible 2009')
    CAR.SPYKER_C8_COUPE_2009 = Name('Spyker C8 Coupe 2009')
    CAR.SUZUKI_AERIO_SEDAN_2007 = Name('Suzuki Aerio Sedan 2007')
    CAR.SUZUKI_KIZASHI_SEDAN_2012 = Name('Suzuki Kizashi Sedan 2012')
    CAR.SUZUKI_SX4_HATCHBACK_2012 = Name('Suzuki SX4 Hatchback 2012')
    CAR.SUZUKI_SX4_SEDAN_2012 = Name('Suzuki SX4 Sedan 2012')
    CAR.TESLA_MODELS_SEDAN_2012 = Name('Tesla Model S Sedan 2012')
    CAR.TOYOTA_SEQUOIA_SUV_2012 = Name('Toyota Sequoia SUV 2012')
    CAR.TOYOTA_CAMRY_SEDAN_2012 = Name('Toyota Camry Sedan 2012')
    CAR.TOYOTA_COROLLA_SEDAN_2012 = Name('Toyota Corolla Sedan 2012')
    CAR.TOYOTA_4RUNNER_SUV_2012 = Name('Toyota 4Runner SUV 2012')
    CAR.VOLKSWAGEN_GOLF_HATCHBACK_2012 = Name('Volkswagen Golf Hatchback 2012')
    CAR.VOLKSWAGEN_GOLF_HATCHBACK_1991 = Name('Volkswagen Golf Hatchback 1991')
    CAR.VOLKSWAGEN_BEETLE_HATCHBACK_2012 = Name('Volkswagen Beetle Hatchback 2012')
    CAR.VOLV0_C30_HATCHBACK_2012 = Name('Volvo C30 Hatchback 2012')
    CAR.VOLV0_240_SEDAN_1993 = Name('Volvo 240 Sedan 1993')
    CAR.VOLVO_XC90_SUV_2007 = Name('Volvo XC90 SUV 2007')
    CAR.SMART_FORTWO_CONVERTIBLE_2012 = Name('Smart fortwo Convertible 2012')
