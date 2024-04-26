# ==========================================================
# INFORMATIONS SUR CE PACKAGE :
# -----------------------------
# UTILITÉ DE SON CONTENU :
# Créer le dictionnaire biomes_dict
# -----------------------------
# CONTENU :
# - add_in_dict(dict, class)
# - biomes_dict_creation()
# ==========================================================
from typing import Dict

from pyprocgen import Biome, BoardColor, Color, Encyclopedia, Tree


###############################################################
######################### add_in_dict #########################
###############################################################


def add_in_dict(biomes_dict: Dict[str, Biome], biome: Biome) -> None:
    # =============================
    # INFORMATIONS :
    # -----------------------------
    # UTILITÉ :
    # Ajoute biome dans le dictionnaire biomes_dict avec
    # biome.name comme référence
    # =============================

    biomes_dict[biome.get_name()] = biome


###############################################################
################### ENCYCLOPEDIA_CREATION #####################
###############################################################
def encyclopedia_creation() -> Encyclopedia:
    # =============================
    # INFORMATIONS :
    # -----------------------------
    # UTILITÉ :
    # Remplie le dictionnaire de l'encyclopédie puis la crée
    # =============================

    dict_biomes = {}

    empty_tree = None

    ###############################################################
    ########################## DESERT #############################
    ###############################################################

    ########################### ARBRES ############################

    desert_tree_1 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="desert_cool",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=-4.0,
            pluviometry_max=-3.0,
            height_min=0.0,
            height_max=0.2,

            ground_color=Color(193, 165, 133),

            trees=[
                desert_tree_1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="desert_tropical",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=-4.0,
            pluviometry_max=-3.0,
            height_min=0.0,
            height_max=0.2,

            ground_color=Color(247, 210, 165),

            trees=[
                desert_tree_1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="desert_warm",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=-4.0,
            pluviometry_max=-3.0,
            height_min=0.0,
            height_max=0.2,

            ground_color=Color(207, 151, 100),

            trees=[
                desert_tree_1
            ]
        )
    )

    ###############################################################
    ######################## DESERT_SCUB ##########################
    ###############################################################

    ########################### ARBRES ############################
    desert_scub_tree_1 = None

    desert_scub_tree_2 = None
    

    desert_scub_bush_1 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="desert_scub_cool",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=-3.0,
            pluviometry_max=-2.0,
            height_min=0.1,
            height_max=0.3,

            ground_color=Color(187, 158, 126),

            trees=[
                desert_scub_tree_1,
                desert_scub_tree_2,
                desert_scub_bush_1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="desert_scub_tropical",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=-3.0,
            pluviometry_max=-2.0,
            height_min=0.1,
            height_max=0.3,

            ground_color=Color(251, 224, 181),

            trees=[
                desert_scub_tree_1,
                desert_scub_tree_2,
                desert_scub_bush_1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="desert_scub_warm",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=-3.0,
            pluviometry_max=-2.0,
            height_min=0.1,
            height_max=0.3,

            ground_color=Color(193, 161, 122),

            trees=[
                desert_scub_tree_1,
                desert_scub_tree_2,
                desert_scub_bush_1
            ]
        )
    )

    ###############################################################
    ######################### DRY_FOREST ##########################
    ###############################################################

    ########################### ARBRES ############################

    dry_forest_tree_1 = None

    dry_forest_tree_2 = None

    dry_forest_tree_3 = None

    dry_forest_bush_1 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="dry_forest_tropical",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=0.0,
            pluviometry_max=1.0,
            height_min=0.2,
            height_max=0.5,

            ground_color=Color(177, 148, 108),

            trees=[
                dry_forest_tree_1,
                dry_forest_tree_2,
                dry_forest_tree_3,
                dry_forest_bush_1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="dry_forest_warm",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=-1.0,
            pluviometry_max=0.0,
            height_min=0.2,
            height_max=0.5,

            ground_color=Color(167, 138, 104),

            trees=[
                dry_forest_tree_1,
                dry_forest_tree_2,
                dry_forest_tree_3,
                dry_forest_bush_1
            ]

        )
    )

    ###############################################################
    ######################## MOIST_FOREST #########################
    ###############################################################

    ########################### ARBRES ############################

    moist_forest_tree_1 = None

    moist_forest_tree_2 = None

    moist_forest_tree_3 = None

    moist_forest_tree_4 = None

    moist_forest_tree_5 = None

    moist_forest_tree_6 = None

    moist_forest_bush_1 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="moist_forest_cool",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=-1.0,
            pluviometry_max=0.0,
            height_min=0.5,
            height_max=1.0,

            ground_color=Color(78, 105, 36),

            trees=[
                moist_forest_tree_1,
                moist_forest_tree_2,
                moist_forest_tree_3,
                moist_forest_tree_4,
                moist_forest_tree_5,
                moist_forest_tree_6,
                moist_forest_bush_1
            ]

        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="moist_forest_tropical",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=1.0,
            pluviometry_max=2.0,
            height_min=0.5,
            height_max=1.0,

            ground_color=Color(93, 84, 51),

            trees=[
                moist_forest_tree_1,
                moist_forest_tree_2,
                moist_forest_tree_3,
                moist_forest_tree_4,
                moist_forest_tree_5,
                moist_forest_tree_6,
                moist_forest_bush_1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="moist_forest_warm",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=0.0,
            pluviometry_max=1.0,
            height_min=0.5,
            height_max=1.0,

            ground_color=Color(86, 104, 56),

            trees=[
                moist_forest_tree_1,
                moist_forest_tree_2,
                moist_forest_tree_3,
                moist_forest_tree_4,
                moist_forest_tree_5,
                moist_forest_tree_6,
                moist_forest_bush_1
            ]
        )
    )

    ###############################################################
    ######################## RAIN_FOREST ##########################
    ###############################################################

    ########################### ARBRES ############################

    rain_forest_tree_1 = None

    rain_forest_tree_2 = None

    rain_forest_tree_3 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="rain_forest",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=1.0,
            pluviometry_max=2.0,
            height_min=0.8,
            height_max=1.2,

            ground_color=Color(89, 93, 66),

            trees=[
                rain_forest_tree_1,
                rain_forest_tree_2,
                rain_forest_tree_3
            ]
        )
    )

    ###############################################################
    ####################### ROCKS_AND_ICE #########################
    ###############################################################

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="rocks_and_ice",

            temperature_min=-3.0,
            temperature_max=-2.0,
            pluviometry_min=-4.0,
            pluviometry_max=-1.0,
            height_min=0.5,
            height_max=1.5,

            ground_color=Color(190, 220, 255),

            trees=[
                empty_tree
            ]
        )
    )

    ###############################################################
    ########################### STEPPE ############################
    ###############################################################

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="steppe",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=-2.0,
            pluviometry_max=-1.0,
            height_min=0.1,
            height_max=0.3,

            ground_color=Color(160, 173, 120),

            trees=[
                empty_tree
            ]
        )
    )

    ###############################################################
    #################### STEPPE_WOODLAND_THORN ####################
    ###############################################################

    ########################### ARBRES ############################

    steppe_woodland_thorn_tree_1 = None

    steppe_woodland_thorn_tree_2 = None

    steppe_woodland_thorn_tree_3 = None

    steppe_woodland_thorn_tree_4 = None
    
    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="steppe_woodland_thorn",
            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=-2.0,
            pluviometry_max=-1.0,
            height_min=0.2,
            height_max=0.4,

            ground_color=Color(160, 173, 120),

            trees=[
                steppe_woodland_thorn_tree_1,
                steppe_woodland_thorn_tree_2,
                steppe_woodland_thorn_tree_3,
                steppe_woodland_thorn_tree_4
            ]
        )
    )

    ###############################################################
    ########################### TAIGA #############################
    ###############################################################

    ########################### ARBRES ############################

    taiga_tree_1 = None

    taiga_tree_2 = None

    taiga_tree_3 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="taiga_desert",

            temperature_min=-1.0,
            temperature_max=0.0,
            pluviometry_min=-4.0,
            pluviometry_max=-3.0,
            height_min=0.3,
            height_max=0.6,

            ground_color=Color(146, 126, 101),

            trees=[
                taiga_tree_1,
                taiga_tree_2,
                taiga_tree_3
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="taiga_dry",
            temperature_min=-1.0,
            temperature_max=0.0,
            pluviometry_min=-3.0,
            pluviometry_max=-2.0,
            height_min=0.3,
            height_max=0.6,

            ground_color=Color(167, 175, 120),

            trees=[
                taiga_tree_1,
                taiga_tree_2,
                taiga_tree_3
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="taiga_moist",

            temperature_min=-1.0,
            temperature_max=0.0,
            pluviometry_min=-2.0,
            pluviometry_max=-1.0,
            height_min=0.3,
            height_max=0.6,

            ground_color=Color(86, 104, 56),

            trees=[
                taiga_tree_1,
                taiga_tree_2,
                taiga_tree_3
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="taiga_rain",

            temperature_min=-1.0,
            temperature_max=0.0,
            pluviometry_min=0.0,
            pluviometry_max=1.0,
            height_min=0.3,
            height_max=0.6,

            ground_color=Color(57, 102, 21),

            trees=[
                taiga_tree_1,
                taiga_tree_2,
                taiga_tree_3
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="taiga_wet",

            temperature_min=-1.0,
            temperature_max=0.0,
            pluviometry_min=-1.0,
            pluviometry_max=0.0,
            height_min=0.3,
            height_max=0.6,

            ground_color=Color(75, 102, 44),

            trees=[
                taiga_tree_1,
                taiga_tree_2,
                taiga_tree_3
            ]
        )
    )

    ###############################################################
    ########################## TUNDRA #############################
    ###############################################################

    ########################### ARBRES ############################

    tundra_bush_1 = None

    tundra_bush_2 = None

    tundra_bush_3 = None

    tundra_bush_4 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="tundra_dry",

            temperature_min=-2.0,
            temperature_max=-1.0,
            pluviometry_min=-4.0,
            pluviometry_max=-3.0,
            height_min=0.1,
            height_max=0.2,

            ground_color=Color(167, 175, 120),

            trees=[
                tundra_bush_1,
                tundra_bush_2,
                tundra_bush_3,
                tundra_bush_4
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="tundra_moist",

            temperature_min=-2.0,
            temperature_max=-1.0,
            pluviometry_min=-3.0,
            pluviometry_max=-2.0,
            height_min=0.1,
            height_max=0.2,

            ground_color=Color(167, 175, 120),

            trees=[
                tundra_bush_1,
                tundra_bush_2,
                tundra_bush_3,
                tundra_bush_4
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="tundra_rain",

            temperature_min=-2.0,
            temperature_max=-1.0,
            pluviometry_min=-1.0,
            pluviometry_max=0.0,
            height_min=0.1,
            height_max=0.2,

            ground_color=Color(167, 175, 120),

            trees=[
                tundra_bush_1,
                tundra_bush_2,
                tundra_bush_3,
                tundra_bush_4
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="tundra_wet",

            temperature_min=-2.0,
            temperature_max=-1.0,
            pluviometry_min=-2.0,
            pluviometry_max=-1.0,
            height_min=0.1,
            height_max=0.2,

            ground_color=Color(75, 102, 44),

            trees=[
                tundra_bush_1,
                tundra_bush_2,
                tundra_bush_3,
                tundra_bush_4
            ]
        )
    )

    ###############################################################
    ###################### TROPICAL_FOREST ########################
    ###############################################################

    ########################### ARBRES ############################

    tropical_forest_tree_1 = None

    tropical_forest_tree_2 = None

    tropical_forest_tree_3 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="tropical_forest_tropical",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=3.0,
            pluviometry_max=4.0,
            height_min=1.0,
            height_max=1.5,

            ground_color=Color(71, 94, 12),

            trees=[
                tropical_forest_tree_1,
                tropical_forest_tree_2,
                tropical_forest_tree_3
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="tropical_forest_warm",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=2.0,
            pluviometry_max=3.0,
            height_min=1.0,
            height_max=1.5,

            ground_color=Color(94, 124, 16),

            trees=[
                tropical_forest_tree_1,
                tropical_forest_tree_2,
                tropical_forest_tree_3
            ]
        )
    )

    ###############################################################
    ###################### VERY_DRY_FOREST ########################
    ###############################################################

    ########################### ARBRES ############################

    very_dry_forest_tree_1 = dry_forest_tree_1

    very_dry_forest_tree_2 = None

    very_dry_forest_tree_3 = dry_forest_tree_3

    very_dry_forest_bush_1 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="very_dry_forest",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=-1.0,
            pluviometry_max=0.0,
            height_min=0.2,
            height_max=0.4,

            ground_color=Color(191, 168, 124),

            trees=[
                very_dry_forest_tree_1,
                very_dry_forest_tree_2,
                very_dry_forest_tree_3,
                very_dry_forest_bush_1
            ]
        )
    )

    ###############################################################
    ######################### WET_FOREST ##########################
    ###############################################################

    ########################### ARBRES ############################

    wet_forest_tree_1_v1 = None

    wet_forest_tree_1_v2 = None

    wet_forest_tree_1_v3 = None

    wet_forest_tree_2_v1 = None
    wet_forest_tree_2_v2 = None

    wet_forest_tree_2_v3 = None

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="wet_forest_cool",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=0.0,
            pluviometry_max=1.0,
            height_min=0.6,
            height_max=1.0,

            ground_color=Color(128, 168, 104),

            trees=[
                wet_forest_tree_1_v1,
                wet_forest_tree_2_v1
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="wet_forest_tropical",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=2.0,
            pluviometry_max=3.0,
            height_min=0.6,
            height_max=1.0,

            ground_color=Color(128, 168, 104),

            trees=[
                wet_forest_tree_1_v2,
                wet_forest_tree_2_v2
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="wet_forest_warm",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=1.0,
            pluviometry_max=2.0,
            height_min=0.6,
            height_max=1.0,

            ground_color=Color(128, 168, 104),

            trees=[
                wet_forest_tree_1_v3,
                wet_forest_tree_2_v3,
            ]
        )
    )

    ###############################################################
    ####################### WOODLAND_THORN ########################
    ###############################################################

    ########################### ARBRES ############################

    woodland_thorn_tree_1 = None

    woodland_thorn_tree_2 = None

    woodland_thorn_tree_3 = None
    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="woodland_thorn",

            temperature_min=2.0,
            temperature_max=3.0,
            pluviometry_min=-2.0,
            pluviometry_max=-1.0,
            height_min=0.2,
            height_max=0.4,

            ground_color=Color(149, 163, 140),

            trees=[
                woodland_thorn_tree_1,
                woodland_thorn_tree_2,
                woodland_thorn_tree_3
            ]
        )
    )
    
    # Add mountain biome
    add_in_dict(
        dict_biomes,
        Biome(
            name="mountain",
            temperature_min=-2.0, # -2.0
            temperature_max=3.0, # 0.0
            pluviometry_min=-3.0, # -2.0
            pluviometry_max=3.0, # 1.0
            ground_color=Color(50, 50, 50),
            trees=[empty_tree],
            height_min=1.5,
            height_max=3.0
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="objective",
            temperature_min=0.0,
            temperature_max=0.0,
            pluviometry_min=0.0,
            pluviometry_max=0.0,
            ground_color=Color(255, 0, 0),
            trees=[empty_tree],
            height_min=0.0,
            height_max=0.0
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="clue",
            temperature_min=0.0,
            temperature_max=0.0,
            pluviometry_min=0.0,
            pluviometry_max=0.0,
            ground_color=Color(255, 165, 0),
            trees=[empty_tree],
            height_min=0.0,
            height_max=0.0
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="home",
            temperature_min=0.0,
            temperature_max=0.0,
            pluviometry_min=0.0,
            pluviometry_max=0.0,
            ground_color=Color(160, 32, 240),
            trees=[empty_tree],
            height_min=0.0,
            height_max=0.0
        )
    )


    ###############################################################
    ############################ WATER ############################
    ###############################################################

    ########################### BIOMES ############################

    add_in_dict(
        dict_biomes,
        Biome(
            name="water",

            temperature_min=0.0,
            temperature_max=0.0,
            pluviometry_min=0.0,
            pluviometry_max=0.0,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(30, 144, 235),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_1",

            temperature_min=1.85,
            temperature_max=2.0,
            pluviometry_min=3.0,
            pluviometry_max=4.0,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_2",

            temperature_min=0.85,
            temperature_max=1.0,
            pluviometry_min=2.0,
            pluviometry_max=3.0,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_3",

            temperature_min=-0.25,
            temperature_max=0.0,
            pluviometry_min=1.0,
            pluviometry_max=2.0,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_4",

            temperature_min=-1.25,
            temperature_max=-1.0,
            pluviometry_min=0.0,
            pluviometry_max=1.0,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_Water_5",

            temperature_min=-2.25,
            temperature_max=-2.0,
            pluviometry_min=-1.0,
            pluviometry_max=0.0,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_6",

            temperature_min=1.0,
            temperature_max=2.0,
            pluviometry_min=3.0,
            pluviometry_max=3.15,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_7",

            temperature_min=0.0,
            temperature_max=1.0,
            pluviometry_min=2.0,
            pluviometry_max=2.15,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_8",

            temperature_min=-1.0,
            temperature_max=0.0,
            pluviometry_min=1.0,
            pluviometry_max=1.15,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_9",

            temperature_min=-2.0,
            temperature_max=-1.0,
            pluviometry_min=0.0,
            pluviometry_max=0.15,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    add_in_dict(
        dict_biomes,
        Biome(
            name="cyan_water_10",

            temperature_min=-3.0,
            temperature_max=-2.0,
            pluviometry_min=-1.0,
            pluviometry_max=-0.85,
            height_min=0.0,
            height_max=0.0,

            ground_color=Color(64, 164, 223),

            trees=[
                empty_tree
            ]
        )
    )

    return Encyclopedia("Classique", dict_biomes)
