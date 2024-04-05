# ==========================================================
# INFORMATIONS SUR CE PACKAGE :
# -----------------------------
# UTILITÉ DE SON CONTENU :
# Définir la classe Biome, qui sert à avoir les infos sur un biome
# -----------------------------
# CONTENU :
# + __slots__
# + HINTS
# + __init__()
# + GETTERS
# + SETTERS
# + in_range()
# + __str__()
# ==========================================================

from typing import List

from pyprocgen.classes.Color import Color
from pyprocgen.classes.Tree import Tree

from pyprocgen.settings import (
    DEBUG_MOD,
    BIOME_NAME_LEN_MIN,
    BIOME_NAME_LEN_MAX,
    BIOME_PLUVIOMETRY_MIN,
    BIOME_PLUVIOMETRY_MAX,
    BIOME_TEMPERATURE_MIN,
    BIOME_TEMPERATURE_MAX,
    BIOME_TREES_LEN_MIN,
    BIOME_TREES_LEN_MAX
)


class Biome:
    ###############################################################
    ########################## __SLOTS__ ##########################
    ###############################################################
    __slots__ = (
        "_name",
        "_pluviometry_min",
        "_pluviometry_max",
        "_temperature_min",
        "_temperature_max",
        "_height_min",
        "_height_max",
        "_ground_color",
        "_trees"
    )

    ###############################################################
    ############################ HINTS ############################
    ###############################################################
    _name: str
    _pluviometry_min: float
    _pluviometry_max: float
    _temperature_min: float
    _temperature_max: float
    _height_min: float
    _height_max: float
    _ground_color: Color
    _trees: List[Tree]

    ###############################################################
    ########################## __INIT__ ###########################
    ###############################################################
    def __init__(
            self,
            name: str,
            pluviometry_min: float,
            pluviometry_max: float,
            temperature_min: float,
            temperature_max: float,
            height_min: float,
            height_max: float,
            ground_color: Color,
            trees: List[Tree]
    ) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Crée un objet Biome, caractérisé par :
        # - son nom
        # - sa température minimale
        # - sa température maximale
        # - sa pluviometrie minimale
        # - sa pluviometrie maximale
        # - la couleur de son sol
        # - une liste des arbres qui y poussent
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_NAME_LEN_MIN <= len(name) <= BIOME_NAME_LEN_MAX
        # - Les temperatures & pluviometries concordent avec leurs minimums et maximums
        # =============================
        self.set_name(name)

        self.set_pluviometry_min(pluviometry_min)
        self.set_pluviometry_max(pluviometry_max)
        self.set_temperature_min(temperature_min)
        self.set_temperature_max(temperature_max)
        self.set_height_min(height_min)
        self.set_height_max(height_max)

        self.set_ground_color(ground_color)
        self.set_trees(trees)

    ###############################################################
    ########################### GETTERS ###########################
    ###############################################################
    def get_name(self) -> str:
        return self._name

    def get_pluviometry_min(self) -> float:
        return self._pluviometry_min

    def get_pluviometry_max(self) -> float:
        return self._pluviometry_max

    def get_temperature_min(self) -> float:
        return self._temperature_min

    def get_temperature_max(self) -> float:
        return self._temperature_max
    
    def get_height_min(self) -> float:
        return self._height_min

    def get_height_max(self) -> float:
        return self._height_max

    def get_ground_color(self) -> Color:
        return self._ground_color

    def get_trees(self) -> List[Tree]:
        return self._trees
    
    ###############################################################
    ########################### SETTERS ###########################
    ###############################################################
    def set_name(self, name: str) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de name puis le set
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_NAME_LEN_MIN <= len(name) <= BIOME_NAME_LEN_MAX
        # =============================
        self._name = name

    def set_pluviometry_min(self, pluviometry_min: float) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de pluviometry_min puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_PLUVIOMETRY_MIN <= pluviometry_min <= BIOME_PLUVIOMETRY_MAX
        # =============================
        self._pluviometry_min = pluviometry_min

    def set_pluviometry_max(self, pluviometry_max: float) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de pluviometry_max puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_PLUVIOMETRY_MIN <= pluviometry_max <= BIOME_PLUVIOMETRY_MAX
        # =============================
        self._pluviometry_max = pluviometry_max

    def set_temperature_min(self, temperature_min: float) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de temperature_min puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_TEMPERATURE_MIN <= temperature_min <= BIOME_TEMPERATURE_MAX
        # =============================
        self._temperature_min = temperature_min

    def set_temperature_max(self, temperature_max: float) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de temperature_max puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_TEMPERATURE_MIN <= temperature_max <= BIOME_TEMPERATURE_MAX
        # =============================
        self._temperature_max = temperature_max

    def set_height_min(self, height_min: float) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de height_min puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_HEIGHT_MIN <= height_min <= BIOME_HEIGHT_MAX
        # =============================
        self._height_min = height_min

    def set_height_max(self, height_max: float) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de height_max puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_HEIGHT_MIN <= height_max <= BIOME_HEIGHT_MAX
        # =============================
        self._height_max = height_max

    def set_ground_color(self, ground_color: Color) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de ground_color puis l'assigne
        # =============================
        self._ground_color = ground_color

    def set_trees(self, trees: List[Tree]) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie la cohérence de pluviometry_max puis l'assigne
        # -----------------------------
        # PRÉCONDITIONS (uniquement si DEBUG_MOD) :
        # - BIOME_PLUVIOMETRY_MIN <= pluviometry_max <= BIOME_PLUVIOMETRY_MAX
        # =============================
        self._trees = trees

    ###############################################################
    ########################## IN_RANGE ###########################
    ###############################################################
    def in_range(self, temperature: float, pluviometry: float, height: float) -> bool:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Renvoie si la température et la pluviométrie
        # correspondent à celles de ce biome
        # =============================
        return (
                self.get_temperature_min() <= float(temperature) <= self.get_temperature_max() and
                self.get_pluviometry_min() <= float(pluviometry) <= self.get_pluviometry_max() # and
                # self.get_height_min() <= float(height) <= self.get_height_max()
        )

    ###############################################################
    ########################### __STR__ ###########################
    ###############################################################
    def __str__(self) -> str:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Passe les données du Biome en str. Utilisé pour debug
        # =============================
        return (
                "name: " + self.get_name() +
                "\ntemperature_min: " + str(self.get_temperature_min()) +
                "\ntemperature_max: " + str(self.get_temperature_max()) +
                "\npluviometry_min: " + str(self.get_pluviometry_min()) +
                "\npluviometry_max: " + str(self.get_pluviometry_max()) +
                "\nheight_min: " + str(self.get_height_min()) +
                "\nheight_max: " + str(self.get_height_max()) +
                "\nground_color: " + str(self.get_ground_color()) +
                "\ntrees: " + str(self.get_trees())
        )
