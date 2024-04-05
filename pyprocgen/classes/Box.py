# ==========================================================
# INFORMATIONS SUR CE PACKAGE :
# -----------------------------
# UTILITÉ DE SON CONTENU :
# Définir la classe Box, une case
# -----------------------------
# CONTENU :
# + __slots__
# + HINTS
# + __init__()
# + getters
# + setters
# + get_color()
# ==========================================================

from pyprocgen.classes.Biome import Biome
from pyprocgen.classes.Color import Color

from pyprocgen.settings import DEBUG_MOD


class Box:
    ###############################################################
    ########################## __SLOTS__ ##########################
    ###############################################################
    __slots__ = (
        "_biome",
        "_temperature",
        "_pluviometry",
        "_height",
        
    )

    ###############################################################
    ############################ HINTS ############################
    ###############################################################
    _biome: Biome

    ###############################################################
    ########################## __INIT__ ###########################
    ###############################################################
    def __init__(
            self,
            biome: Biome,
            temperature=0.0,
            pluviometry=0.0,
            height=0.0,
    ) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Crée un objet Box (case), caractérisé par :
        # - son biome
        # =============================
        self.set_biome(biome)
        self.set_temperature(temperature)
        self.set_pluviometry(pluviometry)
        self.set_height(height)

    ###############################################################
    ########################### GETTERS ###########################
    ###############################################################
    def get_biome(self) -> Biome:
        return self._biome
    
    def get_temperature(self) -> float:
        return self._temperature
    
    def get_pluviometry(self) -> float:
        return self._pluviometry
    
    def get_pluviometry(self) -> float:
        return self._pluviometry

    ###############################################################
    ########################### SETTERS ###########################
    ###############################################################
    def set_biome(self, biome: Biome) -> None:
        self._biome = biome

    def set_temperature(self, temperature: float) -> None:
        self._temperature = temperature

    def set_pluviometry(self, pluviometry: float) -> None:
        self._pluviometry = pluviometry

    def set_height(self, height: float) -> None:
        self._height = height

    ###############################################################
    ########################## GET_COLOR ##########################
    ###############################################################
    def get_color(self) -> Color:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Revoie la couleur de la case,
        # donc celle du sol puisque il n'y a pas d'arbre
        # =============================
        return self.get_biome().get_ground_color()
