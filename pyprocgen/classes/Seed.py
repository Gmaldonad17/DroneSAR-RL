# ==========================================================
# INFORMATIONS SUR CE PACKAGE :
# -----------------------------
# UTILITÉ DE SON CONTENU :
# Définir la classe Seed
# -----------------------------
# CONTENU :
# + __slots__
# + HINTS
# + __init__()
# + GETTERS
# + SETTERS
# + is_seed()
# + __str__()
# ==========================================================

from __future__ import annotations
from typing import Optional
import random

from pyprocgen.settings import DEBUG_MOD, SEED_ELEMENT_MIN, SEED_ELEMENT_MAX

class Seed:
    ###############################################################
    ########################## __SLOTS__ ##########################
    ###############################################################
    __slots__ = (
        "_temperature_x",
        "_temperature_y",
        "_pluviometry_x",
        "_pluviometry_y",
        "_height_x",
        "_height_y"
    )

    ###############################################################
    ########################### HINTS #############################
    ###############################################################
    _temperature_x: int
    _temperature_y: int
    _pluviometry_x: int
    _pluviometry_y: int
    _height_x: int
    _height_y: int

    ###############################################################
    ########################## __INIT__ ###########################
    ###############################################################
    def __init__(
            self,
            pluviometry_x: Optional[int] = None,
            pluviometry_y: Optional[int] = None,
            temperature_x: Optional[int] = None,
            temperature_y: Optional[int] = None,
            height_x: Optional[int] = None,
            height_y: Optional[int] = None,
            seed_in_string: Optional[str] = None,
    ) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Crée un objet Seed, caractérisé par :
        # - sa temperature_x
        # - sa temperature_y
        # - sa pluviometry_x
        # - sa pluviometry_y
        # -----------------------------
        # PRÉCONDITION:
        # - si seed_in_string != None, il valide is_seed()
        # - SEED_ELEMENT_MIN <= (pluviometry_x, pluviometry_y, temperature_x, temperature_y) <= SEED_ELEMENT_MAX
        # =============================
        if isinstance(seed_in_string, str):
            if DEBUG_MOD and not Seed.is_seed(seed_in_string):
                raise ValueError(
                    "Error: trying to set a " + type(self).__name__ +
                    " from a str, but the str hasn't shape of a str(Seed)."
                )
            splitted_string = seed_in_string.split(":")
            self._pluviometry_x = int(splitted_string[0])  # Pas besoin d'utiliser les SETTERS puisque tout est
            self._pluviometry_y = int(splitted_string[1])  # déjà check avec is_seed
            self._temperature_x = int(splitted_string[2])
            self._temperature_y = int(splitted_string[3])
            self._height_x = int(splitted_string[4])
            self._height_y = int(splitted_string[5])
        else:
            self.set_pluviometry_x(pluviometry_x)
            self.set_pluviometry_y(pluviometry_y)
            self.set_temperature_x(temperature_x)
            self.set_temperature_y(temperature_y)
            self.set_height_x(height_x)
            self.set_height_y(height_y)

    ###############################################################
    ########################### GETTERS ###########################
    ###############################################################
    def get_pluviometry_x(self) -> int:
        return self._pluviometry_x

    def get_pluviometry_y(self) -> int:
        return self._pluviometry_y

    def get_temperature_x(self) -> int:
        return self._temperature_x

    def get_temperature_y(self) -> int:
        return self._temperature_y
    
    def get_height_x(self) -> int:
        return self._height_x

    def get_height_y(self) -> int:
        return self._height_y

    ###############################################################
    ########################### SETTERS ###########################
    ###############################################################
    def set_pluviometry_x(self, pluviometry_x: Optional[int]) -> None:
        if pluviometry_x is None:
            self._pluviometry_x = random.randint(SEED_ELEMENT_MIN, SEED_ELEMENT_MAX)
        else:
            self._pluviometry_x = pluviometry_x

    def set_pluviometry_y(self, pluviometry_y: Optional[int]) -> None:
        if pluviometry_y is None:
            self._pluviometry_y = random.randint(SEED_ELEMENT_MIN, SEED_ELEMENT_MAX)
        else:
            self._pluviometry_y = pluviometry_y

    def set_temperature_x(self, temperature_x: Optional[int]) -> None:
        if temperature_x is None:
            self._temperature_x = random.randint(SEED_ELEMENT_MIN, SEED_ELEMENT_MAX)
        else:
            self._temperature_x = temperature_x

    def set_temperature_y(self, temperature_y: Optional[int]) -> None:
        if temperature_y is None:
            self._temperature_y = random.randint(SEED_ELEMENT_MIN, SEED_ELEMENT_MAX)
        else:
            self._temperature_y = temperature_y

    def set_height_x(self, height_x: Optional[int]) -> None:
        if height_x is None:
            self._height_x = random.randint(SEED_ELEMENT_MIN, SEED_ELEMENT_MAX)
        else:
            self._height_x = height_x

    def set_height_y(self, height_y: Optional[int]) -> None:
        if height_y is None:
            self._height_y = random.randint(SEED_ELEMENT_MIN, SEED_ELEMENT_MAX)
        else:
            self._height_y = height_y

    ###############################################################
    ########################### IS_SEED ###########################
    ###############################################################
    @staticmethod
    def is_seed(string: str) -> bool:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Vérifie que seed_in_string est bien
        # de la forme a + ":" + b + ":" + c + ":" + d
        # avec a, b,c,d str(integers)
        # =============================
        if DEBUG_MOD and not isinstance(string, str):
            raise TypeError(
                "Error: impossible to check if a string is a Seed:" +
                "\nstring must be str, but a " + type(string).__name__ + " is given."
            )

        colon_counter = 0
        position_counter = 0
        # number_of_following_colon est initialisé à 1 pour contrer le cas où seed_in_string[0] == ":"
        number_of_following_colon = 1

        # Parcours de string pour s'assurer qu'il a la forme d'un str(Seed)
        while (
                position_counter < len(string)
                and string[position_counter] in "0123456789-:"
                and number_of_following_colon != 2
                and not (  # Pour éviter un - au milieu d'un entier
                        position_counter != 0
                        and string[position_counter] == "-"
                        and string[position_counter - 1] != ":"
                )
        ):

            if string[position_counter] == ":":
                colon_counter += 1
                number_of_following_colon += 1

            else:
                number_of_following_colon = 0

            position_counter += 1

        if (
                position_counter == len(string)
                and colon_counter == 3
                and number_of_following_colon == 0
        ):  # Il a la forme d'un str(Seed) mais il faut vérifier qu'il respecte les conditions des valeurs de Seed
            splitted_string = string.split(":")
            return (
                SEED_ELEMENT_MIN <= int(splitted_string[0]) <= SEED_ELEMENT_MAX and  # pluviometry_x
                SEED_ELEMENT_MIN <= int(splitted_string[1]) <= SEED_ELEMENT_MAX and  # pluviometry_y
                SEED_ELEMENT_MIN <= int(splitted_string[2]) <= SEED_ELEMENT_MAX and  # temperature_x
                SEED_ELEMENT_MIN <= int(splitted_string[3]) <= SEED_ELEMENT_MAX and  # temperature_y
                SEED_ELEMENT_MIN <= int(splitted_string[4]) <= SEED_ELEMENT_MAX and  # height_x
                SEED_ELEMENT_MIN <= int(splitted_string[5]) <= SEED_ELEMENT_MAX  # height_y
            )
        else:
            return False

    ###############################################################
    ########################### __STR__ ###########################
    ###############################################################
    def __str__(self) -> str:
        return (
                str(self.get_pluviometry_x()) + ":" +
                str(self.get_pluviometry_y()) + ":" +
                str(self.get_temperature_x()) + ":" +
                str(self.get_temperature_y()) + ":" +
                str(self.get_height_x()) + ":" +
                str(self.get_height_y())
        )

