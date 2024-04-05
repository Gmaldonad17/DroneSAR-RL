# ==========================================================
# INFORMATIONS SUR CE PACKAGE :
# -----------------------------
# UTILITÉ DE SON CONTENU :
# Définir la classe Color, une couleur au format rgb
# -----------------------------
# CONTENU :
# + __slots__
# + HINTS
# + __init__()
# + GETTERS
# + SETTERS
# + get_color()
# + get_hexadecimal()
# + __str__()
# ==========================================================

from pyprocgen.settings import DEBUG_MOD, COLOR_RGB_MIN, COLOR_RGB_MAX


class Color:
    ###############################################################
    ########################## __SLOTS__ ##########################
    ###############################################################
    __slots__ = (
        "_red",
        "_green",
        "_blue"
    )

    ###############################################################
    ############################ HINTS ############################
    ###############################################################
    _red: int
    _green: int
    _blue: int

    ###############################################################
    ########################## __INIT__ ###########################
    ###############################################################
    def __init__(self, red: int, green: int, blue: int) -> None:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Crée un objet Color, caractérisé par :
        # - red
        # - green
        # - blue
        # -----------------------------
        # PRÉCONDITION:
        # - COLOR_RGB_MIN <= (red, green & blue) <= COLOR_RGB_MAX
        # =============================
        self.set_red(red)
        self.set_green(green)
        self.set_blue(blue)

    ###############################################################
    ########################### GETTERS ###########################
    ###############################################################
    def get_red(self) -> int:
        return self._red

    def get_green(self) -> int:
        return self._green

    def get_blue(self) -> int:
        return self._blue

    ###############################################################
    ########################### SETTERS ###########################
    ###############################################################
    def set_red(self, red: int) -> None:
        self._red = red

    def set_green(self, green: int) -> None:
        self._green = green

    def set_blue(self, blue: int) -> None:
        self._blue = blue

    ###############################################################
    ###################### GET_HEXADECIMAL ########################
    ###############################################################
    def get_hexadecimal(self) -> str:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Renvoi la couleur en hexadecimal
        # =============================
        return hex(self.get_red()) + hex(self.get_green())[2:] + hex(self.get_blue())[2:]  # [2:] pour enlever le 0x

    ###############################################################
    ########################### GET_RGB ###########################
    ###############################################################
    def get_rgb(self) -> str:
        # =============================
        # INFORMATIONS :
        # -----------------------------
        # UTILITÉ :
        # Renvoi la couleur en rgb
        # =============================
        return str(self.get_red()) + " " + str(self.get_green()) + " " + str(self.get_blue())

    ###############################################################
    ########################### __STR__ ###########################
    ###############################################################
    def __str__(self):
        return self.get_rgb()
