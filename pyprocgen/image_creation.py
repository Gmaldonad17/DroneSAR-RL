# ==========================================================
# INFORMATIONS SUR CE PACKAGE :
# -----------------------------
# UTILITÉ DE SON CONTENU :
# Créer une image à partir d'un tableau de Biomes
# -----------------------------
# CONTENU :
# - write_image_header(destination_file, height, width, seed)
# - write_image_body(board)
# ==========================================================

from typing import TextIO
import numpy as np

from pyprocgen import BoardBox


###############################################################
##################### WRITE_IMAGE_HEADER ######################
###############################################################
def write_image_header(destination_file: TextIO, height: int, width: int, seed: str):
    # =============================
    # INFORMATIONS :
    # -----------------------------
    # UTILITÉ :
    # Écrit le header de destination_file
    # selon le modèle d'un header de fichier ppm.
    # -----------------------------
    # DEPEND DE :
    # - os
    # =============================

    destination_file.write("P3\n")
    destination_file.write("# Seed : " + seed + "\n")
    destination_file.write(str(width))
    destination_file.write("\n")
    destination_file.write(str(height))
    destination_file.write("\n")
    destination_file.write("255\n")
    destination_file.write("\n")


###############################################################
####################### WRITE_IMAGE_BODY ######################
###############################################################

def write_image_body(img: np.array, board: BoardBox):
    
    for line in range(board.get_height()):
        for column in range(board.get_width()):
            color = board.get_element(x=column, y=line).get_color().get_rgb()
            color = [int(i) for i in color.split()][::-1]
            img[column, line] = color

    return img

def write_image_body(img: np.array, board: BoardBox):
    
    for line in range(board.get_height()):
        for column in range(board.get_width()):
            color = board.get_element(x=column, y=line).get_color().get_rgb()
            color = [int(i) for i in color.split()][::-1]
            img[column, line] = color

    return img


def write_tile_body(tiles: np.array, board: BoardBox, encyclopedia):
    for line in range(board.get_height()):
        for column in range(board.get_width()):
            box = board.get_element(x=column, y=line)
            biome_name = box._biome._name
            biome_index = list(encyclopedia._biomes.keys()).index(biome_name)
            tiles[column, line] = np.array([biome_index])

    return tiles


def read_tile_body(tiles: np.array, img: np.array, encyclopedia):
    for line in range(tiles.shape[0]):
        for column in range(tiles.shape[1]):
            biome_index = int(tiles[line, column])
            color = list(encyclopedia._biomes.values())[biome_index]._ground_color.get_rgb()
            color = [int(i) for i in color.split()][::-1]
            img[column, line] = color
    
    return img