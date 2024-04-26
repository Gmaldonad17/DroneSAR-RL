import numpy as np

def generate_motor_actions(max_power_steps):
    # Create the range of possible values for each motor
    step = 1 / max_power_steps
    possible_values = np.arange(0, 1 + step, step)
    
    # Generate all possible combinations of these values for the four motors
    actions = np.array(np.meshgrid(possible_values, possible_values, possible_values, possible_values)).T.reshape(-1,4)
    
    return actions

def mask_undiscovered_tiles(tile_map, discovery_map):
    # Create a copy of the tile map
    masked_tile_map = np.copy(tile_map)
    
    # Mask out undiscovered tiles
    # Where discovery_map is 0, set the corresponding tiles in masked_tile_map to a default value, e.g., 0
    masked_tile_map[discovery_map == 0] = 0  # or any other value you consider as 'undiscovered' or 'hidden'
    
    return masked_tile_map