import numpy as np


def generate_motor_actions(max_power_steps):
    # Create the range of possible values for each motor
    step = 1 / max_power_steps
    possible_values = np.arange(0, 1 + step, step)
    
    # Generate all possible combinations of these values for the four motors
    actions = np.array(np.meshgrid(possible_values, possible_values, possible_values, possible_values)).T.reshape(-1, 4)
    
    # Filter out combinations where opposing motors have the same power
    filtered_actions = []
    for action in actions:
        top_left, top_right, bottom_left, bottom_right = action
        if not (top_left == bottom_right and top_right == bottom_left):
            filtered_actions.append(action)

    return np.array(filtered_actions)

def mask_undiscovered_tiles(tile_map, discovery_map):
    # Create a copy of the tile map
    masked_tile_map = np.copy(tile_map)
    
    # Mask out undiscovered tiles
    # Where discovery_map is 0, set the corresponding tiles in masked_tile_map to a default value, e.g., 0
    masked_tile_map[discovery_map == 0] = 0  # or any other value you consider as 'undiscovered' or 'hidden'
    
    return masked_tile_map

def get_model_input(env):
    base_model_input = [env.position_heatmap.astype(np.float32), 
                        env.discovery_map.astype(np.float32),
                       ]
    features_heatmap = env.features_heatmap
    features_heatmap[:5, :] = 0
    features_heatmap[-5:, :] = 0
    features_heatmap[:, :5] = 0
    features_heatmap[:, -5:] = 0
    base_model_input.append(features_heatmap.astype(np.float32))
    
    return base_model_input


def calculate_action_dissimilarity(actions):
    """
    Calculate the average dissimilarity between actions taken by drones.

    Args:
    - actions (dict): A dictionary mapping drones to their actions, each action being a numpy array.

    Returns:
    - float: The average dissimilarity of actions.
    """
    action_list = list(actions.values())
    num_actions = len(action_list)
    if num_actions < 2:
        return 0  # Not enough actions to compare dissimilarity

    # Assuming actions are numpy arrays, calculate pairwise Euclidean distances
    action_matrix = np.stack(action_list)
    diff = np.linalg.norm(action_matrix[:, np.newaxis] - action_matrix[np.newaxis, :], axis=2)
    
    # Sum up all the distances and normalize
    total_dissimilarity = np.sum(np.triu(diff, k=1))  # Only upper triangle to avoid redundant pairs
    num_pairs = num_actions * (num_actions - 1) / 2
    average_dissimilarity = total_dissimilarity / num_pairs if num_pairs > 0 else 0

    return average_dissimilarity