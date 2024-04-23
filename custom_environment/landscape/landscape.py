import numpy as np
import math
import pygame
import cv2
from copy import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from pettingzoo import ParallelEnv
from agents import QuadDrone

from pyprocgen import BoardBox, Seed, Box
from pyprocgen.decisional import generate_box
from pyprocgen.encyclopedia_functions import encyclopedia_creation
from pyprocgen.image_creation import  write_image_body, write_tile_body, read_tile_body


class landscapev0(ParallelEnv): # Unify X, Y CORDS
    metadata = {
        "name": "landscape_v0",
    }

    def __init__(self,
                 pixels_per_meter: int = 4,
                 darkening_factor: float = 0.5,
                 landscape_size: int = 128,
                 center_percent: float = 0.15,
                 clues = [
                     [0.10, 0.20], [0.10, 0.20], [0.10, 0.20],
                     [0.20, 0.30], [0.20, 0.30],
                     [0.30, 0.40], [0.30, 0.40], 
                 ],
                 map_seed=None,
                 heatmap_decay: float = 0.013,
                 terminal_time_steps: int = 500,
                 num_drones: int = 8
                 ) -> None:
        
        # Basic configuration
        self.ppm = pixels_per_meter
        self.darkening_factor = darkening_factor
        self.size = (landscape_size, landscape_size)
        self.center_percent = center_percent
        self.clues_percents = clues
        
        # Simulation state
        self.map_seed = map_seed if map_seed is not None else Seed()
        self.heatmap_decay = heatmap_decay
        self.terminal_time_steps = terminal_time_steps
        self.drones = [QuadDrone() for _ in range(num_drones)]
        
        # Internal mappings and states
        self.home_base = None
        self.objective = None
        self.heatmap = None
        self.done = False
        self.clues = []
        self.original_tiles = [] # [home_base, objective, *clues]
        self.rewards = 0
        self.time_steps = 0
        self.reward_values = {'tiles': 1, 'crash': -100, 'clue': 50, 'objective': 100}

        self.drone_starts = [
            [3, -1], [3, 1],
            [1, -3], [1, 3],
            [-1, -3], [-1, 3],
            [-3, -1], [-3, 1],
        ]
                    
        # Map generation
        self.map = BoardBox.create_empty_board(*self.size)
        self.tile_map = np.zeros(self.size)
        self.img_map = np.zeros((*self.size, 3))
        self.discovery_map = np.zeros(self.size)
        
        self.encyclopedia = encyclopedia_creation()
        self.clue_index = list(self.encyclopedia._biomes.keys()).index('clue')
        self.objective_index = list(self.encyclopedia._biomes.keys()).index('objective')

    def generate_map(self, map):

        for line_number in tqdm(range(self.size[0]), desc="Creating Map"):
            for column_number in range(self.size[1]):

                map.set_element(
                    value=generate_box(
                        self.encyclopedia,
                        column_number,
                        line_number,
                        self.map_seed
                    ),
                    x=column_number,
                    y=line_number
                )

        return map

    def reset(self, seed=None, options={}):
        
        self.reset_map(options)
        self.reset_locations(options)
        self.read_map()
        self.reset_drones(options)
        self.reset_heatmap(options)

        self.done = False
        self.rewards = 0
        self.time_steps = 0

        self.discovery_map = np.zeros(self.size)

        self.screen = pygame.display.set_mode(self.img_map.shape[:2])
        pygame.display.set_caption("Landscape Map")

        return None, None
    
    def reset_map(self, options):
        if not options.get('reset_map', 1):
            if len(self.original_tiles):
                for i, special_tile in enumerate([self.home_base, self.objective, *self.clues]):
                    biome_name = list(self.encyclopedia._biomes.keys())[self.original_tiles[i]]
                    self.map.set_element(
                        value=Box(self.encyclopedia._biomes[biome_name], 0.0, 0.0, 0.0),
                        x=special_tile[0],
                        y=special_tile[1]
                    )
                self.read_map()
            return

        if options.get('map_seed', 0):
            self.map_seed=Seed()

        self.map = self.generate_map(self.map)
        self.read_map()

    def reset_locations(self, options):
        if not options.get('reset_locations', 1):
            return
        
        water_tiles = 11 # Last 10 index are water
        land_mask = self.tile_map < len(self.encyclopedia._biomes.keys()) - water_tiles

        # Create a center mask based on the desired radius
        center = (self.size[1] // 2, self.size[0] // 2)
        home_mask = self.create_centered_mask(self.center_percent, center)
        
        # Combine the land mask and center mask to get valid locations for home_base
        valid_home_base_locations = land_mask & home_mask
        self.home_base = self.pick_location(valid_home_base_locations)
        
        # Create a mask for valid locations outside the center for the searched object
        obj_mask = ~self.create_centered_mask(self.center_percent + 0.2, self.home_base)
        valid_object_locations = land_mask & obj_mask

        # Select a random location for the searched object from the valid locations
        self.objective = self.pick_location(valid_object_locations)

        home_base_biome = self.map.get_element(x=self.home_base[1], y=self.home_base[0])._biome
        home_base_value = list(self.encyclopedia._biomes.keys()).index(home_base_biome._name)
        self.original_tiles.append(home_base_value)

        objective_biome = self.map.get_element(x=self.objective[1], y=self.objective[0])._biome
        objective_value = list(self.encyclopedia._biomes.keys()).index(objective_biome._name)
        self.original_tiles.append(objective_value)
        
        self.map.set_element(
                    value=Box(self.encyclopedia._biomes['home'], 0.0, 0.0, 0.0),
                    x=self.home_base[0],
                    y=self.home_base[1]
                )

        self.map.set_element(
                    value=Box(self.encyclopedia._biomes['objective'], 0.0, 0.0, 0.0),
                    x=self.objective[0],
                    y=self.objective[1]
                )


        for i, clue in enumerate(self.clues_percents):
            inner_mask = ~self.create_centered_mask(clue[0], self.objective)
            outer_mask = self.create_centered_mask(clue[1], self.objective)
            clue_mask = inner_mask & outer_mask & land_mask & ~home_mask

            clue_location = self.pick_location(clue_mask)
            if clue_location is not None:
                self.clues.append(clue_location)
                
                clue_biome = self.map.get_element(x=clue_location[1], y=clue_location[0])._biome
                clue_value = list(self.encyclopedia._biomes.keys()).index(clue_biome._name)
                self.original_tiles.append(clue_value)

                self.map.set_element(
                        value=Box(self.encyclopedia._biomes['clue'], 0.0, 0.0, 0.0),
                        x=clue_location[0],
                        y=clue_location[1]
                    )

        print()
        

    def reset_drones(self, options):
        if not options.get('reset_drone', 1):
            return
        
        if len(self.drones) > len(self.drone_starts):
            raise ValueError("Too Many Drones, Add more starting areas")

        for i in range(len(self.drones)):
            self.drones[i].reset()
            realtive_start = self.drone_starts[i]
            start = np.array(realtive_start) + np.array(self.home_base)
            self.drones[i].position = start.astype(np.float32)
            mask, self.drones[i].observation = self.return_observation(self.drones[i].position)
            self.discovery_map[mask] = 1


    def create_centered_mask(self, center_percent, center):
        center_radius = int(min(self.size) * center_percent)

        y_grid, x_grid = np.ogrid[-center[0]:self.size[0]-center[0], -center[1]:self.size[1]-center[1]]
        centered_mask = x_grid**2 + y_grid**2 <= center_radius**2
        
        return centered_mask
    

    def pick_location(self, location_mask):
        possible_indices = np.argwhere(location_mask)

        if len(possible_indices) > 0:
            picked_index = possible_indices[np.random.choice(len(possible_indices))]
            picked = (picked_index[0], picked_index[1])
            return picked
        else:
            # Deal with this option later
            print("No valid location found for home base")
            return None


    def reset_heatmap(self, options):
        if not options.get('reset_heatmap', 1):
            return
         
        self.heatmap = np.zeros(self.size)

        for drone in self.drones:
            self.heatmap = np.maximum(self.gaussian_heatmap(drone.position), self.heatmap)

        print()

    def read_map(self,):
        self.img_map = np.zeros((*self.size, 3))
        self.tile_map = write_tile_body(self.tile_map, self.map, self.encyclopedia)
        # self.graph_height_biomes()
        self.img_map = read_tile_body(self.tile_map, self.img_map, self.encyclopedia)[:,:,::-1]
        self.img_map = cv2.resize(self.img_map, (0,0), 
                                  fx=self.ppm, fy=self.ppm, 
                                  interpolation=cv2.INTER_NEAREST
                                )

    def gaussian_heatmap(self, center, sigma=2.0):
        """
        Create a heatmap with a Gaussian blur applied around a single point.
        
        Args:
            center (tuple): Tuple of (x, y) coordinates for the Gaussian center.
            grid_size (int): Size of the x and y dimensions of the output grid.
            sigma (float): Standard deviation for the Gaussian blur.
        
        Returns:
            numpy.ndarray: A grid_size x grid_size heatmap with a blurred Gaussian point.
        """
        # Create an empty heatmap
        heatmap = np.zeros(self.size, dtype=np.float32)
        
        # Draw a single point on the heatmap
        cv2.circle(heatmap, center.astype(int)[::-1], radius=0, color=1, thickness=-1)  # 'thickness=-1' fills the circle
        
        # Apply Gaussian blur to the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        # Normalize the heatmap to [0, 1]
        heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return heatmap

    def step(self, actions):
        new_heatmap = np.zeros(self.size)

        self.rewards = 0

        for drone in self.drones:
            if drone.crashed:
                self.rewards -= 100
                continue
            
            action = actions[drone]
            drone.set_motor_powers(action)

            mask, obv = self.return_observation(drone.position)
            # mask = np.transpose(dis_mask, axes=(1, 0))
            discovered_tiles = sum(~self.discovery_map[mask].astype(bool)) * self.reward_values['tiles']
            self.rewards += discovered_tiles

            clues_obved = np.where(self.tile_map[mask] == self.clue_index)[0] # - len(self.tile_map[mask])
            if len(clues_obved):
                # If any of the observed clues have not been discovered
                clue_discovered = ~self.discovery_map[mask][clues_obved].astype(bool)
                if clue_discovered.any():
                    self.rewards += sum(clue_discovered) * self.reward_values['clue']

            self.discovery_map[mask] = 1

            if self.discovery_map[*self.objective]:
                self.done = True
                self.rewards = self.reward_values['objective']

            new_heatmap = np.maximum(self.gaussian_heatmap(drone.position), new_heatmap)

            drone.observation = obv
            if drone.heatmap is None:
                drone.heatmap = self.gaussian_heatmap(drone.position)
            else:
                drone.heatmap -= self.heatmap_decay
                drone.heatmap = np.clip(drone.heatmap, 0, 1)
                drone.heatmap = np.maximum(drone.heatmap, self.gaussian_heatmap(drone.position))

        self.heatmap -= self.heatmap_decay
        self.heatmap = np.clip(self.heatmap, 0, 1)
        self.heatmap = np.maximum(self.heatmap, new_heatmap)

        if self.time_steps > self.terminal_time_steps:
            self.done = True
            distances = []
            for drone in self.drones:
                distances.append(self.distance(drone.position))
            
            avg_distance = np.mean(distances)
            self.rewards -= avg_distance

        self.time_steps += 1 

        return self.rewards, self.done


    def distance(self, position):
        x1, y1 = position
        x2, y2 = self.objective
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    

    def return_observation(self, position, n=4):
        x, y = int(position[0]), int(position[1])
        height, width = self.size

        # Create an array of indices
        indices_y, indices_x = np.meshgrid(np.arange(height), np.arange(width))

        # Calculate the squared distance from the point (x, y)
        distance_squared = (indices_x - x) ** 2 + (indices_y - y) ** 2

        # Create a mask where the squared distance is less than or equal to n squared
        mask = distance_squared <= n ** 2

        # Use the mask to select the values within the radius
        obv = copy(self.tile_map[mask])

        return mask, obv


    def render(self):
        if not pygame.get_init():
            pygame.init()
        
        background = copy(self.img_map)
        map_mask = cv2.resize(copy(self.discovery_map), (0,0), 
                                  fx=self.ppm, fy=self.ppm, 
                                  interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
        map_mask = np.transpose(map_mask, axes=(1, 0))
        background[~map_mask] = (background[~map_mask] * self.darkening_factor)

        self.background_surface = pygame.surfarray.make_surface(background)

        self.screen.blit(self.background_surface, (0, 0))

        for drone in self.drones:
            rect, drone_color = drone.render(self.ppm)
            pygame.draw.rect(self.screen, drone_color, rect)


        pygame.display.flip()
        self.render_heatmap()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def render_heatmap(self,):
        heatmap_normalized = cv2.normalize(self.heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Convert to 8-bit (0-255) and apply colormap
        heatmap_8bit = np.uint8(255 * heatmap_normalized)  # Scale to [0, 255]
        colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)  # Apply the JET colormap
        colored_heatmap = cv2.resize(colored_heatmap, (0,0), 
                                  fx=self.ppm, fy=self.ppm, 
                                  interpolation=cv2.INTER_NEAREST
                                )

        # Optionally, display using matplotlib to compare
        cv2.imshow("Heatmap", colored_heatmap)  # Convert BGR to RGB
        cv2.waitKey(1)