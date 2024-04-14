import numpy as np
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


class landscapev0(ParallelEnv):
    metadata = {
        "name": "landscape_v0",
    }

    def __init__(self, 
                 pixels_per_meter=2, 
                 landscape_size=200,
                 center_percent=0.15,
                 hints=[
                     [0.15, 0.25], 
                     [0.25, 0.35], 
                     [0.35, 0.55],
                     ],
                 map_seed=Seed(), 
                 pre_gen="", 
                 num_drones=8):

        self.drones = [QuadDrone() for i in range(num_drones)]

        self.ppm = pixels_per_meter
        self.size = (landscape_size, landscape_size)
        self.map_seed = map_seed
        self.center_percent = center_percent
        self.hints_percents = hints

        
        self.home_base = None
        self.objective = None
        self.hints = []

        self.drone_starts = [
            [-1, 3], [1, 3], 
            [-3, 1], [3, 1], 
            [-3, -1], [3, -1],
            [-1, -3], [1, -3], 
        ]

        self.encyclopedia = encyclopedia_creation()
        self.map = BoardBox.create_empty_board(*self.size)
        self.tile_map = np.zeros((*self.size, 2))
        self.img_map = np.zeros((*self.size, 3))

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

    def reset(self, options={}):
        
        self.reset_map(options)
        self.reset_locations(options)
        self.read_map()
        self.reset_drones(options)

        self.screen = pygame.display.set_mode(self.img_map.shape[:2])
        pygame.display.set_caption("Landscape Map")

        self.background_surface = pygame.surfarray.make_surface(self.img_map[:,:,::-1])
        
        return None, None
    
    def reset_map(self, options):
        if not options.get('reset_map', 1):
            return

        if options.get('map_seed', 0):
            self.map_seed=Seed()

        self.map = self.generate_map(self.map)
        self.read_map()

    def reset_locations(self, options):
        if not options.get('reset_map', 1):
            return
        
        water_tiles = 11 # Last 10 index are water
        land_mask = self.tile_map[:, :, 0] < len(self.encyclopedia._biomes.keys()) - water_tiles

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

        self.map.set_element(
                    value=Box(self.encyclopedia._biomes['home'], 0.0, 0.0, 0.0),
                    x=self.home_base[1],
                    y=self.home_base[0]
                )

        self.map.set_element(
                    value=Box(self.encyclopedia._biomes['objective'], 0.0, 0.0, 0.0),
                    x=self.objective[1],
                    y=self.objective[0]
                )


        for i, hint in enumerate(self.hints_percents):
            inner_mask = ~self.create_centered_mask(hint[0], self.objective)
            outer_mask = self.create_centered_mask(hint[1], self.objective)
            hint_mask = inner_mask & outer_mask & land_mask & ~home_mask

            hint_location = self.pick_location(hint_mask)
            if hint_location is not None:
                self.hints.append(hint_location)

        for hint in self.hints:
            self.map.set_element(
                    value=Box(self.encyclopedia._biomes['hint'], 0.0, 0.0, 0.0),
                    x=hint[1],
                    y=hint[0]
                )
        
    def reset_drones(self, options):
        if not options.get('reset_drone', 1):
            return
        
        if len(self.drones) > len(self.drone_starts):
            raise "Too Many Drones, Add more starting areas"

        for i in range(len(self.drones)):
            realtive_start = self.drone_starts[i]
            start = np.array(realtive_start) + np.array(self.home_base)
            self.drones[i].position = start.astype(np.float32)


    def create_centered_mask(self, center_percent, center):
        center_radius = int(min(self.size) * center_percent)

        y_grid, x_grid = np.ogrid[-center[1]:self.size[0]-center[1], -center[0]:self.size[1]-center[0]]
        centered_mask = x_grid**2 + y_grid**2 <= center_radius**2
        
        return centered_mask
    

    def pick_location(self, location_mask):
        possible_indices = np.argwhere(location_mask)

        if len(possible_indices) > 0:
            picked_index = possible_indices[np.random.choice(len(possible_indices))]
            picked = (picked_index[1], picked_index[0])
            return picked
        else:
            # Deal with this option later
            print("No valid location found for home base")
            return None

    def read_map(self,):
        self.img_map = np.zeros((*self.size, 3))
        self.tile_map = write_tile_body(self.tile_map, self.map, self.encyclopedia)
        # self.graph_height_biomes()
        self.img_map = read_tile_body(self.tile_map, self.img_map, self.encyclopedia)
        self.img_map = cv2.resize(self.img_map, (0,0), 
                                  fx=self.ppm, fy=self.ppm, 
                                  interpolation=cv2.INTER_NEAREST
                                )

    # def graph_height_biomes(self):
    #     biome_idx = abs(copy(self.tile_map)[:, :, 0].flatten() - len(self.encyclopedia._biomes.keys()))  # Assuming this has values in the range 0-32
    #     heights = self.tile_map[:, :, 1].flatten()  # Assuming this has values in the range 0-3

    #     fig, ax1 = plt.subplots(figsize=(10, 6))

    #     # Histogram for biome_idx on the primary x-axis
    #     color = 'tab:blue'
    #     ax1.set_xlabel('Biome_idx / Height')
    #     ax1.set_ylabel('Frequency', color=color)
    #     ax1.hist(biome_idx, bins=np.arange(-0.5, 42, 1), alpha=0.5, label='Biome_idx Slot (0-32)', color=color)
    #     ax1.tick_params(axis='y', labelcolor=color)

    #     # Instantiate a second axes that shares the same y-axis
    #     ax2 = ax1.twiny()  # Create a twin x-axis sharing the y-axis
    #     color = 'tab:red'
    #     # Histogram for heights on the secondary x-axis
    #     ax2.hist(heights, bins=np.arange(-0.875, 3, 0.1), alpha=0.5, label='Heights Slot (0-3)', color=color)
    #     ax2.tick_params(axis='x', labelcolor=color)

    #     # Add a legend to the plot
    #     fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    #     plt.title('Distribution of Biomes and Heights')
    #     plt.grid(True)
    #     plt.show()
    #     print()

    def step(self, actions):
        for action, drone in zip(actions, self.drones):

            # Setting 
            drone.set_motor_powers(action)

    def render(self):
        if not pygame.get_init():
            pygame.init()

        self.screen.blit(self.background_surface, (0, 0))

        for drone in self.drones:
            rect, drone_color = drone.render(self.ppm)
            pygame.draw.rect(self.screen, drone_color, rect)

    
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]