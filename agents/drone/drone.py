import numpy as np
from copy import copy
import pygame


class QuadDrone:
    def __init__(self, 
                 pixel_size=6, 
                 motor_powers=None, 
                 position=None, 
                 velocity=None, 
                 mass=0.5, 
                 max_motor_thrust=10.5, 
                 time_step=0.01,
                 drag_coefficient = 0.80
                 ):
        
        self.pixel_size = pixel_size
        self.motor_powers = np.zeros(4, dtype=np.float32) if motor_powers is None else motor_powers
        self.position = np.zeros(2, dtype=np.float32) if position is None else position
        self.velocity = np.zeros(2, dtype=np.float32) if velocity is None else velocity
        self.mass = mass
        self.max_motor_thrust = max_motor_thrust
        self.time_step = time_step
        self.drag_coefficient = drag_coefficient

        self.crashed = False
        self.observation = None
        self.heatmap = None

    def reset(self,):
        self.crashed = False
        self.motor_powers = np.zeros(4)
        self.position = np.array([0.0, 0.0], dtype=np.float32) # Y, Z position
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # Y, Z velocity

    def set_motor_powers(self, powers):
        """Sets the power for each motor and updates the drone's state."""
        self.motor_powers = np.clip(np.array(powers), 0, 1)  # Ensure powers are within [0, 1]
        self.update_dynamics()

    def update_dynamics(self):
        # Compute thrust difference for X and Y directions
        thrust_diff_x = ((self.motor_powers[1] + self.motor_powers[3]) - \
                         (self.motor_powers[0] + self.motor_powers[2])) * self.max_motor_thrust
                         
        thrust_diff_y = ((self.motor_powers[2] + self.motor_powers[3]) - \
                         (self.motor_powers[1] + self.motor_powers[0])) * self.max_motor_thrust

        # Calculate acceleration (F = ma; a = F/m)
        acceleration_x = thrust_diff_x / self.mass
        acceleration_y = thrust_diff_y / self.mass

        # Update velocity (v = u + at)
        self.velocity[0] += acceleration_y * self.time_step - self.drag_coefficient * self.velocity[0] * self.time_step
        self.velocity[1] += acceleration_x * self.time_step - self.drag_coefficient * self.velocity[1] * self.time_step

        # Update position (s = s0 + vt)
        self.position[0] += self.velocity[0] * self.time_step
        self.position[1] += self.velocity[1] * self.time_step

    def get_status(self):
        """Returns the current status of the drone."""
        return {
            "position": self.position,
            "velocity": self.velocity,
        }
    
    def render(self, scale=1):
        position = copy(self.position) * scale
        color = 255 if not self.crashed else 0
        color = [color for i in range(3)]

        rect = pygame.Rect(position[1] - scale//2,
                           position[0] - scale//2,
                           scale,
                           scale,
                           )
        return rect, color