import numpy as np
from copy import copy
import pygame


class QuadDrone:
    def __init__(self):
        self.pixel_size = 6
        self.motor_powers = np.zeros(4)  # Motor powers [0, 1] normalized
        self.position = np.array([0.0, 0.0], dtype=np.float32) # Y, Z position
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # Y, Z velocity
        self.mass = 0.5  # Mass of the drone in kilograms
        self.max_motor_thrust = 10.5  # Max thrust per motor in Newtons
        self.time_step = 0.01  # Time step for simulation in seconds
        self.drone = None

        self.crashed = False
        self.observation = None
        self.heatmap = None

    def reset(self,):
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
        self.velocity[0] += acceleration_y * self.time_step
        self.velocity[1] += acceleration_x * self.time_step

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
        color = [np.clip(255, 0, 255) for i in range(3)]

        rect = pygame.Rect(position[1] - scale//2,
                           position[0] - scale//2,
                           scale,
                           scale,
                           )
        return rect, color