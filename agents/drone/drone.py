import numpy as np
from copy import copy
import pygame


class QuadDrone:
    def __init__(self):
        self.pixel_size = 6
        self.motor_powers = np.zeros(4)  # Motor powers [0, 1] normalized
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32) # X, Y, Z position
        self.orientation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Roll, Pitch, Yaw angles in radians
        self.velocity = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # X, Y, Z velocity
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Roll, Pitch, Yaw rates
        self.mass = 0.5  # Mass of the drone in kilograms
        self.gravity = 9.81  # Gravitational acceleration (m/s^2)
        self.max_motor_thrust = 10.5  # Max thrust per motor in Newtons
        self.time_step = 0.01  # Time step for simulation in seconds

        self.crashed = False
        self.observation = None

    def set_motor_powers(self, powers):
        """Sets the power for each motor and updates the drone's state."""
        self.motor_powers = np.clip(np.array(powers), 0, 1)  # Ensure powers are within [0, 1]
        self.update_dynamics()

    def update_dynamics(self):
        # Compute thrust difference for X and Y directions
        thrust_diff_x = ((self.motor_powers[0] + self.motor_powers[2]) - (self.motor_powers[1] + self.motor_powers[3])) * self.max_motor_thrust
        thrust_diff_y = ((self.motor_powers[0] + self.motor_powers[1]) - (self.motor_powers[2] + self.motor_powers[3])) * self.max_motor_thrust

        # Calculate acceleration (F = ma; a = F/m)
        acceleration_x = thrust_diff_x / self.mass
        acceleration_y = thrust_diff_y / self.mass

        # Update velocity (v = u + at)
        self.velocity[0] += acceleration_x * self.time_step
        self.velocity[1] += acceleration_y * self.time_step

        # Update position (s = s0 + vt)
        self.position[0] += self.velocity[0] * self.time_step
        self.position[1] += self.velocity[1] * self.time_step



    def update_dynamics_complex(self):
        """Updates the drone's dynamics based on the current motor powers."""
        # Calculate total thrust
        total_thrust = np.sum(self.motor_powers) * self.max_motor_thrust

        roll, pitch, yaw = self.orientation

        thrust_vector = np.array([
            -np.sin(pitch) * total_thrust,
            np.sin(roll) * np.cos(pitch) * total_thrust,
            np.cos(roll) * np.cos(pitch) * total_thrust
        ])

        # Calculate acceleration
        acceleration = thrust_vector / self.mass - np.array([0, 0, self.gravity])

        # Update velocity
        self.velocity += acceleration * self.time_step

        # Update position
        self.position += self.velocity * self.time_step

        # Calculate torques based on motor powers (simplistic model)
        torque_roll = (self.motor_powers[0] - self.motor_powers[1]) * self.max_motor_thrust
        torque_pitch = (self.motor_powers[2] - self.motor_powers[3]) * self.max_motor_thrust
        torque_yaw = (self.motor_powers[0] + self.motor_powers[1] - self.motor_powers[2] - self.motor_powers[3]) * self.max_motor_thrust

        # Update angular velocities based on torques (simplistic model)
        self.angular_velocity[0] += torque_roll / self.mass * self.time_step
        self.angular_velocity[1] += torque_pitch / self.mass * self.time_step
        self.angular_velocity[2] += torque_yaw / self.mass * self.time_step

        # Update orientation based on angular velocities (simplistic model)
        self.orientation += self.angular_velocity * self.time_step

    def get_status(self):
        """Returns the current status of the drone."""
        return {
            "position": self.position,
            "orientation": self.orientation,
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity,
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