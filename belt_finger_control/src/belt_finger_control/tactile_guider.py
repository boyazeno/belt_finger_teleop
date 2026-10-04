from belt_finger_control.tactile_sensor_reader import TactileSensorReader
import numpy as np


class TactileGuider():
    def __init__(self,sensor:TactileSensorReader, modes="y"):
        self.sensor = sensor
        self.modes = modes if isinstance(modes, list) else [modes]

    def start(self):
        # Implement the logic to close the gripper
        self.init_forces, self.init_locations, self.init_timestamp = self.sensor.get_latest()
        self.init_locations = self.init_locations[:,0] # only care x
        self.init_mag = np.linalg.norm(self.init_forces,axis=-1)
        pass

    def guide(self, forces, locations):
        mag = np.linalg.norm(forces,axis=-1)
        action = np.zeros(7)
        locations = locations[:,0]

        # x
        if "y" in self.modes:
            max_sum_mag = 2.0
            action[1] = -1*np.clip((mag.sum()-self.init_mag.sum())/max_sum_mag, -1, 1)
        # z
        if "z" in self.modes:
            max_mean_location_diff = 10.0
            action[2] = -1*np.clip((self.init_locations.mean()-locations.mean())/max_mean_location_diff, -1, 1)
        # rx
        if "ry" in self.modes:
            max_belt_location_diff = 10.0
            action[4] = -1*np.clip(((locations[1]-locations[0]) - (self.init_locations[1] - self.init_locations[0]))/max_belt_location_diff, -1, 1)
        # rz
        if "rz" in self.modes:
            max_belt_mag_diff = 2.0
            action[5] = -1*np.clip(((mag[1]-mag[0]) - (self.init_mag[1] - self.init_mag[0]))/max_belt_mag_diff, -1, 1)

        return action