import numpy as np

class GripperForceRegulator:
    def __init__(self, sensor_reader, kp:float = 0.001, kd:float=0.000, target_force_ratio:float=15.0):
        self.sensor_reader = sensor_reader
        self.kp = kp
        self.kd = kd
        self.target_force_ratio = target_force_ratio
        self.prev_opening = 0.08
        self.prev_force_error = 0.0
        pass

    def start(self, prev_opening):
        self.prev_opening = prev_opening   

    def regulate(self, target_opening:float, max_sum_force:float=1.0):

        forces, locations, timestamp = self.sensor_reader.get_latest()
        if forces is None:
            return target_opening
        
        force_magnitude = np.linalg.norm(forces, axis=-1)
        sum_force_magnitude = force_magnitude.sum()
        average_location = (locations[:,0]*force_magnitude).mean(axis=0)/sum_force_magnitude
        sum_force = forces.sum(axis=0)

        if target_opening > self.prev_opening:
            self.prev_opening = target_opening
            return target_opening, 0.0

        # Simple PD control based on the average force
        force_error = min(self.target_force_ratio*(0.08-target_opening),max_sum_force) - sum_force_magnitude
        opening_adjustment = self.kp * force_error

        opening_adjustment += self.kd * (force_error - getattr(self, "prev_force_error", 0.0))

        # D term (derivative) can be added if needed, but for simplicity we will skip it here
        new_opening = self.prev_opening - opening_adjustment
        new_opening = np.clip(new_opening, 0.0, 0.08)  # Assuming 0.08 is the max opening

        self.prev_force_error = force_error
        self.prev_opening = new_opening
        return new_opening, sum_force_magnitude