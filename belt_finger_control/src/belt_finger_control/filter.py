import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import time
from typing import Optional

class FilterDict:
    def __init__(self, filter_class, keys:list, filter_kwargs=None):
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs if filter_kwargs else {}
        self.filters = { key:filter_class(**self.filter_kwargs) for key in keys }

    def update(self, key, value):
        """
        Update the filter for a specific key with a new value.
        :param key: The key for which to update the filter.
        :param value: The new value to add to the filter.
        :return: The updated value from the filter.
        """
        if key not in self.filters:
            raise KeyError(f"Key '{key}' not found in filter dictionary.")
        return self.filters[key].update(value)
    
    def reset(self):
        """
        Reset all filters in the dictionary.
        """
        for key, filter_instance in self.filters.items():
            filter_instance.reset()


class MovingAverageFilter:
    def __init__(self, window_len=1):
        """
        Initialize the MovingAverageFilter with a specified window length.
        :param window_len: The number of values to consider for the moving average.
        """
        if window_len <= 0:
            raise ValueError("Window length must be greater than 0.")
        self.window_len = window_len
        self.reset()

    def update(self, new_value):
        """
        Update the filter with a new value and compute the moving average.
        :param new_value: The new value to add to the filter.
        :return: The updated moving average.
        """
        self.values.append(new_value)
        if len(self.values) > self.window_len:
            self.values.pop(0)
        return np.sum(np.stack(self.values,axis=0),axis=0) / len(self.values)

    def reset(self):
        """
        Reset the filter by clearing all stored values.
        """
        self.values = []

    
class ExponentialMovingAverageFilter:
    def __init__(self, alpha=0.8):
        """
        Initialize the ExponentialMovingAverageFilter with a specified smoothing factor.
        :param alpha: The smoothing factor, a value between 0 and 1.
        """
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.reset()

    def update(self, new_value):
        """
        Update the filter with a new value and compute the exponential moving average.
        :param new_value: The new value to add to the filter.
        :return: The updated exponential moving average.
        """
        if self.current_value is None:
            self.current_value = new_value
        else:
            self.current_value = self.alpha * new_value + (1 - self.alpha) * self.current_value
        return self.current_value

    def reset(self):
        """
        Reset the filter by clearing the current value.
        """
        self.current_value = None

class ExponentialMovingAveragePoseFilter:
    def __init__(self, alpha=0.5):
        """
        Initialize the ExponentialMovingAverageFilter with a specified smoothing factor.
        :param alpha: The smoothing factor, a value between 0 and 1.
        """
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.reset()

    def update(self, new_value):
        """
        Update the filter with a new value and compute the exponential moving average.
        :param new_value: The new value to add to the filter.
        :return: The updated exponential moving average.
        """
        if self.current_value is None:
            self.current_value = new_value
        else:
            # 1. Translation EMA
            t_prev = self.current_value[:3, 3]
            t_meas = new_value[:3, 3]
            t_filt = (1 - self.alpha) * t_prev + self.alpha * t_meas

            # 2. Rotation EMA via SLERP
            R_prev = R.from_matrix(self.current_value[:3, :3])
            R_meas = R.from_matrix(new_value[:3, :3])
            slerp = Slerp([0, 1], R.from_matrix([self.current_value[:3,:3], new_value[:3,:3]]))
            R_filt = slerp([self.alpha])[0]

            # 3. Reconstruct homogeneous transform
            self.current_value = np.eye(4)
            self.current_value[:3, :3] = R_filt.as_matrix()
            self.current_value[:3, 3]  = t_filt
        return self.current_value

    def reset(self):
        """
        Reset the filter by clearing the current value.
        """
        self.current_value = None


class OneEuroPoseFilter:
    """Adaptive One Euro filter for a 4x4 rigid-body pose.

    A fixed EMA adds the same delay during a fast deliberate motion as it
    does while the controller is stationary. This filter uses a low cutoff at
    low velocity to reject tracking jitter, then raises the cutoff as the
    translation or angular velocity increases. Translation is low-pass
    filtered in R^3 and rotation is updated along the shortest SO(3) rotation
    vector, so every output remains a valid rigid transform.

    ``translation_beta`` has units of Hz/(unit/s), and ``rotation_beta`` has
    units of Hz/(rad/s). Larger values make fast motions more responsive;
    larger minimum cutoffs make all motions more responsive but less smooth.
    """

    def __init__(self, translation_min_cutoff: float = 1.5,
                 translation_beta: float = 5.0,
                 rotation_min_cutoff: float = 1.5,
                 rotation_beta: float = 0.5,
                 derivative_cutoff: float = 1.0):
        values = (translation_min_cutoff, translation_beta,
                  rotation_min_cutoff, rotation_beta, derivative_cutoff)
        if any(value < 0.0 for value in values):
            raise ValueError("One Euro filter parameters must be non-negative.")
        if translation_min_cutoff == 0.0 or rotation_min_cutoff == 0.0:
            raise ValueError("Minimum cutoffs must be greater than zero.")
        if derivative_cutoff == 0.0:
            raise ValueError("Derivative cutoff must be greater than zero.")

        self.translation_min_cutoff = translation_min_cutoff
        self.translation_beta = translation_beta
        self.rotation_min_cutoff = rotation_min_cutoff
        self.rotation_beta = rotation_beta
        self.derivative_cutoff = derivative_cutoff
        self.reset()

    @staticmethod
    def _smoothing_factor(cutoff: float, dt: float) -> float:
        rate = 2.0 * np.pi * cutoff * dt
        return rate / (rate + 1.0)

    @classmethod
    def _low_pass(cls, previous: np.ndarray, measurement: np.ndarray,
                  cutoff: float, dt: float) -> np.ndarray:
        alpha = cls._smoothing_factor(cutoff, dt)
        return alpha * measurement + (1.0 - alpha) * previous

    def update(self, new_value: np.ndarray,
               timestamp: Optional[float] = None) -> np.ndarray:
        """Return the filtered pose. ``timestamp`` is in monotonic seconds."""
        pose = np.asarray(new_value, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("Pose must have shape (4, 4).")

        now = time.monotonic() if timestamp is None else timestamp
        if self.current_value is None:
            self.current_value = pose.copy()
            self._previous_measurement = pose.copy()
            self._previous_timestamp = now
            self._filtered_translation_velocity = np.zeros(3)
            self._filtered_angular_velocity = np.zeros(3)
            return self.current_value.copy()

        dt = now - self._previous_timestamp
        if dt <= np.finfo(float).eps:
            return self.current_value.copy()

        previous_measurement = self._previous_measurement
        translation_velocity = (pose[:3, 3] - previous_measurement[:3, 3]) / dt
        raw_rotation_delta = (
            R.from_matrix(previous_measurement[:3, :3]).inv()
            * R.from_matrix(pose[:3, :3])
        ).as_rotvec()
        angular_velocity = raw_rotation_delta / dt

        self._filtered_translation_velocity = self._low_pass(
            self._filtered_translation_velocity, translation_velocity,
            self.derivative_cutoff, dt)
        self._filtered_angular_velocity = self._low_pass(
            self._filtered_angular_velocity, angular_velocity,
            self.derivative_cutoff, dt)

        translation_cutoff = (
            self.translation_min_cutoff
            + self.translation_beta * np.linalg.norm(self._filtered_translation_velocity))
        rotation_cutoff = (
            self.rotation_min_cutoff
            + self.rotation_beta * np.linalg.norm(self._filtered_angular_velocity))
        filtered_translation = self._low_pass(
            self.current_value[:3, 3], pose[:3, 3], translation_cutoff, dt)

        current_rotation = R.from_matrix(self.current_value[:3, :3])
        target_rotation = R.from_matrix(pose[:3, :3])
        rotation_delta = (current_rotation.inv() * target_rotation).as_rotvec()
        rotation_alpha = self._smoothing_factor(rotation_cutoff, dt)
        filtered_rotation = current_rotation * R.from_rotvec(rotation_alpha * rotation_delta)

        self.current_value = np.eye(4)
        self.current_value[:3, :3] = filtered_rotation.as_matrix()
        self.current_value[:3, 3] = filtered_translation
        self._previous_measurement = pose.copy()
        self._previous_timestamp = now
        return self.current_value.copy()

    def reset(self):
        """Clear pose and velocity history."""
        self.current_value = None
        self._previous_measurement = None
        self._previous_timestamp = None
        self._filtered_translation_velocity = None
        self._filtered_angular_velocity = None

class MedianFilter:
    def __init__(self, window_len=3):
        """
        Initialize the MedianFilter with a specified window length.
        :param window_len: The number of values to consider for the median filter.
        """
        if window_len <= 0:
            raise ValueError("Window length must be greater than 0.")
        self.window_len = window_len
        self.reset()

    def update(self, new_value):
        """
        Update the filter with a new value and compute the median.
        :param new_value: The new value to add to the filter.
        :return: The updated median value.
        """
        self.values.append(new_value)
        if len(self.values) > self.window_len:
            self.values.pop(0)
        return  np.sort(np.stack(self.values,axis=0),axis=0)[len(self.values) // 2]

    def reset(self):
        """
        Reset the filter by clearing all stored values.
        """
        self.values = []


class ExtrapolationFilter:
    def __init__(self, window_len=5, alpha=0.5):
        """
        Initialize the ExtrapolationFilter with a specified window length and update factor.
        :param window_len: The number of values to consider for extrapolation.
        :param alpha: The update factor for incorporating new measurements (0 < alpha <= 1).
        """
        if window_len <= 1:
            raise ValueError("Window length must be greater than 1.")
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.window_len = window_len
        self.alpha = alpha
        self.reset()

    def update(self, new_value):
        """
        Update the filter with a new value and compute the extrapolated estimate.
        :param new_value: The new measured value.
        :return: The updated extrapolated value.
        """
        assert isinstance(new_value, (int, float)), "New value must be a number."
        self.values.append(new_value)
        if len(self.values) > self.window_len:
            self.values.pop(0)

        if len(self.values) < 2:
            # Not enough data points to extrapolate, return the current value
            self.estimate = new_value
        else:
            # Perform linear extrapolation based on the last two points
            x = list(range(len(self.values)))
            y = self.values
            # Perform least squares estimation to calculate the slope
            n = len(self.values)
            x_mean = sum(range(n)) / n
            y_mean = np.sum(self.values) / n
            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0
            self.estimate = y[-1] + slope

        # Blend the extrapolated estimate with the new measurement
        self.estimate = self.alpha * new_value + (1 - self.alpha) * self.estimate
        return self.estimate

    def reset(self):
        """
        Reset the filter by clearing all stored values and resetting the estimate.
        """
        self.values = []
        self.estimate = 0.0