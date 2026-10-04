import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from copy import deepcopy
import threading


class TactileSensorReader(Node):
    def __init__(self):
        super().__init__('tactile_sensor_reader_node')
        
        # Create the subscriber (Message Type, Topic Name, Callback, Queue Size)
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'forces_locations_1',
            self.listener_callback,
            10
        )
        self._data_lock = threading.Lock()
        # Prevent unused variable warning
        self.subscription  
        self.get_logger().info("Sensor Reader Node is actively listening...")

        self.latest_average_force = None
        self.latest_average_location = None
        self.latest_timestamp = None

    def listener_callback(self, msg):
        # Safety check: Ensure we received exactly 13 floats
        if len(msg.data) != 13:
            self.get_logger().warning(f"Expected 13 floats, received {len(msg.data)}")
            return

        average_force = np.array(msg.data[:6]).reshape(2,3)
        average_location = np.array(msg.data[6:12]).reshape(2,3)
        timestamp = msg.data[-1]

        self.latest_average_force = average_force
        self.latest_average_location = average_location
        self.latest_timestamp = timestamp

    def get_latest(self):
        with self._data_lock:
            return self.latest_average_force, self.latest_average_location, self.latest_timestamp