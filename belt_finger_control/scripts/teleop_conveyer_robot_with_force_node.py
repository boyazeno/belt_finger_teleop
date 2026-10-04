import rclpy
import numpy as np
from belt_finger_control import JoystickModeDetector
from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG
from panda_control.robot import Robot
from panda_control.customized_franka_interface import CustomizedBeltFrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from belt_finger_control.tactile_sensor_reader import TactileSensorReader
from belt_finger_control.gripper_force_regulator import GripperForceRegulator
import time
import threading

class ROSJoystickModeDetector():
    def __init__(self,controller_cfg_dir = CONFIG_DIR, interface_cfg = DEFAULT_INTERFACE_CFG):
        super().__init__()
        self.controller_cfg_dir =controller_cfg_dir
        self.detector = JoystickModeDetector()
        self.robot_interface = CustomizedBeltFrankaInterface(
                interface_cfg, use_visualizer=False,automatic_gripper_reset=False
            )
        valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE"]
        self.controller_cfg_dict = {k:YamlConfig(f"{self.controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}


    def start(self, force_regulator):
        self.detector.start()
        np.set_printoptions(precision=3, suppress=True)
        while True:
            try:
                arm_action, gripper_action = self.detector.control()
                special_button_state = self.detector.joystick_controller.get_special_botton_state()
                if special_button_state.button_y:
                    break
                action = np.concatenate([arm_action, gripper_action],axis=0)
                action[0:3] *= 200
                action[3:6] *= 75
                if special_button_state.tactile:
                   prev_opening = action[6]
                   action[6], cur_force_mag =force_regulator.regulate(action[6])
                   print(f"Regulating gripper force based on tactile sensing... {prev_opening}-> {action[6]}: {cur_force_mag}N")
                print(f"action is: {action}")
                self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
                # rate.sleep()
            except KeyboardInterrupt:
                print("Quitting...")
                exit(0)




            

if __name__ == "__main__":
    rclpy.init()
    sensor_reader = TactileSensorReader()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor_reader,), daemon=True)
    spin_thread.start()
    force_regulator = GripperForceRegulator(sensor_reader)
    ROSJoystickModeDetector().start(force_regulator=force_regulator)