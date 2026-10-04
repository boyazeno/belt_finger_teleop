import rclpy
import numpy as np
from belt_finger_control import JoystickModeDetector
from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG
from panda_control.robot import Robot
from panda_control.customized_franka_interface import CustomizedBeltFrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from belt_finger_control.tactile_guider import TactileGuider
from belt_finger_control.tactile_sensor_reader import TactileSensorReader
import time
import threading

class ROSJoystickModeDetector():
    def __init__(self,sensor_reader, controller_cfg_dir = CONFIG_DIR, interface_cfg = DEFAULT_INTERFACE_CFG):
        super().__init__()

        self.controller_cfg_dir =controller_cfg_dir
        self.robot_interface = CustomizedBeltFrankaInterface(
                interface_cfg, use_visualizer=False,automatic_gripper_reset=False
            )
        valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE", "CARTESIAN_VELOCITY"]
        self.controller_cfg_dict = {k:YamlConfig(f"{self.controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}
        self.sensor_reader = sensor_reader
        time.sleep(1) # Wait for the first tactile message
        self.tactile_guider = TactileGuider(sensor=self.sensor_reader, modes=["y","z"]) #["y","z","rz"])


    def start(self):
        np.set_printoptions(precision=3, suppress=True)

        action = np.zeros(6+1+3)
        action[6] = 0.04
        self.robot_interface.control(controller_type="CARTESIAN_VELOCITY", action=action, controller_cfg=self.controller_cfg_dict["CARTESIAN_VELOCITY"])
        # self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
        opening = input("Press Enter to start tactile guiding...}")
        opening = np.clip(0.003, 0.0,0.08)
        # opening = np.clip(float(opening), 0.0,0.08)
        action[6] = opening
        self.robot_interface.control(controller_type="CARTESIAN_VELOCITY", action=action, controller_cfg=self.controller_cfg_dict["CARTESIAN_VELOCITY"])
        # self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
        time.sleep(4.0)
        self.tactile_guider.start()
        poses = []
        timestamps = []
        while True:
            try:
                forces, locations, timestamp = self.sensor_reader.get_latest()
                arm_action = self.tactile_guider.guide(forces=forces, locations=locations)
                arm_action[-1] = opening
                gripper_action = np.zeros(3)
                action = np.concatenate([arm_action, gripper_action],axis=0)
                action[0:3] *= 0.03
                action[3:6] *= 0.06
                print(f"action is: {action}")
                self.robot_interface.control(controller_type="CARTESIAN_VELOCITY", action=action, controller_cfg=self.controller_cfg_dict["CARTESIAN_VELOCITY"])
                poses.append(self.robot_interface.last_eef_pose)
                timestamps.append(timestamp)
                # self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
                # rate.sleep()
            except Exception as e:
                print(f"Error occurred: {e}")
                exit(0)
            finally:
                np.savez("guided_circle.npz", poses=poses, timestamps=timestamps)




            

if __name__ == "__main__":
    rclpy.init()
    sensor_reader = TactileSensorReader()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor_reader,), daemon=True)
    spin_thread.start()
    ROSJoystickModeDetector(sensor_reader=sensor_reader).start()