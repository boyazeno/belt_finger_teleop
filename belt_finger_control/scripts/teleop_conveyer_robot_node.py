import numpy as np
from belt_finger_control import JoystickModeDetector
from belt_finger_control import VRModeDetector
from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG
from panda_control.robot import Robot
from panda_control.customized_franka_interface import CustomizedBeltFrankaInterface
from deoxys.utils import YamlConfig, transform_utils
import time

class ROSJoystickModeDetector():
    def __init__(self,controller_cfg_dir = CONFIG_DIR, interface_cfg = DEFAULT_INTERFACE_CFG):
        super().__init__()
        self.controller_cfg_dir =controller_cfg_dir
        self.detector = VRModeDetector() #JoystickModeDetector()
        self.robot_interface = CustomizedBeltFrankaInterface(
                interface_cfg, use_visualizer=False,automatic_gripper_reset=False
            )
        valid_controller_type = ["JOINT_POSITION","OSC_POSE","JOINT_IMPEDANCE"]
        self.controller_cfg_dict = {k:YamlConfig(f"{self.controller_cfg_dir}/{k.lower()}_controller.yml").as_easydict() for k in valid_controller_type}


    def start(self):
        self.detector.start()
        np.set_printoptions(precision=3, suppress=True)
        while True:
            try:
                arm_action, gripper_action = self.detector.control()
                special_button_state = self.detector.joystick_controller.get_special_botton_state()
                if special_button_state.button_y:
                    break
                if special_button_state.start:
                    print("Start button pressed! !!!!!!!!!!!!!!!!")
                arm_action[0:3] *= 50
                arm_action[3:6] *= 3
                action = np.concatenate([arm_action, gripper_action],axis=0)
                print(f"action is: {action}")
                self.robot_interface.control(controller_type="OSC_POSE", action=action, controller_cfg=self.controller_cfg_dict["OSC_POSE"])
                # time.sleep(1/20.0)
            except KeyboardInterrupt:
                print("Quitting...")
                exit(0)




            

if __name__ == "__main__":
    ROSJoystickModeDetector().start()