from panda_control.customized_franka_interface import CustomizedFrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger
import numpy as np
import time
import os
from panda_control import CONFIG_DIR, DEFAULT_INTERFACE_CFG

controller_cfg = os.path.join(CONFIG_DIR, "joint_impedance_controller.yml")
interface_cfg = DEFAULT_INTERFACE_CFG
logger = get_deoxys_example_logger()
robot_interface = CustomizedFrankaInterface(
    interface_cfg, use_visualizer=False
)
time.sleep(1.0)

for i in range(1000):
    robot_interface.gripper_control(action=0.08*(np.cos(np.pi*(i/1000.0-0.5))+1.0)*0.5)
    time.sleep(0.05)

print(len(robot_interface._gripper_state_buffer))