import logging
import time
import numpy as np
from typing import Union

import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
import deoxys.proto.franka_interface.franka_robot_state_pb2 as franka_robot_state_pb2
from deoxys.franka_interface import FrankaInterface
logger = logging.getLogger(__name__)


class CustomizedFrankaInterface(FrankaInterface):
    def get_gripper_state(self):
        while True:
            try:
                franka_gripper_state = (
                    franka_robot_state_pb2.FrankaGripperStateMessage()
                )
                message = self._gripper_subscriber.recv()
                franka_gripper_state.ParseFromString(message)
                self._gripper_state_buffer.append(franka_gripper_state)
            except:
                pass

    def gripper_control(self, action: float, speed: float=0.1):
        """Control the gripper

        Args:
            action (float): The control command for Franka gripper. Unit is abs opening size
        """

        gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()

        # will stop executing the previous command
        move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
        move_msg.width = action
        move_msg.speed = speed #! deprecated
        # print(f"Set new command at {time.time()}")
        gripper_control_msg.control_msg.Pack(move_msg)
        
        logger.debug(f"Gripper move to {move_msg.width}")

        self._gripper_publisher.send(gripper_control_msg.SerializeToString())
        self.last_gripper_action = action


class CustomizedBeltFrankaInterface(CustomizedFrankaInterface):
    def __init__(self,
                general_cfg_file: str = "config/local-host.yml",
                control_freq: float = 20.0,
                state_freq: float = 100.0,
                control_timeout: float = 1.0,
                has_gripper: bool = False,
                use_visualizer: bool = False,
                automatic_gripper_reset: bool=True,):
        super().__init__(general_cfg_file=general_cfg_file,
                         control_freq=control_freq,
                        state_freq=state_freq,
                        control_timeout=control_timeout,
                        has_gripper=False, # Jump over auto control 
                        use_visualizer=use_visualizer,
                        automatic_gripper_reset=automatic_gripper_reset)
        self.last_gripper_action = None
        self.last_gripper_dim = -4

    def reset(self):
        super().reset()
        self.last_gripper_action = None
        self.last_gripper_dim = -4


    def control(
            self,
            controller_type: str,
            action: Union[np.ndarray, list],
            controller_cfg: dict = None,
            termination: bool = False,
        ):
        super().control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
            termination=termination,
        )
        self.gripper_control(action=action[self.last_gripper_dim:])

        
    
    def gripper_control(self, action: np.ndarray):
        """Control the gripper

        Args:
            action (float): The control command for Franka gripper. Unit is abs opening size
        """

        gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()

        # will stop executing the previous command
        move_msg = franka_controller_pb2.BeltGripperMoveMessage()
        move_msg.opening = action[0]
        move_msg.transy = action[1]
        move_msg.rotz = action[2]
        move_msg.rotx = action[3]

        gripper_control_msg.control_msg.Pack(move_msg)
        
        logger.debug(f"Belt Gripper move to {move_msg.opening}, transy: {move_msg.transy}, rotz: {move_msg.rotz}, rotx: {move_msg.rotx}")

        self._gripper_publisher.send(gripper_control_msg.SerializeToString())
        self.last_gripper_action = action[self.last_gripper_dim:]