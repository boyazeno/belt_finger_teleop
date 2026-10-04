import threading
import zmq
import time
from deoxys.utils.yaml_config import YamlConfig
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
import deoxys.proto.franka_interface.franka_robot_state_pb2 as franka_robot_state_pb2
from panda_control.gripper import Gripper

import rclpy
from std_msgs.msg import Float32MultiArray


def find_arduino_nano_port():
    import serial.tools.list_ports
    # Vendor and Product IDs for the QinHeng Electronics CH340 serial converter
    target_vid = 0x1A86
    target_pid = 0x7523
    
    # Get a list of all available serial ports
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Check if the port has VID and PID attributes and if they match our target
        if port.vid == target_vid and port.pid == target_pid:
            return port.device  # Returns the string like '/dev/ttyUSB1'
            
    return None # Return None if the device is not found


class GripperInterface:
    def __init__(
        self,
        general_cfg_file: str = "config/local-host.yml",
        control_timeout: float = 1.0,
        automatic_gripper_reset: bool=True,
    ):
        self._automatic_gripper_reset = automatic_gripper_reset
        self._control_timeout = control_timeout
        general_cfg = YamlConfig(general_cfg_file).as_easydict()
        self._control_freq = general_cfg.CONTROL.POLICY_RATE
        self._state_freq = general_cfg.CONTROL.STATE_PUBLISHER_RATE
        self._state_freq_t = 1.0/self._state_freq
        self._name = general_cfg.PC.NAME
        self._ip = general_cfg.PC.IP
        self._gripper_pub_port = general_cfg.NUC.GRIPPER_PUB_PORT
        self._gripper_sub_port = general_cfg.NUC.GRIPPER_SUB_PORT
        self._gripper = Gripper(control_frequence=self._control_freq, substeps=5)
        self._is_terminated = False
        self.init()

        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        self._subscriber = self._context.socket(zmq.SUB)

        self._publisher.bind(f"tcp://*:{self._gripper_pub_port}")
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._subscriber.connect(f"tcp://{self._ip}:{self._gripper_sub_port}")


        self._set_next_command_thread = threading.Thread(target=self.set_next_command)
        self._set_next_command_thread.daemon = True
        self._set_next_command_thread.start()

        self._state_pub_thread = threading.Thread(target=self.publish_state)
        self._state_pub_thread.daemon = True
        self._state_pub_thread.start()
        print(f"Gripper is ready for command")

    def init(self):
        print(f"Start initialization ...")
        self._gripper.init()
        if self._automatic_gripper_reset:
            self.reset_gripper()
        print(f"Initialization done")

    def reset_gripper(self):
        print(f"Reset gripper ...")
        self._gripper.reset()
        self._gripper.start()
        time.sleep(0.1)
        self._gripper.set_next_command(target=0.08)
        print(f"Reset gripper done")
        pass

    def move_gripper(self, target:float):
        self._gripper.set_next_command(target=target)

    def set_next_command(self, no_block: bool = False):
        if no_block:
            recv_kwargs = {"flags": zmq.NOBLOCK}
        else:
            recv_kwargs = {}
        while True:
            try:
                homing_msg = franka_controller_pb2.FrankaGripperStopMessage()
                move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
                grasp_msg = franka_controller_pb2.FrankaGripperGraspMessage()
                stop_msg = franka_controller_pb2.FrankaGripperStopMessage()

                gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()
                message = self._subscriber.recv(**recv_kwargs)
                gripper_control_msg.ParseFromString(message)
                if gripper_control_msg.control_msg.Unpack(homing_msg):
                    self.reset_gripper()
                elif gripper_control_msg.control_msg.Unpack(move_msg):
                    # print(f"Get new command at {time.time()}")
                    self.move_gripper(move_msg.width)
                    pass
                else:
                    print(f"Command skipped!")
            except:
                pass

    def publish_state(self):
        while not self.terminated():
            ts = time.time()
            
            gripper_state_msg = franka_robot_state_pb2.FrankaGripperStateMessage()
            gripper_state_msg.width = self._gripper.get_width()
            gripper_state_msg.max_width = self._gripper.get_max_width()
            gripper_state_msg.is_grasped = self._gripper.get_grasp_state()
            gripper_state_msg.temperature = 0 #TODO
            self._publisher.send(gripper_state_msg.SerializeToString())
            time.sleep(max(self._state_freq_t-time.time()+ts-0.0001, 0.0))

        pass

    def set_terminate(self, flag:bool):
        self._is_terminated = flag
        self._gripper.set_terminate(self._is_terminated)
        
    def terminated(self)->bool:
        return self._is_terminated
    
    def close(self):
        self._state_pub_thread.join(1.0)
        self._set_next_command_thread.join(1.0)


class BeltGripperInterface(GripperInterface):
    def __init__(
        self,
        general_cfg_file: str = "config/local-host.yml",
        control_timeout: float = 1.0,
        automatic_gripper_reset: bool=True,
        ros_node=None,
        use_serial=True,
    ):
        self._automatic_gripper_reset = automatic_gripper_reset
        self._control_timeout = control_timeout
        general_cfg = YamlConfig(general_cfg_file).as_easydict()
        self._control_freq = general_cfg.CONTROL.POLICY_RATE
        self._state_freq = general_cfg.CONTROL.STATE_PUBLISHER_RATE
        self._state_freq_t = 1.0/self._state_freq
        self._name = general_cfg.PC.NAME
        self._ip = general_cfg.PC.IP
        self._gripper_pub_port = general_cfg.NUC.GRIPPER_PUB_PORT
        self._gripper_sub_port = general_cfg.NUC.GRIPPER_SUB_PORT
        self._gripper = Gripper(control_frequence=self._control_freq, substeps=5)
        self._is_terminated = False
        self.init()

        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        self._subscriber = self._context.socket(zmq.SUB)

        self._publisher.bind(f"tcp://*:{self._gripper_pub_port}")
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._subscriber.connect(f"tcp://{self._ip}:{self._gripper_sub_port}")


        self._set_next_command_thread = threading.Thread(target=self.set_next_command)
        self._set_next_command_thread.daemon = True
        self._set_next_command_thread.start()

        self._state_pub_thread = threading.Thread(target=self.publish_state)
        self._state_pub_thread.daemon = True
        self._state_pub_thread.start()

        # Setup communication with the arduino
        self._use_serial = use_serial
        if self._use_serial: # Send commands via serial 
            import serial
            self.ser = serial.Serial(find_arduino_nano_port(), 115200 , timeout=1) #115200
        else: # Send commands via ROS2 topic
            if ros_node is not None:
                self.node = ros_node
            else:
                rclpy.init()
                self.node = rclpy.create_node('belt_gripper_interface')
            self.pub = self.node.create_publisher(Float32MultiArray, '/conveyer_finger', 10)
        time.sleep(0.5)
        print(f"Gripper is ready for command")

    def set_next_command(self, no_block: bool = False):
        if no_block:
            recv_kwargs = {"flags": zmq.NOBLOCK}
        else:
            recv_kwargs = {}
        while True:
            try:
                homing_msg = franka_controller_pb2.FrankaGripperStopMessage()
                move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
                belt_move_msg = franka_controller_pb2.BeltGripperMoveMessage()

                gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()
                message = self._subscriber.recv(**recv_kwargs)
                gripper_control_msg.ParseFromString(message)
                if gripper_control_msg.control_msg.Unpack(homing_msg):
                    self.reset_gripper()
                elif gripper_control_msg.control_msg.Unpack(belt_move_msg):
                    transy, rotz, rotx = belt_move_msg.transy, belt_move_msg.rotz, belt_move_msg.rotx
                    opening = belt_move_msg.opening

                    # keep only 5 digits
                    transy_5 = round(transy, 5)
                    rotz_5 = round(rotz, 5)
                    rotx_5 = round(rotx, 5)

                    if self._use_serial:
                        command_str = f"{transy_5:.5f},{rotz_5:.5f},{rotx_5:.5f}\n"
                        self.move_gripper(opening)
                        print(f"Sending command to belt: {command_str}")
                        self.ser.write(command_str.encode('utf-8'))
                    else:
                        msg = Float32MultiArray()
                        msg.data = [transy_5, rotz_5, rotx_5, opening]
                        self.move_gripper(opening)
                        self.pub.publish(msg)
                else:
                    print(f"Command skipped!")
            except:
                pass