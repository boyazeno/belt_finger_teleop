import os
import numpy as np
from dynamixel_sdk import *
import time
import subprocess
import re
import threading

if os.name == "nt":
    import msvcrt

    def getch():
        return msvcrt.getch().decode()

else:
    import sys, tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    

def get_device_port():
    pattern = r"ttyACM\d+"
    dmesg_info = subprocess.Popen("dmesg", stdout=subprocess.PIPE).stdout.read()
    dmesg_info = dmesg_info.decode("utf-8").splitlines()
    for l in dmesg_info[::-1]:
        if "ttyACM" in l:
            break
    else:
        print(f"[PORT ERROR] No ttyACM port find recently.")
        return None

    result = re.findall(pattern, l)
    if len(result) != 1:
        print(f"[PORT ERROR] Please check tty port, get {l}")
        return None
    return result[0]


class Gripper:
    def __init__(self, control_frequence:float=10.0, substeps:int=5) -> None:
        self.targ_pose = None
        self.substep = None
        self.cur_pose = None

        self.portHandler = None
        self.packetHandler = None
        self.substeps = substeps
        self.control_frequence = control_frequence
        self.is_terminated = False
        self.is_updated = False
        self.sub_step_thread = threading.Thread(target=self.run_substep)
        self.sub_step_thread.daemon = True
        self.RANGE_IN_METER = 0.122
        self.max_force_thres = 0.08
        pass

    def init(self):
        if not self.set_const():
            print("Fail to set const for gripper!")
            quit()

        # Open port
        try:
            self.portHandler.openPort()
            print("Succeeded to open the port")
        except:
            print("Failed to open the port")
            quit()

        # Set port baudrate
        try:
            self.portHandler.setBaudRate(self.BAUDRATE)
            print("Succeeded to change the baudrate")
        except:
            print("Failed to change the baudrate")
            quit()

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            quit()
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            quit()
        else:
            print("DYNAMIXEL has been successfully connected")
        print("Ready to get & set Position.")

    def reset(self):
        # close:
        delta = 3
        dt = 0.004
        load_thres = 0.08
        self.DXL_MINIMUM_POSITION_VALUE = -5000
        self.DXL_MAXIMUM_POSITION_VALUE = 5000
        cur_pose = self.get_present_pos(return_raw=True)
        cur_load = self.get_present_load()
        while cur_pose>self.DXL_MINIMUM_POSITION_VALUE:
            cur_pose -= delta
            self.move_to(position=cur_pose, is_raw=True)
            time.sleep(dt)
            cur_load = self.get_present_load()
            if cur_load < -1*load_thres:
                new_min = self.get_present_pos(return_raw=True)
                break

        while cur_pose<self.DXL_MAXIMUM_POSITION_VALUE:
            cur_pose += delta
            self.move_to(position=cur_pose, is_raw=True)
            time.sleep(dt)
            cur_load = self.get_present_load()
            if cur_load > load_thres:
                new_max = self.get_present_pos(return_raw=True)
                break
        self.DXL_MINIMUM_POSITION_VALUE = new_min
        self.DXL_MAXIMUM_POSITION_VALUE = new_max
        self.targ_pose = self.cur_pose = self.get_present_pos()
        self.substep = 0.0
        pass

    def start(self):
        self.sub_step_thread.start()
        print(f"Gripper control started!")

    def get_width(self)->float:
        return self.percentage_2_meter(self.cur_pose)
    
    def get_max_width(self)->float:
        return self.RANGE_IN_METER
    
    def get_grasp_state(self, thres:float=0.05):
        return abs(self.get_present_load())>thres

    def set_next_command(self, target:float):
        target = self.meter_2_percentage(target)
        self.targ_pose = target
        self.set_updated()

    def set_terminate(self, flag:bool=True):
        self.terminate = flag
    
    def set_updated(self, flag:bool=True):
        self.is_updated = flag

    def terminated(self)->bool:
        return self.is_terminated

    def updated(self)->bool:
        return self.is_updated
    
    def percentage_2_meter(self, percentage:float)->float:
        return percentage*self.RANGE_IN_METER

    def meter_2_percentage(self, meter:float)->float:
        return meter/self.RANGE_IN_METER

    def run_substep(self):
        p_tar = self.targ_pose = self.cur_pose = self.get_present_pos()
        self.substep = 0.0
        while not self.terminated():
            ts = time.time()

            if self.updated():
                p_tar = self.targ_pose
                self.substep = 0.0
                self.set_updated(False)

            self.cur_pose = self.get_present_pos()
            e = p_tar - self.cur_pose
            substep_percentage = min(self.substep / self.substeps, 1.0)
            new_target = self.cur_pose + e / (1 + np.exp(-substep_percentage * 12.0)) * 1.2
            cur_force = self.get_present_load()
            if abs(cur_force)<self.max_force_thres:
                self.move_to(new_target)
            elif (cur_force<-1*self.max_force_thres) and new_target>self.cur_pose:
                self.move_to(new_target)
            elif (cur_force>self.max_force_thres) and new_target<self.cur_pose:
                self.move_to(new_target)
            else:
                pass
            t = time.time() - ts
            time.sleep(np.maximum(1.0 / (self.control_frequence * self.substeps) - t - 0.00481, 0.0))
            self.substep += 1.0

    def set_new_target(self, target:float):
        self.targ_pose = target

    def set_const(self)->bool:
        # Control table address
        self.ADDR_TORQUE_ENABLE = 64  # Control table address is different in Dynamixel model
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PRESENT_LOAD = 126

        # Protocol version
        self.PROTOCOL_VERSION = 2.0  # See which protocol version is used in the Dynamixel

        # Default setting
        self.DXL_ID = 1  # Dynamixel ID : 1
        self.BAUDRATE = 57600 #57600  #115200 Dynamixel default baudrate : 57600
        device_port = get_device_port()
        if device_port is None:
            return False
        self.DEVICENAME = f"/dev/{device_port}"  # Check which port is being used on your controller
        print(f"[INFO] Get device name: {self.DEVICENAME}")

        self.TORQUE_ENABLE = 1  # Value for enabling the torque
        self.TORQUE_DISABLE = 0  # Value for disabling the torque
        self.DXL_MINIMUM_POSITION_VALUE = 73  # Dynamixel will rotate between this value
        self.DXL_MAXIMUM_POSITION_VALUE = 2543  # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
        self.DXL_MOVING_STATUS_THRESHOLD = 5  # Dynamixel moving status threshold

        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        return True

    def move_to(self, position:float, is_raw:bool=False):
        if not is_raw:
            position = position * (self.DXL_MAXIMUM_POSITION_VALUE - self.DXL_MINIMUM_POSITION_VALUE) + self.DXL_MINIMUM_POSITION_VALUE

        cliped_position = int(
            np.clip(
                position,
                a_min=self.DXL_MINIMUM_POSITION_VALUE,
                a_max=self.DXL_MAXIMUM_POSITION_VALUE,
            )
        )

        if cliped_position < 0:
            cliped_position = 65535 + cliped_position

        # print("Set Goal Position of ID %s = %s" % (1, cliped_position))
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, 1, self.ADDR_GOAL_POSITION, cliped_position
        )
        # print(f"[Result]: {dxl_comm_result}:{dxl_error}")

    def get_present_pos(self, return_raw:bool=False):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, 1, self.ADDR_PRESENT_POSITION
        )

        if dxl_present_position > 32767:
            dxl_present_position = dxl_present_position - 65535

        if return_raw:
            return dxl_present_position
        
        dxl_present_position = float(dxl_present_position - self.DXL_MINIMUM_POSITION_VALUE) / (
            self.DXL_MAXIMUM_POSITION_VALUE - self.DXL_MINIMUM_POSITION_VALUE
        )
        # print("Present Position of ID %s = %s" % (1, dxl_present_position))
        return dxl_present_position
    
    def get_present_load(self):
        dxl_present_load, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.portHandler, 1, self.ADDR_PRESENT_LOAD
        )
        if dxl_present_load > 32767:
            dxl_present_load = dxl_present_load - 65535
        dxl_present_load = dxl_present_load*0.001
        # print("Present Position of ID %s = %s" % (1, dxl_present_load))
        return dxl_present_load
    