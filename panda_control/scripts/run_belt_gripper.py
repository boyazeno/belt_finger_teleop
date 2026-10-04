from panda_control.gripper_interface import BeltGripperInterface
import argparse
import signal
import rclpy

if __name__=="__main__":
    parser = argparse.ArgumentParser("Gripper Interface")
    parser.add_argument("--general_cfg_file", default="/root/ws/3rd_party_libs/panda_control/config/charmander.yml",type=str, help="charmander.yml")
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('teleop_conveyer_finger')
    gi = BeltGripperInterface(general_cfg_file=args.general_cfg_file, ros_node=node, use_serial=True)
    try:
        while True:
            signal.pause()
            # time.sleep(0.1)
    except Exception as ex:
        gi.close()
        exit()