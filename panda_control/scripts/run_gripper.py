from panda_control.gripper_interface import GripperInterface
import argparse
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser("Gripper Interface")
    parser.add_argument("--general_cfg_file", default="/root/ws/3rd_party_libs/panda_control/config/charmander.yml",type=str, help="charmander.yml")
    args = parser.parse_args()
    gi = GripperInterface(general_cfg_file=args.general_cfg_file)
    try:
        while True:
            time.sleep(0.1)
    except Exception as ex:
        gi.close()
        exit()