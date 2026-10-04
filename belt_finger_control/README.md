# How to use:

Structure:

belt_finger_control --> panda_control --> deoxys_control

(teleoperation robot and gripper use camera and xbox360 controller) --> (customized franka interface with belt gripper interface) --> (protobuf msg and robot control node) 

All commands below are run from the repository root inside the dev container, where the Python environment (`/opt/venv`) is already active.

## Teleoperate Belt Gripper With Robot

### Release the safety button
Make sure the robot's safety (stop) button is released before starting.

### Connect Camera and Xbox360 controller via usb
* change `serial_number` in `teleoperation_mode_detector.py`

### Start Gripper:
* connect both the Arduino and the gripper motor to the PC via two USB cables

* start gripper interface node
```sh
cd panda_control/scripts
python3 run_belt_gripper.py
```

### Start Robot Interface:
```sh
cd /root/ws/3rd_party_libs/deoxys_control/deoxys  # deoxys location in the dev container
auto_scripts/auto_arm.sh /root/ws/3rd_party_libs/panda_control/config/charmander.yml /root/ws/3rd_party_libs/panda_control/config/control_config.yml
```

### Start Teleoperation of the whole robot
```sh
cd belt_finger_control/scripts
python3 teleop_conveyer_robot_node.py
```

### Key Mapping
**A:** switch pause/resume

**B:** reset

**Y:** quit

**X:** recalibrate markers

**LT:** opening

**L axis-x:** translation along finger

**R axis-x:** rotation around finger moving axis 

**R axis-y:** rotation around finger longtitude axis

You can also start belt finger with force regulator:
```sh
cd belt_finger_control/scripts
python3 teleop_conveyer_robot_with_force_node.py
```


## Teleoperate Belt Gripper Only
### Start Gripper:
* connect both arduino and gripper motor through 2 usb cable with pc
* start gripper node
```sh
cd belt_finger_control
python3 gripper_node.py
```
**RB:** enable force regulator


### Connect Camera:
* connect realsense camera  with usb3 cable with pc
* change the `self.serial_number` of it in `teleoperation_mode_detector.py` to real serial number
* put the non-electronic controller under the camera

### Start Teleoperation
```sh
cd belt_finger_control/scripts
python3 teleop_conveyer_hand_node.py
```

### Data recording
For recording trajectories for imitation learning, you need to run:

```sh
cd belt_finger_control/scripts
python3 record_play_dataset.py
```

This inherit all key mapping from previous with extra for start and end recording:

**start:** click once to start or end recording one episode (The start vibrates stronger, stop weaker).

After recording, the trajectories are saved as pickle file with state-action pair with class type RobotState and RobotAction.

To further convert to LerobotDataset V3.0 format, please refer to 
```sh
cd scripts/run_policy
python3 convert_record_trajectory_to_lerobot.py
```

For training baseline model and inference server, please use the singularity file in repo `belt_finger_vla`

# VR teleop

## Connect Quest 3S to PC
0. Install ADB on PC
For connection
```
sudo apt update
sudo apt install android-tools-adb -y
```

1. Download vpn
Go to [Genymobile/gnirehtet GitHub Releases](https://github.com/Genymobile/gnirehtet/releases), download gnirehtet-rust-linux***.zip and unzip it.

2. Connect quest
Connect the device with usb, then run:
```
adb devices
```
This will pop usb debug window in quest and need to accept.

3. Install vpn on quest
```
adb install gnirehtet.apk
```

4. Connect VPN & forward port:
Run this on pc and keep it open, accept the use vpn popup in quest.
```
./gnirehtet run
```
In new terminal, reverse forward port from quest to pc (this need to do everytime you unplug and replug the quest from the pc):
```
adb reverse tcp:8443 tcp:8443
```

* Note, start vr app is slow somehow with usb in the beginning.
