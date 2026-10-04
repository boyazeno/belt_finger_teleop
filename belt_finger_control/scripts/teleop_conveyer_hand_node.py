import time
import serial
# from belt_finger_control import TeleoperationModeDetector
from belt_finger_control import JoystickModeDetector
from belt_finger_control import Gripper
import numpy as np

def getch():
    import sys, termios, tty

    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)  # or tty.setraw(fd) if you prefer raw mode's behavior.
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, orig)

class ROSTeleoperationModeDetector():
    def __init__(self):
        super().__init__()

        self.rate = 20

        self.should_end = False

        # self.input_thread = threading.Thread(target=self.cli_input, daemon=True)
        # self.input_thread.start()

        # self.detector = TeleoperationModeDetector()
        self.detector = JoystickModeDetector(control_arm=False)
        self.gripper = Gripper(control_frequence=self.rate, substeps=5)
        self.gripper.init()
        self.gripper.reset()
        self.gripper.start()
        time.sleep(0.1)
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

    def start(self):
        self.detector.start()
        np.set_printoptions(precision=3, suppress=True)
        while True:
            arm_action, gripper_action = self.detector.control()
            special_button_state = self.detector.joystick_controller.get_special_botton_state()
            if special_button_state.button_y:
                break

            opening, transy, rotz, rotx = gripper_action[0],gripper_action[1],gripper_action[2],gripper_action[3]
            if self.should_end:
                transy,rotz,rotx = 0.0,0.0,0.0
                opening = 0.08

            # Send command to hand base
            self.gripper.set_next_command(target=opening)

            # Send command to fingers
            # keep only 5 digits
            transy_5 = round(transy, 5)
            rotz_5 = round(rotz, 5)
            rotx_5 = round(rotx, 5)
            command_str = f"{transy_5:.5f},{rotz_5:.5f},{rotx_5:.5f}\n"
            self.ser.write(command_str.encode('utf-8'))

            time.sleep(1.0 / self.rate)

    def cli_input(self):
        import sys
        while True:
            try:
                print("Press Q to exist, R to reset, E to end, B to begin")
                user_input = getch()
                if user_input.lower() == 'q':
                    print("User requested shutdown")
                    sys.exit(0)
                elif user_input.lower() == 'r':
                    self.should_reset = True
                    print("Resetting mode detector...")
                elif user_input.lower() == 'e':
                    self.should_end = True
                    print("End mode activated")
                elif user_input.lower() == 'b':
                    self.should_end = False
                    print("Begin mode activated")
                else:
                    print("Invalid input, please use 'r', 'q','e','b'")
                time.sleep(0.2)  # 5Hz rate equivalent
            except (KeyboardInterrupt, SystemExit):
                print("\nInput thread shutting down")
                break
            

if __name__ == "__main__":
    ROSTeleoperationModeDetector().start()