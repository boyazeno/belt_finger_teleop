import matplotlib.pyplot as plt
import torch
from collections import deque

from PIL import Image
import cv2
import numpy as np
import time
import threading

from typing import Dict
import pyrealsense2 as rs
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing

class Camera():
    def __init__(self, serial_number:str = "846112072071") -> None: #!Change serial number to in hand camera
        super().__init__()
        self._pipeline = None
        self._align = None

        self.serial_number = serial_number

    def start_camera(self)->None:
        # Get serial number
        # Configure the RealSense camera pipeline
        self._pipeline = rs.pipeline()
        config = rs.config()
        ctx = rs.context()
        devices = ctx.query_devices()
        if devices.size()<0:
            raise RuntimeError(f"No realsense found!")
        for device in devices:
            if self.serial_number == device.get_info(rs.camera_info.serial_number):
                break
        else:
            raise RuntimeError(f"No camera found: {self.serial_number}")
        
        config.enable_device(self.serial_number)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 60)

        # Start the pipeline
        self._pipeline.start(config)
        align_to = rs.stream.color
        self._align = rs.align(align_to)

    def capture_vision_data(self) -> Dict:
        if self._pipeline is None:
            raise RuntimeError(f"Camera should be started before.")
        
        # Wait for a coherent pair of frames: depth and color
        frames = self._pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert the depth image to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the color image to a numpy array
        color_image = np.asanyarray(color_frame.get_data())


        # Get the camera info and convert it to a ROS message
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array([intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1]).reshape(3,3) #! original K, not crop one

        # Convert the depth image to a point cloud and publish it
        pc = rs.pointcloud()
        pc_points = pc.calculate(depth_frame)
        vertices = np.asanyarray(pc_points.get_vertices()).view(np.float32)
        # return cv2.resize(color_image[:,320-240:320+240, :],(40,40)), cv2.resize(depth_image[:,320-240:320+240]/1000.0,(40,40)), vertices, K
        return cv2.resize(color_image[:,212-120:212+120, :],(40,40)), cv2.resize(depth_image[:,212-120:212+120]/1000.0,(40,40)), vertices, K

    def __del__(self):
        if self._pipeline:
            self._pipeline.stop()

@dataclass
class VisionState:
    time_stamp: float 
    rgb: np.ndarray 
    depth: np.ndarray 
    intrinsics: np.ndarray

class VisionPerceptor:
    def __init__(self, buffer_size:int=1) -> None:
        self.manager = multiprocessing.Manager()
        self.buffer = self.manager.Queue(maxsize=buffer_size)
        self.thread = None
        self.is_killing = False
        pass


    def start(self):
        # start thread to streaming the camera
        if self.thread is not None and self.thread.is_alive():
            print(f"[Vision Perceptor] a thread is running! Don't restart a new thread.")
        self.thread = multiprocessing.Process(target=self.thread_get_vision_info, args=[self.buffer],daemon=True)

        self.thread.start()
        pass

    def thread_get_vision_info(self, buffer):
        print("Initialize Camera")
        camera = Camera()
        camera.start_camera()
        print("Camera initialized")
        while not self.is_killing:
            rgb = None
            depth = None
            intrinsics = None
            
            rgb, depth, pc, intrinsics = camera.capture_vision_data()
            
            # print(f"[Vision Perceptor] Get a new vision state!")

            try:
                buffer.get_nowait() # remove old one if has
            except:
                pass

            # print(f"[Vision Perceptor] Put a new vision state!")
            buffer.put(VisionState(
                time_stamp = time.time(),
                rgb = rgb,
                depth = depth,
                intrinsics = intrinsics,
            ))
            # print(f"[Vision Perceptor] Done Put a new vision state!")
        del camera

    @property
    def last_state(self)->VisionState:
        if self.thread is None:
            raise RuntimeError("[Vision Perceptor] No thread is running!")
        try:
            data = self.buffer.get()
            if (delta_t:= (time.time()-data.time_stamp))>0.02:
                print(f"[Vision Perceptor] Warning: Vision state too old [{delta_t}]")
            return deepcopy(data), data.time_stamp
        except:
            raise ValueError("[Vision Perceptor] No buffer available at the moment!")
        

    def __del__(self):
        self.is_killing = True

        if self.thread is not None:
            self.thread.join()
        self.manager.shutdown()
        del self.camera


if __name__ == "__main__":
    vision_perceptor = VisionPerceptor()
    vision_perceptor.start()
    time.sleep(1.0)
    while True:
        time.sleep(0.05)
        state, time_stamp = vision_perceptor.last_state
        print(time_stamp)