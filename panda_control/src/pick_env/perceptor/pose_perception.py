import matplotlib.pyplot as plt
import torch
from collections import deque

from PIL import Image
import numpy as np
import time
import threading

from typing import Dict
import pyrealsense2 as rs
from copy import deepcopy
from dataclasses import dataclass
import sys 
import os
import trimesh
import cv2
from pick_env.perceptor.utils import SAMSegmentor
from panda_control import EXTERNAL_HANDEYE_CALIBRATION_FILE
# FoundationPose is not pip-installable; set FOUNDATIONPOSE_ROOT to a local clone of it.
sys.path.append(os.environ.get("FOUNDATIONPOSE_ROOT", "FoundationPose"))
from estimater import *
from datareader import *
import logging
import yaml

class Camera():
    def __init__(self, serial_number:str = "739112060472") -> None:
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
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)


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
        return color_image, depth_image/1000.0, vertices, K

    def __del__(self):
        if self._pipeline:
            self._pipeline.stop()

@dataclass
class PoseState:
    time_stamp: float 
    pose: np.ndarray
    rgb: np.ndarray 
    depth: np.ndarray 
    intrinsics: np.ndarray

class PosePerceptor:
    def __init__(self, buffer_size:int=5, log_dir:str="", mesh_path:str=None) -> None:
        print("Initialize Camera")
        self.camera = Camera()
        self.camera.start_camera()
        print("Camera initialized")

        self.log_dir = log_dir
        self.init(mesh_path=mesh_path)
        self.buffer = deque(maxlen=buffer_size)
        self.thread = None
        self.is_killing = False
        self.is_first = True
        pass

    def init(self, mesh_path:str):
        print(f"Load mesh from {mesh_path}")
        self.mesh = trimesh.load(mesh_path)
        self.mesh_to_origin, self.mesh_extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.mesh_bbox = np.stack([-self.mesh_extents/2, self.mesh_extents/2], axis=0).reshape(2,3)

        print(f"Set logging format")
        FORMAT = '[%(funcName)s()] %(message)s'
        logging.basicConfig(level=logging.WARNING, format=FORMAT)

        print(f"Load Calibration Data")
        with open(EXTERNAL_HANDEYE_CALIBRATION_FILE, 'r') as file:
            config = yaml.safe_load(file)["transform_camera_2_ee"]
            self.tf_robot_2_external_camera = np.array(config).reshape(4,4).T

        print(f"Init SAM2")
        self.segmenter = SAMSegmentor()

        print(f"Init Foundation Pose")
        debug = 0
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.log_dir, debug=debug, glctx=self.glctx)

    def get_init_mask(self, rgb, color=(42,62,15), color_tolerance:int=5): #! change color to real color
        # lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        # target_color_lab = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2Lab)[0][0]
        lab_image = rgb
        target_color_lab = np.array(color)[np.newaxis,np.newaxis]

        # Calculate the distance from the target color
        diff = np.abs(lab_image - target_color_lab.reshape(1,1,-1))
        diff = np.sqrt(np.sum(diff**2, axis=2))
        
        polygon = [[108,36],[407,51],[427,370],[30,332]] #! Table area
        hand_polygon = [[316,10],[409,10],[349,165],[284,155]]
        valid_area = np.ones_like(diff, dtype=np.float32)
        non_valid_area = np.ones_like(diff, dtype=np.float32)
        valid_area = cv2.fillPoly(valid_area, [np.array(polygon)], 0)
        non_valid_area = cv2.fillPoly(non_valid_area, [np.array(hand_polygon)], 0)
        valid_area = (~valid_area.astype(bool)) & non_valid_area.astype(bool)
        gaussian_blur = cv2.GaussianBlur(diff, (15, 15), 0, 0)
        gaussian_blur = cv2.GaussianBlur(gaussian_blur, (15, 15), 0, 0)
        gaussian_blur[~valid_area] = np.inf
        pix = np.unravel_index(np.argmin(gaussian_blur), gaussian_blur.shape)
        pix = np.array([pix[1],pix[0]]).reshape(1,-1)

        mask = self.segmenter.segment(img=rgb, key_pt=pix)

        return mask


    def start(self):
        # start thread to streaming the camera
        if self.thread is not None and self.thread.is_alive():
            print(f"[Pose Perceptor] a thread is running! Don't restart a new thread.")
        self.thread = threading.Thread(target=self.thread_get_pose_info, daemon=True)
        self.thread.start()
        print(f"[Pose Perceptor] Pre estimating...")
        while self.is_first:
            time.sleep(0.5)
        print(f"[Pose Perceptor] Ready")
        pass

    def restart(self):
        self.is_killing = True
        print(f"[Pose Perceptor] Stopping...")
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        print(f"[Pose Perceptor] Stopped!")
        self.is_killing = False
        self.is_first = True
        print(f"[Pose Perceptor] Restarting")
        self.start()
        print(f"[Pose Perceptor] Restarted!")

    def thread_get_pose_info(self):
        while not self.is_killing:
            rgb = None
            depth = None
            intrinsics = None

            time_stamp = time.time()
            rgb, depth, pc, intrinsics = self.camera.capture_vision_data()

            # * estimate pose
            if self.is_first:
                # estimate mask
                mask = self.get_init_mask(rgb=rgb,color = (48,54,56)) #! Change mesh color
                self.is_first = False
                pose = self.est.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
            else:
                pose = self.est.track_one(K=intrinsics, rgb=rgb, depth=depth, iteration=self.track_refine_iter)

            # center_pose = pose#@np.linalg.inv(self.mesh_to_origin)
            # vis = draw_posed_3d_box(intrinsics, img=rgb, ob_in_cam=center_pose, bbox=self.mesh_bbox)
            # vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)
            # cv2.imshow('1', vis[...,::-1])
            # cv2.waitKey(1)
            pose = self.tf_robot_2_external_camera@pose
            self.buffer.append(PoseState(
                time_stamp = time_stamp,
                pose=pose,
                rgb = rgb,
                depth = depth,
                intrinsics = intrinsics,
            ))

    @property
    def last_state(self)->PoseState:
        if self.thread is None:
            raise RuntimeError("[Pose Perceptor] No thread is running!")
        if len(self.buffer)>0:
            if (delta_t:= (time.time()-self.buffer[-1].time_stamp))>0.02:
                print(f"[Pose Perceptor] Warning: Pose state too old [{delta_t}]")
            return deepcopy(self.buffer[-1]), self.buffer[-1].time_stamp
        else:
            raise ValueError("[Pose Perceptor] No buffer available at the moment!")
        

    def __del__(self):
        self.is_killing = True
        if self.thread is not None:
            self.thread.join()
        del self.camera