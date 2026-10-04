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

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
from typing import List
import matplotlib.patches as patches

class Segmenter():
    def __init__(self) -> None:
        
        # Set EDGE_SAM_ROOT to a local clone of EdgeSAM with downloaded checkpoints.
        self.sam = sam_model_registry["edge_sam"](checkpoint=os.path.join(os.environ.get("EDGE_SAM_ROOT", "EdgeSAM"), "checkpoints", "edge_sam_3x.pth"))
        self.sam.to(device="cuda")
        from edge_sam import SamPredictor, sam_model_registry
        self.predictor = SamPredictor(self.sam)
        self.history = []
        self.lf_history = []
        self.rf_history = []
        self.click_points = []
        self.init_bbox = None
        self.init_finger_bboxs = np.array([[80,165,137,228],[296,169,356,229]])
        pass

    @property
    def is_initialized(self)->bool:
        return self.init_bbox is not None

    def reset(self):
        self.init_bbox = None
        self.history = []
        self.lf_history = []
        self.rf_history = []
        self.click_points = []
        pass

    def select_object(self, img):
        self.select_object_fig, self.select_object_ax = plt.subplots()
        self.select_object_ax.imshow(img)
        self.select_object_fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, self.select_object_ax, img))
        plt.show()

    def get_bbox(self, mask:np.ndarray, expand=True, expand_size = 8, previous_bbox=None):
        if not mask.any():
            return previous_bbox

        # Get the coordinates of the non-zero (True) mask entries
        rows = np.any(mask[0], axis=1)  # Detect rows with at least one '1'
        cols = np.any(mask[0], axis=0)  # Detect columns with at least one '1'

        # Get the bounding box coordinates (min and max rows and columns)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        if expand:
            H,W = mask[0].shape
            x_min = max(0,x_min-expand_size)
            y_min = max(0,y_min-expand_size)
            x_max = min(W,x_max+expand_size)
            y_max = min(H,y_max+expand_size)

        return np.array([x_min, y_min, x_max, y_max])
    
    def get_centroid(self,mask,last_centroid):
        true_indices = np.array(np.nonzero(mask))
        if true_indices.shape[1] > 0:
            centroid = true_indices.mean(axis=1)
            return np.flip(centroid).reshape(-1,2)
        else:
            return last_centroid

    def __call__(self, img):
        if len(self.history) == 0:
            # First segment
            # self.predictor.set_image(img)
            # masks, _, _ = self.predictor.predict(
            #     point_coords=np.array(self.click_points),
            #     point_labels=np.ones(len(self.click_points))
            #     )
            # self.history.append(np.array(self.click_points)[:1])
            # self.history.append(self.get_centroid(mask=masks[0],last_centroid=self.history[-1]))
            self.predictor.set_image(img)
            mask, _, _ = self.predictor.predict(
                box=self.init_bbox,num_multimask_outputs=1
                )
            lf_mask, _, _ = self.predictor.predict(
                box=self.init_finger_bboxs[0],num_multimask_outputs=1
                )
            rf_mask, _, _ = self.predictor.predict(
                box=self.init_finger_bboxs[1],num_multimask_outputs=1
                )
            self.history.append(self.get_bbox(mask, expand=True, previous_bbox=self.init_bbox))
            self.lf_history.append(self.get_bbox(lf_mask, expand=True, previous_bbox=self.init_finger_bboxs[0]))
            self.rf_history.append(self.get_bbox(rf_mask, expand=True, previous_bbox=self.init_finger_bboxs[1]))
            # self.history.append(cv2.resize(masks[0].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool))
        else:
            # Streaming segment based on previous bbox
            self.predictor.set_image(img)
            # masks, _, _ = self.predictor.predict(
            #     point_coords=np.array(self.history[-1]),
            #     point_labels=np.ones(1)
            #     )
            # self.history.append(self.get_centroid(mask=masks[0],last_centroid=self.history[-1]))
            mask, _, _ = self.predictor.predict(
                box=self.history[-1],num_multimask_outputs=1
                )
            lf_mask, _, _ = self.predictor.predict(
                box=self.lf_history[-1],num_multimask_outputs=1
                )
            rf_mask, _, _ = self.predictor.predict(
                box=self.rf_history[-1],num_multimask_outputs=1
                )
            self.history.append(self.get_bbox(mask, expand=True, previous_bbox=self.history[-1]))
            self.lf_history.append(self.get_bbox(lf_mask, expand=True, previous_bbox=self.lf_history[-1]))
            self.rf_history.append(self.get_bbox(rf_mask, expand=True, previous_bbox=self.rf_history[-1]))
            # masks, _, _ = self.predictor.predict(
            #     mask_input=self.history[-1][np.newaxis,...]
            #     )
            
        return mask[0].astype(np.uint8),lf_mask[0].astype(np.uint8),rf_mask[0].astype(np.uint8)

    def on_click(self, event, ax, img):
        # Get the coordinates of the mouse click
        x, y = int(event.xdata), int(event.ydata)
        
        # Save the coordinates
        print(f"Nr.{len(self.click_points)+1} Clicked at: ({x}, {y})")
        self.click_points.append((x,y))

        # Overlay the coordinates on the image
        ax.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=12)
        circle = patches.Circle((x, y), radius=5, transform=ax.transAxes, color='red', fill=True)
        ax.add_patch(circle)

        if len(self.click_points)==4:
            xs = np.array([self.click_points[i][0] for i in range(len(self.click_points))])
            ys = np.array([self.click_points[i][1] for i in range(len(self.click_points))])
            x_min = np.min(xs)
            y_min = np.min(ys)
            x_max = np.max(xs)
            y_max = np.max(ys)
            self.init_bbox = np.array([x_min, y_min, x_max, y_max])

        # Redraw the image with the new annotation
        ax.imshow(img)
        plt.draw()


class Camera():
    def __init__(self, serial_number:str = "846112072071", use_depth:bool=True) -> None: #!Change serial number to in hand camera
        super().__init__()
        self._pipeline = None
        self._align = None
        self.use_depth = use_depth

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
        # config.enable_stream(rs.stream.color, 1920,1080, rs.format.rgb8, 30)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        # config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60) #!disable depth stream due to camera issue
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 60)

        # Start the pipeline
        self._pipeline.start(config)
        align_to = rs.stream.color
        self._align = rs.align(align_to)
        time.sleep(1.0)  # Allow some time for the camera to warm up

    def capture_vision_data(self) -> Dict:
        if self._pipeline is None:
            raise RuntimeError(f"Camera should be started before.")
        
        # Wait for a coherent pair of frames: depth and color
        frames = self._pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if self.use_depth:
            depth_frame = aligned_frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            depth_image = np.zeros(color_image.shape[:2], dtype=np.uint16)

        # Convert the color image to a numpy array


        # Get the camera info and convert it to a ROS message
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array([intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1]).reshape(3,3) #! original K, not crop one

        # Convert the depth image to a point cloud and publish it
        # pc = rs.pointcloud()
        # pc_points = pc.calculate(depth_frame)
        # vertices = np.asanyarray(pc_points.get_vertices()).view(np.float32)
        vertices = None
        return color_image, depth_image/1000.0, vertices, K

    def __del__(self):
        if self._pipeline:
            self._pipeline.stop()

@dataclass
class VisionState:
    time_stamp: float 
    rgb: np.ndarray 
    depth: np.ndarray 
    mask: np.ndarray 
    intrinsics: np.ndarray

class VisionPerceptor:
    def __init__(self, buffer_size:int=5, output_img_size:tuple = (40,40), use_segment:bool=True, serial_number:str="846112072071", raw_crop_range:tuple = (0,-1,212-120,212+120),use_depth:bool=True) -> None:
        print("Initialize Camera")
        self.camera = Camera(serial_number=serial_number,use_depth=use_depth)
        self.camera.start_camera()
        time.sleep(2.0) 
        # del self.camera
        # self.camera = Camera(serial_number=serial_number,use_depth=use_depth)
        # self.camera.start_camera()
        # print("Camera restared!")

        self.use_segment = use_segment
        self.segmenter = None
        if self.use_segment:
            print("Initialize Segmenter")
            self.segmenter = Segmenter()
        print("Camera initialized")

        self.buffer = deque(maxlen=buffer_size)
        self.thread = None
        self.is_killing = False
        self.is_selecting_object = False

        self.output_img_size = output_img_size
        self.raw_crop_range = raw_crop_range
        pass


    def start(self):
        # start thread to streaming the camera
        if self.thread is not None and self.thread.is_alive():
            print(f"[Vision Perceptor] a thread is running! Don't restart a new thread.")
        self.thread = threading.Thread(target=self.thread_get_vision_info, daemon=True)
        self.thread.start()
        if self.use_segment:
            self.segmenter.reset()
        pass

    def select_object(self):
        print(f"[Vision Perceptor] Please select object.")
        self.is_selecting_object = True
        while self.is_selecting_object:
            time.sleep(0.5)
        print(f"[Vision Perceptor] Object selected.")

    def stop_mask(self):
        if self.use_segment:
            self.segmenter.init_bbox = None
            print(f"[Vision Perceptor] Stop generating mask.")

    def thread_get_vision_info(self):
        while not self.is_killing:
            rgb = None
            depth = None
            intrinsics = None
            
            rgb, depth, pc, intrinsics = self.camera.capture_vision_data()
            if self.is_selecting_object and self.use_segment:
                self.segmenter.reset()
                self.segmenter.select_object(rgb)
                self.is_selecting_object = False

            if self.use_segment and self.segmenter.is_initialized :
                mask,lf_mask,rf_mask = self.segmenter(img=rgb)
            else:
                mask = np.zeros(rgb.shape[:2], dtype=rgb.dtype)
                lf_mask = np.zeros(rgb.shape[:2], dtype=rgb.dtype)
                rf_mask = np.zeros(rgb.shape[:2], dtype=rgb.dtype)

            # rgb = cv2.resize(rgb,self.output_img_size)
            # depth = cv2.resize(depth,self.output_img_size)*(-1.0)
            # mask = cv2.resize(mask,self.output_img_size)
            # lf_mask = cv2.resize(lf_mask,self.output_img_size)
            # rf_mask = cv2.resize(rf_mask,self.output_img_size)
            rgb = cv2.resize(rgb[self.raw_crop_range[0]:self.raw_crop_range[1], self.raw_crop_range[2]:self.raw_crop_range[3], :],self.output_img_size)
            depth = cv2.resize(depth[self.raw_crop_range[0]:self.raw_crop_range[1], self.raw_crop_range[2]:self.raw_crop_range[3]],self.output_img_size)*(-1.0)
            mask = cv2.resize(mask[self.raw_crop_range[0]:self.raw_crop_range[1], self.raw_crop_range[2]:self.raw_crop_range[3]],self.output_img_size)
            lf_mask = cv2.resize(lf_mask[self.raw_crop_range[0]:self.raw_crop_range[1], self.raw_crop_range[2]:self.raw_crop_range[3]],self.output_img_size)
            rf_mask = cv2.resize(rf_mask[self.raw_crop_range[0]:self.raw_crop_range[1], self.raw_crop_range[2]:self.raw_crop_range[3]],self.output_img_size)
            mask = (mask*2+lf_mask+rf_mask+np.ones_like(mask)).astype(np.int32)
            
            self.buffer.append(VisionState(
                time_stamp = time.time(),
                rgb = rgb,
                depth = depth,
                mask = mask,
                intrinsics = intrinsics,
            ))

    @property
    def last_state(self)->VisionState:
        if self.thread is None:
            raise RuntimeError("[Vision Perceptor] No thread is running!")
        if len(self.buffer)>0:
            if (delta_t:= (time.time()-self.buffer[-1].time_stamp))>0.02:
                print(f"[Vision Perceptor] Warning: Vision state too old [{delta_t}]")
            return deepcopy(self.buffer[-1]), self.buffer[-1].time_stamp
        else:
            raise ValueError(f"[Vision Perceptor] No buffer available at the moment ({self.camera.serial_number})!")
        

    def __del__(self):
        self.is_killing = True
        if self.thread is not None:
            self.thread.join()
        del self.camera