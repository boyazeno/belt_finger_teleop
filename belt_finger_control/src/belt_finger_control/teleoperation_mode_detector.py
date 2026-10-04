import pytransform3d.rotations as protations
from typing import Union, Tuple
import numpy as np
import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
import time
import belt_finger_control.filter as bfilter

def calculate_tf_center_2_marker(cube_length:float=0.02):
    tf_center_2_marker = []
    tf_marker_2_center = []
    for i in range(6*2):
        tf = np.eye(4)
        
        #Rot
        if (i%6)<4:
            tf[:3,:3] = protations.matrix_from_axis_angle([0, 1, 0, -1 * (i%6) * np.pi / 2])
        elif (i%6)==4:
            tf[:3,:3] = protations.matrix_from_axis_angle([1, 0, 0, -1 * np.pi / 2])
        elif (i%6)==5:
            tf[:3,:3] = protations.matrix_from_axis_angle([1, 0, 0,  1 * np.pi / 2])
        else:
            raise Exception("Invalid index")
        
        # Trans
        if (i%6) == 0:
            tf[2,3] = -1 * cube_length / 2
        elif (i%6) == 1:
            tf[0,3] =  1 * cube_length / 2
        elif (i%6) == 2:
            tf[2,3] =  1 * cube_length / 2
        elif (i%6) == 3:
            tf[0,3] = -1 * cube_length / 2
        elif (i%6) == 4:
            tf[1,3] = -1 * cube_length / 2
        elif (i%6) == 5:
            tf[1,3] =  1 * cube_length / 2 + 0.003
        else:
            raise Exception("Invalid index")
        tf_center_2_marker.append(tf)
        tf_marker_2_center.append(np.linalg.inv(tf))
    return np.stack(tf_center_2_marker,axis=0), np.stack(tf_marker_2_center,axis=0)


def get_center_pose(ids:np.ndarray, tfs_world_2_marker:np.ndarray, tfs_marker_2_center:np.ndarray):
    masked_tfs_marker_2_center = tfs_marker_2_center[ids]
    centers =  tfs_world_2_marker @ masked_tfs_marker_2_center
    if len(centers.shape) == 2:
        center = centers
    elif len(centers.shape) == 3:
        u,s,vt = np.linalg.svd(np.mean(centers[:,:3,:3], axis=0), full_matrices=True)
        m = np.diag([1, 1, np.linalg.det(u @ vt)]) # reflection
        center_rot = u@m@vt
        center = np.eye(4)
        center[:3,:3] = center_rot
        center[:3,3] = np.mean(centers[:, :3, 3], axis=0)
    else:
        raise Exception(f"Invalid shape: {centers.shape}")
    return center


def draw_pose_cv(img, t, R, intrinsics, thickness=2, length=0.1):
    origin = t.flatten().tolist()
    axes = {
        'x': (R.dot([length, 0, 0]) + t.flatten()).tolist(),
        'y': (R.dot([0, length, 0]) + t.flatten()).tolist(),
        'z': (R.dot([0, 0, length]) + t.flatten()).tolist()
    }
    colors = {'x': (0,0,255), 'y': (0,255,0), 'z': (255,0,0)}

    # Project and draw each axis
    o_px = rs.rs2_project_point_to_pixel(intrinsics, origin)
    for ax, end_pt in axes.items():

        e_px = rs.rs2_project_point_to_pixel(intrinsics, end_pt)
        o_px_int = (int(round(o_px[0])), int(round(o_px[1])))
        e_px_int = (int(round(e_px[0])), int(round(e_px[1])))
        cv2.line(img, o_px_int, e_px_int, colors[ax], thickness)
        cv2.putText(img, ax.upper(), e_px_int, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors[ax], 2)

def draw_point_cv(img, t, intrinsics, color=(0, 255, 0), radius=5):
    # Project the 3D point to 2D pixel coordinates
    pixel = rs.rs2_project_point_to_pixel(intrinsics, t)
    pixel_int = (int(round(pixel[0])), int(round(pixel[1])))
    
    # Draw the point as a circle
    cv2.circle(img, pixel_int, radius, color, -1)

def draw_axis_cv(img, axis, t, intrinsics, length=0.1, color=(0, 255, 0), thickness=5):
    origin = t.flatten().tolist()
    axis_normalized = axis / np.linalg.norm(axis)
    end = (axis_normalized.flatten() * length + t).tolist()
    o_px = rs.rs2_project_point_to_pixel(intrinsics, origin)
    e_px = rs.rs2_project_point_to_pixel(intrinsics, end)
    o_px_int = (int(round(o_px[0])), int(round(o_px[1])))
    e_px_int = (int(round(e_px[0])), int(round(e_px[1])))
    cv2.line(img, o_px_int, e_px_int, color, thickness)


class TeleoperationModeDetector:
    def __init__(self):
        # self.scale_max = {"open": (0.03,0.08), "transy": (0.04, -1.0), "rotz": (np.deg2rad(45), -1.0), "rotx":  (np.deg2rad(45), -1.0)} # origin->converted
        # self.scale_min = {"open": (0.005,0.0), "transy": (-0.04, 1.0), "rotz": (np.deg2rad(-45), 1.0), "rotx": (np.deg2rad(-45), 1.0)}
        self.scale_max = {"open": (0.03,0.08), "transy": (0.04, -0.5), "rotz": (np.deg2rad(45), -0.5), "rotx":  (np.deg2rad(45), -0.5)} # origin->converted
        self.scale_min = {"open": (0.005,0.0), "transy": (-0.04, 0.5), "rotz": (np.deg2rad(-45), 0.5), "rotx": (np.deg2rad(-45), 0.5)}
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.detector = None
        self.tag_size = 0.016
        self.axis_length = self.tag_size * 0.5
        self.tfs_center_2_marker = None 
        self.tfs_marker_2_center = None
        self.tf_left_corner = None
        self.tf_right_corner = None
        self.tf_left_middle = None
        self.tf_right_middle = None
        self.tf_left_corner_init = None
        self.tf_right_corner_init = None
        self.tf_corner_2_center = None
        self.filter_dict = bfilter.FilterDict(bfilter.ExponentialMovingAverageFilter, ["open", "transy", "rotz", "rotx"])#, {"window_len": 5})
        self.visualize = True
        self.serial_number = "846112072071"

    def reset(self):
        self.tf_left_corner = None
        self.tf_right_corner = None
        self.tf_corner_2_middle = np.eye(4)
        self.tf_corner_2_middle[:3,3] = [0.0, 0.05, 0.0]
        self.init_detection_pipeline()
        self.tfs_center_2_marker, self.tfs_marker_2_center = calculate_tf_center_2_marker(cube_length=0.02)

        # Wait for get first frame as reference
        print("Waiting for the first pose...")
        time.sleep(0.5)
        while self.tf_left_corner is None or self.tf_right_corner is None:
            self.update_hand_pose()
        self.tf_left_corner_init = self.tf_left_corner
        self.tf_right_corner_init = self.tf_right_corner

        self.vec_center_init = (self.tf_left_corner_init[:3,3] + self.tf_right_corner_init[:3,3])/2
        self.vec_middle_init = (self.tf_left_middle[:3,3] + self.tf_right_middle[:3,3])/2
        self.vec_x_init = (self.tf_left_corner_init[:3,0] + self.tf_right_corner_init[:3,0])/2
        self.vec_x_init = self.vec_x_init/np.linalg.norm(self.vec_x_init)
        self.vec_y_init = (self.tf_left_corner_init[:3,1] + self.tf_right_corner_init[:3,1])/2
        self.vec_y_init = self.vec_y_init/np.linalg.norm(self.vec_y_init)
        self.vec_z_init = (self.tf_left_corner_init[:3,2] + self.tf_right_corner_init[:3,2])/2
        self.vec_z_init = self.vec_z_init/np.linalg.norm(self.vec_z_init)
        print("Waiting for the first pose...")
        self.filter_dict.reset()


    def init_detection_pipeline(self):
        # Initialize RealSense
        self.pipeline = rs.pipeline()
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
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Get intrinsics for the color stream
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # AprilTag detector for tag36h11
        self.detector = Detector(families='tag36h11',
                            nthreads=4,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=True,
                            decode_sharpening=0.25,
                            debug=False)
        
        
    def normalize(self, key, val, use_filter=True):
        if key in self.scale_max:
            val = np.clip((val - self.scale_min[key][0]) / (self.scale_max[key][0] - self.scale_min[key][0])*2-1, -1.0, 1.0)
            val = (val**3 + 1.0)/2.0
            rescaled_val = val * (self.scale_max[key][1] - self.scale_min[key][1]) + self.scale_min[key][1]
            if use_filter:
                return self.filter_dict.update(key, rescaled_val)
            else:
                return rescaled_val
        else:
            raise Exception(f"Invalid key: {key}")
        
    def get_mode_value(self) -> Tuple:
        opening = np.linalg.norm(self.tf_left_corner[:2,3]-self.tf_right_corner[:2,3])
        vec_middle = (self.tf_left_middle[:3,3] + self.tf_right_middle[:3,3])/2
        # vec_center = (self.tf_left_corner[:3,3] + self.tf_right_corner[:3,3])/2
        vec_x = (self.tf_left_corner[:3,0] + self.tf_right_corner[:3,0])/2
        vec_y = (self.tf_left_corner[:3,1] + self.tf_right_corner[:3,1])/2
        vec_z = (self.tf_left_corner[:3,2] + self.tf_right_corner[:3,2])/2
        vec_x = vec_x/np.linalg.norm(vec_x)
        vec_y = vec_y/np.linalg.norm(vec_y)
        vec_z = vec_z/np.linalg.norm(vec_z)


        transy = (vec_middle-self.vec_middle_init)@self.vec_y_init
        rotz = np.arccos(vec_x@self.vec_x_init) * np.sign(np.cross(self.vec_x_init,vec_x)@self.vec_z_init)
        rotx = np.arccos(vec_z@self.vec_z_init) * np.sign(np.cross(self.vec_z_init,vec_z)@self.vec_x_init)

        opening = self.normalize("open", opening)
        transy = self.normalize("transy", transy)
        rotz = self.normalize("rotz", rotz)
        rotx = self.normalize("rotx", rotx)
        return opening,transy,rotz,rotx


    def update_hand_pose(self) -> str:
            # Wait for and align frames
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                return None,None

            # Convert color frame to image
            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect tags and estimate pose
            detections = self.detector.detect(gray,
                                        estimate_tag_pose=True,
                                        camera_params=[ self.intrinsics.fx,
                                                        self.intrinsics.fy,
                                                        self.intrinsics.ppx,
                                                        self.intrinsics.ppy],
                                        tag_size=self.tag_size)

            # Draw detections and axes
            ids_left = []
            ids_right = []
            tfs_world_2_marker_left = []
            tfs_world_2_marker_right = []
            for det in detections:
                # Add detected marker poses
                tf_world_2_marker = np.eye(4)
                tf_world_2_marker[:3, :3] = det.pose_R
                tf_world_2_marker[:3, 3] = det.pose_t.flatten()
                if det.tag_id//6 == 1:
                    ids_left.append(det.tag_id)
                    tfs_world_2_marker_left.append(tf_world_2_marker)
                else:
                    ids_right.append(det.tag_id)
                    tfs_world_2_marker_right.append(tf_world_2_marker)


                if self.visualize:
                    # Draw polygon around tag
                    corners = np.rint(det.corners).astype(int)
                    for i in range(4):
                        pt1 = tuple(corners[i])
                        pt2 = tuple(corners[(i+1) % 4])
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                    # Draw tag ID near center
                    cx = int(np.mean(corners[:, 0]))
                    cy = int(np.mean(corners[:, 1]))
                    cv2.putText(img, f"ID:{det.tag_id}", (cx-10, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # draw_pose_cv(img, t=det.pose_t, R=det.pose_R, intrinsics=self.intrinsics, thickness=2, length=self.axis_length)

            # Get center pose of all detected tags
            if len(ids_left)!=0:
                tf_world_2_center_left = get_center_pose(ids_left, tfs_world_2_marker_left, self.tfs_marker_2_center)
                self.tf_left_corner = np.eye(4)
                self.tf_left_corner[:3,3] = [0.022, 0.01, 0.0]
                self.tf_left_corner = tf_world_2_center_left@self.tf_left_corner
                self.tf_left_middle = self.tf_left_corner@self.tf_corner_2_middle

            if len(ids_right)!=0:
                tf_world_2_center_right = get_center_pose(ids_right, tfs_world_2_marker_right, self.tfs_marker_2_center)
                self.tf_right_corner = np.eye(4)
                self.tf_right_corner[:3,3] = [-0.022, 0.01, 0.0]
                self.tf_right_corner = tf_world_2_center_right@self.tf_right_corner
                self.tf_right_middle = self.tf_right_corner@self.tf_corner_2_middle


            if self.visualize:
                if self.tf_left_corner is not None:
                    draw_pose_cv(img, t=self.tf_left_corner[:3,3], R=self.tf_left_corner[:3,:3], intrinsics=self.intrinsics, thickness=4, length=self.axis_length*2)
                    pass
                if self.tf_right_corner is not None:
                    draw_pose_cv(img, t=self.tf_right_corner[:3,3], R=self.tf_right_corner[:3,:3], intrinsics=self.intrinsics, thickness=4, length=self.axis_length*2)
                    pass
                if self.tf_left_corner is not None and self.tf_right_corner is not None and self.tf_left_corner_init is not None and self.tf_right_corner_init is not None:
                    center = (self.tf_left_middle[:3,3]+self.tf_right_middle[:3,3])/2
                    vec_z_init = (self.tf_left_corner_init[:3,2]+self.tf_right_corner_init[:3,2])/2
                    vec_z = (self.tf_left_corner[:3,2]+self.tf_right_corner[:3,2])/2
                    vec_x_init = (self.tf_left_corner_init[:3,0]+self.tf_right_corner_init[:3,0])/2
                    vec_x = (self.tf_left_corner[:3,0]+self.tf_right_corner[:3,0])/2

                    draw_point_cv(img, t=self.vec_middle_init, intrinsics=self.intrinsics, color=(0, 0, 0), radius=5)
                    draw_axis_cv(img, axis=vec_z_init, t=center, intrinsics=self.intrinsics, length=self.axis_length*2,thickness=2, color=(0, 0, 255))
                    draw_axis_cv(img, axis=vec_x_init, t=center, intrinsics=self.intrinsics, length=self.axis_length*2,thickness=2, color=(255, 0, 0))
                    draw_point_cv(img, t=center, intrinsics=self.intrinsics, color=(0, 100, 0), radius=3)
                    draw_axis_cv(img, axis=vec_z, t=center, intrinsics=self.intrinsics, length=self.axis_length*4,thickness=1, color=(0, 0, 255))
                    draw_axis_cv(img, axis=vec_x, t=center, intrinsics=self.intrinsics, length=self.axis_length*4,thickness=1, color=(255, 0, 0))

            if self.visualize:
                cv2.imshow('Tag36h11 Detection with Axes', img)
                cv2.waitKey(1)
