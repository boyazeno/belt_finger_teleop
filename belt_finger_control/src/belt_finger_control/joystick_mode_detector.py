import pytransform3d.rotations as protations
import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
import time
import belt_finger_control.filter as bfilter
from dataclasses import dataclass
from xbox360controller import Xbox360Controller
from copy import deepcopy
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean


def calculate_tf_center_2_marker(cube_length:float=0.04):
    tf_center_2_marker = []
    tf_marker_2_center = []
    for i in range(5):
        tf = np.eye(4)
        tile_angle = np.pi/6
        
        #Rot
        if i== 0:
            pass
        elif i==1:
            tf[:3,:3] = protations.matrix_from_axis_angle([0, 1, 0, -1 * tile_angle])
        elif i==2:
            tf[:3,:3] = protations.matrix_from_axis_angle([0, 1, 0,  1 * tile_angle])
        elif i==3:
            tf[:3,:3] = protations.matrix_from_axis_angle([1, 0, 0,  1 * tile_angle])
        elif i==4:
            tf[:3,:3] = protations.matrix_from_axis_angle([1, 0, 0, -1 * tile_angle])
        else:
            raise Exception("Invalid index")
        
        # Trans
        if i== 0:
            pass
        elif i==1:
            tf[:3,3] = [-1*(np.cos(tile_angle)+1.0)*0.5*cube_length, 0.0, -1*np.sin(tile_angle)*cube_length]
        elif i==2:
            tf[:3,3] = [ 1*(np.cos(tile_angle)+1.0)*0.5*cube_length, 0.0, -1*np.sin(tile_angle)*cube_length]
        elif i==3:
            tf[:3,3] = [ 0.0, -1*(np.cos(tile_angle)+1.0)*0.5*cube_length, -1*np.sin(tile_angle)*cube_length]
        elif i==4:
            tf[:3,3] = [ 0.0,  1*(np.cos(tile_angle)+1.0)*0.5*cube_length, -1*np.sin(tile_angle)*cube_length]
        else:
            raise Exception("Invalid index")
        tf_center_2_marker.append(tf)
        tf_marker_2_center.append(np.linalg.inv(tf))
    return np.stack(tf_center_2_marker,axis=0), np.stack(tf_marker_2_center,axis=0)

def get_center_pose(ids:np.ndarray, tfs_world_2_marker:np.ndarray, tfs_marker_2_center:np.ndarray):
    use_naive_mean = True
    masked_tfs_marker_2_center = tfs_marker_2_center[ids]
    centers =  tfs_world_2_marker @ masked_tfs_marker_2_center
    if len(centers.shape) == 2:
        center = centers
    elif len(centers.shape) == 3:
        if use_naive_mean:
            u,s,vt = np.linalg.svd(np.mean(centers[:,:3,:3], axis=0), full_matrices=True)
            m = np.diag([1, 1, np.linalg.det(u @ vt)]) # reflection
            center_rot = u@m@vt
            center = np.eye(4)
            center[:3,:3] = center_rot
            center[:3,3] = np.mean(centers[:, :3, 3], axis=0)
        else:
            center,_ = robust_average_pose(poses=centers)
    else:
        raise Exception(f"Invalid shape: {centers.shape}")
    return center


def robust_average_pose(poses, 
                        outlier_thresh=None, 
                        rot_weight=1.0, 
                        max_iter=4):
    """
    Compute a robust average of multiple SE(3) poses, rejecting outliers.
    
    Parameters
    ----------
    poses : array-like, shape (N, 4, 4)
        List of N homogeneous transform matrices in SE(3).
    outlier_thresh : float or None
        Distance threshold (in same units as rot_weight) for outlier rejection.
        If None, uses 1.5 * IQR of distances to initial estimate.
    rot_weight : float
        Weighting factor between rotation (radians) and translation (same units).
    max_iter : int
        Number of re-estimation iterations (outlier rejection + mean recompute).
    
    Returns
    -------
    T_avg : ndarray, shape (4,4)
        The estimated average pose.
    inliers : list of int
        Indices of poses deemed inliers.
    """
    # Instantiate SE(3) group with left-invariant metric
    se3 = SpecialEuclidean(n=3, point_type='matrix')
    metric = se3.metric * rot_weight  # scale rotation part

    # Convert poses to array
    X = np.stack(poses)  # shape (N,4,4)
    N = len(X)
    
    # 1. Initial estimate: choose pose minimizing sum of pairwise distances
    #    (a simple 1-center approximation)
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dij = metric.dist(X[i], X[j])
            dists[i,j] = dists[j,i] = dij
    init_idx = np.argmin(dists.sum(axis=1))
    T_est = X[init_idx]
    
    # 2. Iteratively reject outliers and recompute mean
    inliers = np.arange(N)
    for _ in range(max_iter):
        # Compute distances to current estimate
        d = np.array([metric.dist(T_est, X[i]) for i in inliers])
        
        # Determine threshold if not provided: 1.5 * IQR rule
        if outlier_thresh is None:
            q1, q3 = np.percentile(d, [25, 75])
            iqr = q3 - q1
            thresh = q3 + 1.5 * iqr
        else:
            thresh = outlier_thresh
        
        # Select inliers
        new_inliers = inliers[d <= thresh]
        if new_inliers.shape[0] == inliers.shape[0]:
            # no change ⇒ convergence
            break
        inliers = new_inliers
        
        # Recompute the Frechét (Karcher) mean on SE(3) of inlier set 
        mean_estimator = FrechetMean(space=se3, method="default")
        mean_estimator.fit(X[inliers])
        T_est = mean_estimator.estimate_
    
    return T_est, inliers.tolist()


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

@dataclass
class JoystickState:
    axis_l_x: float = 0.0
    axis_l_y: float = 0.0
    axis_r_x: float = 0.0
    axis_r_y: float = 0.0
    trigger_l: float = 0.0
    trigger_r: float = 0.0
    reset: bool = False # button_b
    pause: bool = False # button_a
    recalibrate: bool = False # button_x

@dataclass
class CameraState:
    tf_world_2_center: np.ndarray = None
    tf_world_2_palm: np.ndarray = None
    tf_world_2_center_init: np.ndarray = None
    tf_world_2_palm_init: np.ndarray = None

@dataclass
class SpecialButtonState:
    start: bool=False
    select: bool=False
    button_y: bool=False
    tactile: bool=False


class JoystickController:
    def __init__(self, debug: bool = False):
        self.joystick_state = None
        self.controller = None
        self.debug = debug
        self.special_button_state = SpecialButtonState()

    def start(self):
        self.reset()
        self.controller = Xbox360Controller(0, axis_threshold=0.0)
        # Button A events
        self.controller.button_a.when_released = self.on_button_released
        self.controller.button_b.when_released = self.on_button_released
        self.controller.button_x.when_released = self.on_button_released
        self.controller.button_y.when_released = self.special_button_released
        self.controller.button_start.when_released = self.special_button_released
        self.controller.button_trigger_r.when_released = self.special_button_released

        # Left and right axis move event
        self.controller.axis_l.when_moved = self.on_axis_moved
        self.controller.axis_r.when_moved = self.on_axis_moved
        self.controller.trigger_l.when_moved = self.on_raw_axis_moved
        self.controller.trigger_r.when_moved = self.on_raw_axis_moved
    
    def special_button_released(self, button):
        if button.name == "button_start":
            self.special_button_state.start = not self.special_button_state.start 
        elif button.name == "button_y":
            self.special_button_state.button_y = True
        elif button.name == "button_trigger_r":
            self.special_button_state.tactile = not self.special_button_state.tactile
        else:
            print("Not assigned button released!")

    def stop(self):
        if self.controller is not None:
            self.controller.close()

    def reset(self):
        if self.joystick_state is not None:
            prev_state = self.joystick_state
        else:
            prev_state = JoystickState()
        self.joystick_state = JoystickState()
        self.joystick_state.axis_l_x = prev_state.axis_l_x 
        self.joystick_state.axis_l_y = prev_state.axis_l_y 
        self.joystick_state.axis_r_x = prev_state.axis_r_x 
        self.joystick_state.axis_r_y = prev_state.axis_r_y 
        self.joystick_state.trigger_l = prev_state.trigger_l 
        self.joystick_state.trigger_r = prev_state.trigger_r 
        print("[JoystickController] Reset done.")

    def get_state(self):
        return deepcopy(self.joystick_state)
    
    def get_special_botton_state(self):
        return deepcopy(self.special_button_state)

    def on_button_released(self, button):
        if button.name == "button_b":
            self.joystick_state.reset = True
        elif button.name == "button_a":
            self.joystick_state.pause = not self.joystick_state.pause
        elif button.name == "button_y":
            self.joystick_state.quit = True
        elif button.name == "button_x":
            self.joystick_state.recalibrate = True
        if self.debug:
            print(f'Button {button.name} was released')

    def on_axis_moved(self, axis):
        self.joystick_state.__dict__[axis.name+"_x"] = axis.x
        self.joystick_state.__dict__[axis.name+"_y"] = -1*axis.y
        if self.debug:
            print(f'Axis {axis.name} moved to x: {axis.x} y:{axis.y}')

    def on_raw_axis_moved(self, axis):
        self.joystick_state.__dict__[axis.name] = axis.value
        if self.debug:
            print(f'Axis {axis.name} moved to {axis.value}')
    
    def overwrite_state(self, state: JoystickState):
        self.joystick_state = state

    def vibrate(self, times:int=1, duration:float=0.5):
        while times>0:
            times -= 1
            self.controller.set_rumble(0.3, 0.0, int(duration*1000))
            time.sleep(0.5)

class DummyCameraController:
    def __init__(self) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_state(self):
        camera_state = CameraState()
        camera_state.tf_world_2_center = np.eye(4)
        camera_state.tf_world_2_palm = np.eye(4)
        camera_state.tf_world_2_center_init = np.eye(4)
        camera_state.tf_world_2_palm_init = np.eye(4)
        return camera_state
    
    def init_detection_pipeline(self) -> None:
        pass

    def update_pose(self) -> None:
        pass

    def calibrate_marker_tf(self, min_frames:int=10) -> None:
        pass

class CameraController:
    def __init__(self, visualize:bool=False, debug:bool=False):
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.detector = None
        self.tag_size = 0.032
        self.axis_length = self.tag_size * 0.5
        self.tfs_center_2_marker = None 
        self.tfs_marker_2_center = None
        self.visualize = visualize
        self.debug = debug
        self.serial_number = "739112060472" #"846112072071"
        
        self.tf_world_2_center = None
        self.tf_world_2_palm = None
        self.tf_world_2_center_init = None
        self.tf_world_2_palm_init = None
        self.filter_pose = None

    def start(self):
        self.tf_center_2_palm = np.eye(4)
        self.tf_center_2_palm[:3,3] = [0.0, 0.12, 0.0]
        self.init_detection_pipeline()
        self.tfs_center_2_marker, self.tfs_marker_2_center = calculate_tf_center_2_marker(cube_length=0.04)
        self.filter_pose = bfilter.ExponentialMovingAveragePoseFilter(alpha=0.5)
        self.reset()
        pass

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        pass

    def reset(self):
        # Wait for get first frame as reference
        print("Waiting for the first pose...")
        time.sleep(0.5)
        while self.tf_world_2_center is None or self.tf_world_2_palm is None:
            self.update_pose()
        self.tf_world_2_center_init = self.tf_world_2_center
        self.tf_world_2_palm_init = self.tf_world_2_palm
        self.filter_pose.reset()
        if self.debug:
            print(f"tf_world_2_center_init: {self.tf_world_2_center_init}")
            print(f"tf_world_2_palm_init: {self.tf_world_2_palm_init}")
        print("[CameraController] Reset done.")
        pass

    def get_state(self):
        self.update_pose()
        self.camera_state = CameraState()
        self.camera_state.tf_world_2_center = deepcopy(self.tf_world_2_center)
        self.camera_state.tf_world_2_palm = deepcopy(self.tf_world_2_palm)
        self.camera_state.tf_world_2_center_init = deepcopy(self.tf_world_2_center_init)
        self.camera_state.tf_world_2_palm_init = deepcopy(self.tf_world_2_palm_init)
        return self.camera_state

    def init_detection_pipeline(self):
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

    def update_pose(self) -> str:
        # Wait for and align frames
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
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
        ids = []
        tfs_world_2_marker = []
        for det in detections:
            # Add detected marker poses
            tf_world_2_marker = np.eye(4)
            tf_world_2_marker[:3, :3] = det.pose_R
            tf_world_2_marker[:3, 3] = det.pose_t.flatten()
            if not det.tag_id in (0,1,2,3,4):
                continue
            ids.append(det.tag_id)
            tfs_world_2_marker.append(tf_world_2_marker)
            

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
        if len(ids)!=0:
            self.tf_world_2_center = get_center_pose(ids, tfs_world_2_marker, self.tfs_marker_2_center)
            self.tf_world_2_center = self.filter_pose.update(self.tf_world_2_center)
            self.tf_world_2_palm = self.tf_world_2_center@self.tf_center_2_palm

        if self.visualize:
            if self.tf_world_2_palm is not None and self.tf_world_2_palm_init is not None:
                draw_pose_cv(img, t=self.tf_world_2_palm[:3,3], R=self.tf_world_2_palm_init[:3,:3], intrinsics=self.intrinsics, thickness=5, length=self.axis_length*2)
                draw_pose_cv(img, t=self.tf_world_2_palm[:3,3], R=self.tf_world_2_palm[:3,:3], intrinsics=self.intrinsics, thickness=3, length=self.axis_length*2)
                draw_point_cv(img, t=self.tf_world_2_palm_init[:3,3], intrinsics=self.intrinsics, color=(0, 0, 0), radius=10)
                draw_point_cv(img, t=self.tf_world_2_palm[:3,3], intrinsics=self.intrinsics, color=(0, 100, 0), radius=6)

        if self.visualize:
            cv2.imshow('Tag36h11 Detection with Axes', img)
            cv2.waitKey(1)

    def calibrate_marker_tf(self, min_frames:int=10) -> str:

        tf_center_2_marker_per_frame = {i:[] for i in range(5)}
        while True:
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
            tf_w_2_markers = {}
            for det in detections:
                # Add detected marker poses
                tf_world_2_marker = np.eye(4)
                tf_world_2_marker[:3, :3] = det.pose_R
                tf_world_2_marker[:3, 3] = det.pose_t.flatten()
                tf_w_2_markers[det.tag_id] = tf_world_2_marker
                

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
            cv2.circle(img, [20,20], 10, (255,0,0), -1) # indicate calibration mode


            # Get the relative pose of the detected markers w.r.t the center (marker 0)
            if tf_w_2_markers.get(0) is None:
                print(f"Marker 0 not detected! Drop this frame.")
                continue
            
            cur_min_frames = np.inf
            for i,tf_w_2_marker in tf_w_2_markers.items():
                tf_center_2_marker_per_frame[i].append(np.linalg.inv(tf_w_2_markers[0])@tf_w_2_marker)
                cv2.putText(img, f"ID:{i} detected frames:{len(tf_center_2_marker_per_frame[i])}", (40, i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cur_min_frames = min(cur_min_frames, len(tf_center_2_marker_per_frame[i]))

            if self.visualize:
                cv2.imshow('Tag36h11 Detection with Axes', img)
                cv2.waitKey(1)
            
            if cur_min_frames >= min_frames:
                break

        # Calculate the average transformation matrix for each marker
        tf_center_2_marker = []
        tf_marker_2_center = []
        for i in range(5):
            tf_center_2_marker_per_frame[i] = np.stack(tf_center_2_marker_per_frame[i], axis=0)
            u,s,vt = np.linalg.svd(np.mean(tf_center_2_marker_per_frame[i][:,:3,:3], axis=0), full_matrices=True)
            m = np.diag([1, 1, np.linalg.det(u @ vt)]) # reflection
            center_rot = u@m@vt
            center = np.eye(4)
            center[:3,:3] = center_rot
            center[:3,3] = np.mean(tf_center_2_marker_per_frame[i][:, :3, 3], axis=0)
            tf_center_2_marker.append(center)
            tf_marker_2_center.append(np.linalg.inv(center))

        self.tfs_center_2_marker, self.tfs_marker_2_center = np.stack(tf_center_2_marker,axis=0), np.stack(tf_marker_2_center,axis=0)
        print(f"calibration succeed!")

        return 
        
class JoystickModeDetector:
    def __init__(self, control_arm:bool=True):
        self.scale_max = {"robot":(np.array([0.05,0.05,0.05,np.pi/4,np.pi/4,np.pi/4]),
                                   np.array([0.005,0.005,0.005,0.005,0.005,0.005])), 
                          "opening": (0.00,0.08), "transy": (-1.0, 0.5), "rotz": (1.0, -0.5), "rotx":  (1.0, -0.5)} # origin->converted
        self.scale_min = {"robot":(np.array([-0.05,-0.05,-0.05,-np.pi/4,-np.pi/4,-np.pi/4]),
                                   np.array([-0.005,-0.005,-0.005,-0.005,-0.005,-0.005])), 
                          "opening": (1.0,0.0), "transy": (1.0, -0.5), "rotz": (-1.0, 0.5), "rotx": (-1.0, 0.5)}
        
        self.visualize = True
        self.debug = False
        self.joystick_controller = JoystickController(debug=self.debug)
        if control_arm:
            self.camera_controller = CameraController(visualize=self.visualize, debug=self.debug)
        else:
            self.camera_controller = DummyCameraController()
        self.filter_dict = bfilter.FilterDict(bfilter.ExponentialMovingAverageFilter, ["robot","opening", "transy", "rotz", "rotx"])#, {"window_len": 5})
        self.visualize = True

        self.last_joystick_state = None
        self.last_camera_state = None

        self.tf_ee_2_init = self.get_tf_ee_2_init()

    def get_tf_ee_2_init(self):
        tf_ee_2_init = np.eye(4)
        tf_ee_2_init[:3,0] = [0,1,0]
        tf_ee_2_init[:3,1] = [1,0,0]
        tf_ee_2_init[:3,2] = [0,0,-1]
        return tf_ee_2_init
    
    def __del__(self):
        self.joystick_controller.stop()
        self.camera_controller.stop()

    def start(self):
        print(f"Stopping joystick controller...")
        self.joystick_controller.start()
        print(f"Stopping camera controller...")
        self.camera_controller.start()

    def reset(self):
        self.joystick_controller.reset()
        self.camera_controller.reset()
        self.filter_dict.reset()
        # self.last_joystick_state = None
        self.last_camera_state:CameraState = None

    def control(self):
        joystick_state = self.joystick_controller.get_state()
        joystick_special_botton_state = self.joystick_controller.get_special_botton_state()
        camera_state = self.camera_controller.get_state()

        if joystick_special_botton_state.button_y:
            print(f"Quit now!")
            return None, None

        if joystick_state.reset:
            self.reset()

        if joystick_state.recalibrate:
            self.camera_controller.calibrate_marker_tf(min_frames=10)
            joystick_state.recalibrate = False
            self.joystick_controller.overwrite_state(joystick_state)

        arm_action = np.zeros((6,)) #dx,dy,dz,drx,dry,drz (real value)
        gripper_action = np.array([0.08, 0.0, 0.0, 0.0]) #opening, transy, rotz, rotx (real value)
        if not joystick_state.pause:
            # Convert state to action
            if self.last_camera_state is not None:
                # Get robot action
                # tf_last_palm_2_current_palm = np.linalg.inv(self.last_camera_state.tf_world_2_palm) @ camera_state.tf_world_2_palm
                tf_init_palm_2_current_palm = np.linalg.inv(camera_state.tf_world_2_palm_init) @ camera_state.tf_world_2_palm
                
                tf_ee_palm_2_current_palm = self.tf_ee_2_init@tf_init_palm_2_current_palm@np.linalg.inv(self.tf_ee_2_init)
                rxyz = protations.euler_from_matrix(tf_ee_palm_2_current_palm[:3,:3],i=0,j=1,k=2,extrinsic=False)
                xyz = tf_ee_palm_2_current_palm[:3,3].flatten()
                arm_action = np.concatenate([xyz, rxyz],axis=0)

            arm_action = arm_action * (0.8 if joystick_state.trigger_r>0.5 else 0.3) # speed up when trigger_r pressed
            arm_action = self.normalize_action("robot", arm_action)
            # Get gripper action
            opening = self.normalize_action("opening", joystick_state.trigger_l)
            transy = self.normalize_action("transy", joystick_state.axis_l_y)
            rotz = self.normalize_action("rotz", joystick_state.axis_r_x)
            rotx = self.normalize_action("rotx", joystick_state.axis_r_y)
            gripper_action = np.array([opening, transy, rotz, rotx])

            self.last_camera_state = camera_state
            self.last_joystick_state = joystick_state
        else:
            opening = self.normalize_action("opening", self.last_joystick_state.trigger_l)
            gripper_action = np.array([opening, 0.0, 0.0, 0.0])
        
        return arm_action, gripper_action
        

    def normalize_action(self, key, val, use_filter=True):
        if key in self.scale_max:
            val = np.clip((val - self.scale_min[key][0]) / (self.scale_max[key][0] - self.scale_min[key][0])*2-1, -1.0, 1.0) # convert to [-1,1]
            if key == "opening":
                val = (val + 1.0)/2.0
            else:
                # val = (np.sign(val)*val**2 + 1.0)/2.0 # slow down the changes near zeros, then convert to [0,1]
                val = (val+ 1.0)/2.0 # dont slow down
            rescaled_val = val * (self.scale_max[key][1] - self.scale_min[key][1]) + self.scale_min[key][1] # remap to [min,max]
            if use_filter:
                return self.filter_dict.update(key, rescaled_val)
            else:
                return rescaled_val
        else:
            raise Exception(f"Invalid key: {key}")
    