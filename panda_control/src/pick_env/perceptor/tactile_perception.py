import matplotlib.pyplot as plt
import torch
import os
from collections import deque

from torch.autograd import Variable
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import time
import threading
from minsight.dataset import Postprocessor
from minsight.train import load_checkpoint
from minsight.utils import read_json, LocalParams
from minsight.model import get_model
from copy import deepcopy
from dataclasses import dataclass
from pytransform3d import rotations as pr

# Set MINSIGHT_ROOT to a local clone of minsight containing the sensor config and checkpoints.
MINSIGHT_CONFIG_DIR = os.path.join(os.environ.get("MINSIGHT_ROOT", "minsight"), "config")

class Sensor:
    def __init__(self, data_path, use_gpu, cam, sensor_model):

        self.indentation_info = torch.from_numpy(
            np.load(os.path.join(data_path, "indentation_info.npy"))
        )
        self.col_ind = torch.from_numpy(
            np.load(os.path.join(data_path, "column_idx_mapping_post.npy"))
        )
        self.skeleton_surface = np.squeeze(
            np.load(os.path.join(data_path, "beam_nodes.npy"))
        )
        self.skeleton_surface = np.insert(
            self.skeleton_surface, 0, np.zeros(len(self.skeleton_surface)), axis=1
        )
        self.val_index = np.load(os.path.join(data_path, "01_valid_index.npy"))
        self.use_gpu = use_gpu
        self.cap = self.set_camera(cam)
        config = read_json(data_path + "config_local.json")
        params = LocalParams(config)
        # os.makedirs(params.model_dir, exist_ok=True)

        # use gpu or not
        torch.cuda.empty_cache()
        print("use_gpu:{}".format(use_gpu))

        self.model, optimizer_ft, exp_lr_scheduler = get_model(params)
        self.postprocessor = Postprocessor(use_gpu, params)
        checkpoint_path = os.path.join(data_path, "checkpoint.pt")
        epoch, step, wandb_id = load_checkpoint(
            checkpoint_path, self.model, optimizer_ft, exp_lr_scheduler
        )

        print("Wandb ID: %s: " % wandb_id)
        # print(epoch)

        if use_gpu:
            self.model = self.model.cuda()

        print("Testing fully trained model")
        self.model.eval()

        # self.X3 = Image.open(os.path.join(data_path, 'Resized_NoContact.png'))
        X3_orig = cv2.imread(os.path.join(MINSIGHT_CONFIG_DIR, f"default_img_{sensor_model}.png"))
        self.X3 = self.imaging()
        while np.abs(np.abs(self.X3).mean()-np.abs(X3_orig).mean())>10:
            self.X3 = self.imaging()
            
        self.X4 = TF.to_tensor(cv2.imread((os.path.join(data_path, "gradient1.png"))))
        self.X5 = TF.to_tensor(cv2.imread((os.path.join(data_path, "gradient2.png"))))

        self.indentation_z_gt_zero_mask = (self.indentation_info[:, 3] > 0).cpu().numpy()

    def __del__(self):
        plt.close()

    def set_camera(self, cam):
        time.sleep(3.0)
        cap = cv2.VideoCapture(cam)

        # Check whether user selected camera is opened successfully.
        if not (cap.isOpened()):
            print("Could not open video device")
            cap = cv2.VideoCapture(cam)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        # cap.set(cv2.CAP_PROP_GAMMA, 170)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 156.0)  # 300
        cap.set(cv2.CAP_PROP_AUTO_WB, -1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        return cap

    def set_no_contact_img(self, img):
        self.X3 = img

    def imaging(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Could not take picture, restarting camera")
            # cap = cv2.VideoCapture(1)
            for i in range(10):
                ret, frame = self.cap.read()
        crop = frame[:, 160:1120]
        crop = cv2.flip(crop, -1)
        crop = cv2.resize(crop, (410, 308))
        return crop

    def inference(self, threshold=0.003):

        input_img = self.imaging()
        X1 = input_img

        X13 = TF.to_tensor(cv2.subtract(np.array(X1), np.array(self.X3)))
        input = torch.cat([X13, self.X4[0][None, :, :], self.X5[0][None, :, :]], 0).unsqueeze(0).to(device="cuda" if self.use_gpu else "cpu")

        # forward
        output = self.model(input)

        force_map = self.postprocessor.undo_rescale(torch.squeeze(output)).detach().cpu().numpy()

        #* New added
        filter_mask = np.ones((40, 40))
        filter_mask[np.where(np.linalg.norm(force_map, axis=0) < threshold)] = 0
        predict = force_map * filter_mask
        force_map = predict.reshape((3, 1600))
        # postprocessor.transform(force)
        force_in_pts = force_map[:, self.col_ind]

        filter_mask_in_pts =  (np.linalg.norm(force_in_pts, axis=0) > threshold)&self.indentation_z_gt_zero_mask
        force_map[:, self.col_ind][:,~filter_mask_in_pts] = 0.0
        
        # force_map_sim = np.stack([-1*force_map[1], force_map[2], -1*force_map[0]],axis=0)
        # force_map_sim = force_map_sim.transpose(0,2,1)

        # force_map_sim[force_map_sim<threshold] = 0.0
        return force_map.reshape(3,40,40)
    

@dataclass
class TactileState:
    time_stamp: float 
    force_map_left: np.ndarray 
    force_map_right: np.ndarray 
    sum_force_left: np.ndarray
    sum_force_right: np.ndarray
    discret_forces_left: np.ndarray
    discret_forces_right: np.ndarray

class TactilePerceptor:
    def __init__(self, buffer_size:int=5, ignore_left=False) -> None:
        print("Initialize Sensor")
        data_path=MINSIGHT_CONFIG_DIR + os.sep
        use_gpu=torch.cuda.is_available()
        self.ignore_left = ignore_left
        
        #! For port, run run lsusb /bin/udevadm info --name=/dev/video0 | grep ID to see which port is it (fix the sensor first)
        if self.ignore_left:
            self.sensor_left = None
        else:
            self.sensor_left = Sensor(os.path.join(data_path, "C2/"), use_gpu, "/dev/video8", "C2")
        self.sensor_right = Sensor(os.path.join(data_path, "C1/"), use_gpu, "/dev/video6", "C1")

        self.buffer = deque(maxlen=buffer_size)
        self.thread = None
        self.is_killing = False

        self.tf_tcp_2_left = np.eye(4)@self._mat(rxyz=[np.pi/3,0,0])@self._mat(rxyz=[0,0,np.pi/2])
        self.tf_tcp_2_right = np.eye(4)@self._mat(rxyz=[-np.pi/3,0,0])@self._mat(rxyz=[0,0,-np.pi/2])

        self.discret_force_areas = [self.get_area_pts(area_idx) for area_idx in range(5)]
        pass

    def _mat(self, xyz=None,rxyz=None):
        m = np.eye(4)
        if xyz:
            m[:3,3] = xyz
        if rxyz:
            m[:3,:3] = pr.matrix_from_euler(rxyz, i=0,j=1,k=2, extrinsic=False) 
        return m
    
    def start(self):
        # start thread to streaming the camera
        if self.thread is not None and self.thread.is_alive():
            print(f"[Tactile Perceptor] a thread is running! Don't restart a new thread.")
        self.thread = threading.Thread(target=self.thread_get_tactile_info, daemon=True)
        self.thread.start()
        pass

    def thread_get_tactile_info(self):
        threshold = 0.01 #0.005
        while not self.is_killing:
            force_map_right = self.sensor_right.inference(threshold=threshold)
            if self.ignore_left:
                force_map_left = np.zeros_like(force_map_right)
            else:
                force_map_left = self.sensor_left.inference(threshold=threshold)
            force_map_left = (self.tf_tcp_2_left[:3,:3]@force_map_left.reshape(3,-1)).reshape(3,40,40)
            force_map_right = (self.tf_tcp_2_right[:3,:3]@force_map_right.reshape(3,-1)).reshape(3,40,40)

            discret_forces_left = self.get_discret_forces(force_map=force_map_left)
            discret_forces_right = self.get_discret_forces(force_map=force_map_right)

            # sum of the forces
            self.buffer.append(TactileState(
                time_stamp= time.time(),
                force_map_left=force_map_left,
                force_map_right=force_map_right,
                sum_force_left=force_map_left.reshape(3,-1).sum(axis=-1),
                sum_force_right=force_map_right.reshape(3,-1).sum(axis=-1),
                discret_forces_left = discret_forces_left,
                discret_forces_right = discret_forces_right,
            ))

    @property
    def last_state(self)->TactileState:
        if self.thread is None:
            raise RuntimeError("[Tactile Perceptor] No thread is running!")
        if len(self.buffer)>0:
            if (delta_t:= (time.time()-self.buffer[-1].time_stamp))>0.02:
                print(f"[Tactile Perceptor] Warning: tactile state too old [{delta_t}]")
            return deepcopy(self.buffer[-1]), self.buffer[-1].time_stamp
        else:
            raise ValueError("[Tactile Perceptor] No buffer available at the moment!")
        
    def __del__(self):
        self.is_killing = True
        if self.thread is not None:
            self.thread.join()
        del self.sensor_left
        del self.sensor_right

    def get_discret_forces(self, force_map:np.ndarray, num:int=5):
        discret_forces = []
        if num == 5:
            for discret_force_area in self.discret_force_areas:
                discret_forces.append(np.sum(force_map[:,discret_force_area[0],discret_force_area[1]], axis=-1))
        else:
            raise RuntimeError("not implemented for num!=5")
            
        return np.stack(discret_forces,axis=0)

    @staticmethod
    def get_area_pts(area_idx:int, theta_c:float=np.pi/2.0, discrete_num:int=5, round_num:int=10):
        shape = 40
        r_center = 6
        r = [r_center] + [(shape*0.5 - r_center)/(discrete_num-1) for i in range(1,discrete_num)]
        x, y = np.meshgrid(np.arange(shape), np.arange(shape), indexing='ij')
        xc = x - (shape-1)/2.0
        yc = y - (shape-1)/2.0
        rc = (xc**2+ yc**2)**0.5
        theta = np.where(yc>=0.0,np.arccos(xc/rc),- np.arccos(xc/rc))

        theta_1,theta_2 = theta_c - 2*np.pi/round_num/2, theta_c + 2*np.pi/round_num/2
        if theta_2>np.pi: theta_2 = theta_2 - np.pi*2

        r_1, r_2 = np.sum(r[:area_idx]), np.sum(r[:area_idx+1])
        
        mask = (r_1<rc) & (rc<=r_2)
        if area_idx!=0:
            if theta_2<0.0 and theta_1>0.0:
                mask &= (theta_1<theta) | (theta<=theta_2)
            else:
                mask &= (theta_1<theta) & (theta<=theta_2)
        return np.stack([x[mask],y[mask]],axis=0)