import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from edge_sam import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import os
from pick_env.utils import save_states, load_states
import torch
from robot_grasping_sim.env.states import State, InternalState
from typing import List
import matplotlib.patches as patches
from pick_env.perceptor.vision_perception import VisionPerceptor


def get_images_from_states(states:List[State]):
    def convert(img, normalized:bool=False):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)

        if img.max()>1.0 and normalized:
            img = img / 255.0
        return img.astype(np.uint8)
    
    images = []
    for i,state in enumerate(states):
        image = convert(state.wrist_camera_rgb_image,False)
        images.append(image)
    
    return images

# class Segmenter():
#     def __init__(self) -> None:
#         self.sam = sam_model_registry["edge_sam"](checkpoint="EdgeSAM/checkpoints/edge_sam_3x.pth")
#         self.sam.to(device="cuda")
#         self.predictor = SamPredictor(self.sam)
#         self.history = []
#         self.click_points = []
#         self.init_bbox = None

#         pass

#     def reset(self):
#         self.history = []
#         self.click_points = []
#         pass

#     def select_object(self, img):
#         self.select_object_fig, self.select_object_ax = plt.subplots()
#         self.select_object_ax.imshow(img)
#         self.select_object_fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, self.select_object_ax, img))
#         plt.show()

#     def get_bbox(self, masks:List, expand=True, expand_size = 10):
#         # Get the coordinates of the non-zero (True) mask entries
#         rows = np.any(masks[0], axis=1)  # Detect rows with at least one '1'
#         cols = np.any(masks[0], axis=0)  # Detect columns with at least one '1'

#         # Get the bounding box coordinates (min and max rows and columns)
#         y_min, y_max = np.where(rows)[0][[0, -1]]
#         x_min, x_max = np.where(cols)[0][[0, -1]]

#         if expand:
#             H,W = masks[0].shape
#             x_min = max(0,x_min-expand_size)
#             y_min = max(0,y_min-expand_size)
#             x_max = min(W,x_max+expand_size)
#             y_max = min(H,y_max+expand_size)

#         return np.array([x_min, y_min, x_max, y_max])
    
#     def get_centroid(self,mask,last_centroid):
#         true_indices = np.array(np.nonzero(mask))
#         if true_indices.shape[1] > 0:
#             centroid = true_indices.mean(axis=1)
#             return np.flip(centroid).reshape(-1,2)
#         else:
#             return last_centroid

#     def __call__(self, img):
#         if len(self.history) == 0:
#             # First segment
#             # self.predictor.set_image(img)
#             # masks, _, _ = self.predictor.predict(
#             #     point_coords=np.array(self.click_points),
#             #     point_labels=np.ones(len(self.click_points))
#             #     )
#             # self.history.append(np.array(self.click_points)[:1])
#             # self.history.append(self.get_centroid(mask=masks[0],last_centroid=self.history[-1]))
#             self.predictor.set_image(img)
#             masks, _, _ = self.predictor.predict(
#                 box=self.init_bbox
#                 )
            
#             self.history.append(self.get_bbox(masks, expand=True))
#             # self.history.append(cv2.resize(masks[0].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool))
#         else:
#             # Streaming segment based on previous bbox
#             self.predictor.set_image(img)
#             # masks, _, _ = self.predictor.predict(
#             #     point_coords=np.array(self.history[-1]),
#             #     point_labels=np.ones(1)
#             #     )
#             # self.history.append(self.get_centroid(mask=masks[0],last_centroid=self.history[-1]))
#             masks, _, _ = self.predictor.predict(
#                 box=self.history[-1]
#                 )
#             self.history.append(self.get_bbox(masks, expand=True))
#             # masks, _, _ = self.predictor.predict(
#             #     mask_input=self.history[-1][np.newaxis,...]
#             #     )
            
#         return masks

#     def on_click(self, event, ax, img):
#         # Get the coordinates of the mouse click
#         x, y = int(event.xdata), int(event.ydata)
        
#         # Save the coordinates
#         print(f"Nr.{len(self.click_points)+1} Clicked at: ({x}, {y})")
#         self.click_points.append((x,y))

#         # Overlay the coordinates on the image
#         ax.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=12)
#         circle = patches.Circle((x, y), radius=5, transform=ax.transAxes, color='red', fill=True)
#         ax.add_patch(circle)

#         if len(self.click_points)==4:
#             xs = np.array([self.click_points[i][0] for i in range(len(self.click_points))])
#             ys = np.array([self.click_points[i][1] for i in range(len(self.click_points))])
#             x_min = np.min(xs)
#             y_min = np.min(ys)
#             x_max = np.max(xs)
#             y_max = np.max(ys)
#             self.init_bbox = np.array([x_min, y_min, x_max, y_max])

#         # Redraw the image with the new annotation
#         ax.imshow(img)
#         plt.draw()


# def overlay_masks_on_image(img, masks, save_path, alpha=0.5):
#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Show the original image
#     ax.imshow(img)

#     mask = np.zeros([*masks[0].shape,3])
#     mask[...,0] = masks[0]
#     mask = mask.astype(int)*255

#     # Overlay the mask on the image
#     ax.imshow(mask, alpha=alpha)  # Use the mask with transparency
#     fig.savefig(save_path)
#     plt.close()



import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from copy import deepcopy

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.image_pub1 = self.create_publisher(Image, '/img', 10)
        self.image_pub2 = self.create_publisher(Image, '/mask', 10)
        self.bridge = CvBridge()
        self.vp:VisionPerceptor = None

def main():
    rclpy.init()
    node = ImagePublisherNode()
    node.vp = VisionPerceptor()
    node.vp.start()
    time.sleep(1)


    # segmenter = Segmenter()
    # segmenter.reset()

    # test_states = load_states(path="data/recorded_trajectory/1_0-08_0-024_duplo-3-blocks.npy", device="cuda")
    # imgs = get_images_from_states(test_states)
    # imgs = imgs[30:]


    node.vp.select_object()

    times = []
    try:
        while rclpy.ok():
            img = node.vp.last_state[0].rgb
            mask = node.vp.last_state[0].mask

            mask = np.stack([mask,mask,mask],axis=-1).astype(np.uint8)*85
        
            # overlay_masks_on_image(img=img,masks=[masks[0]],save_path=f"data/recorded_trajectory/segmentation_result/{i}.png")
            ros_image1 = node.bridge.cv2_to_imgmsg(img, encoding="rgb8")
            ros_image2 = node.bridge.cv2_to_imgmsg(mask, encoding="rgb8")
            node.image_pub1.publish(ros_image1)
            node.image_pub2.publish(ros_image2)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

