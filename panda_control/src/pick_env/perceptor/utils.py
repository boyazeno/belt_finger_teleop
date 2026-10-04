import os
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAMSegmentor:
    def __init__(self) -> None:
        # Set SAM2_ROOT to a local clone of segment-anything-2 with downloaded checkpoints.
        self.checkpoint = os.path.join(os.environ.get("SAM2_ROOT", "segment-anything-2"), "checkpoints", "sam2_hiera_large.pt")
        self.model_cfg = "sam2_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.checkpoint))

    def segment(self, img:np.ndarray, key_pt:np.ndarray)->np.ndarray:
        # input need to be HWC, 0-255
        input_point = key_pt # N*3
        input_label = np.array([1]*input_point.shape[0]) # N*1, 1 for select, 0 for not select
        self.predictor.set_image(img)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        return masks[0]>0.0 # HW
    
if __name__ == "__main__":
    import sys
    from PIL import Image
    import matplotlib.pyplot as plt 
    segmenter = SAMSegmentor()
    image = Image.open(sys.argv[1]) # path to a test RGB image
    image = np.array(image.convert("RGB"))
    input_point = np.array([[500, 375]])
    plt.imshow(segmenter.segment(image, input_point))
    plt.show()