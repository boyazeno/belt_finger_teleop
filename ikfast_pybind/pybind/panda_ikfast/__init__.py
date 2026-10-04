from .ikfast_pybind import get_ik as bind_get_ik
import numpy as np
from typing import List, Tuple
import pytransform3d.rotations as pr

def test():
    target = np.eye(4, dtype=np.float64)
    target[:3,:3] = pr.matrix_from_quaternion([0.053535, 0.298545, 0.952881, -0.004759])
    target[:3, 3] = [0.5, 0.5, 0.5]
    ref = [0,0,0,0,0,0,0]
    max_val = [3.14,3.14,3.14,3.14,3.14,3.14,3.14]
    min_val = [-3.14,-3.14,-3.14,-3.14,-3.14,-3.14,-3.14]
    success, joint_values = bind_get_ik(target,ref,max_val,min_val)
    print(f"success: {success}, joint_values: {joint_values}")

def get_ik(target:np.ndarray=np.eye(4, dtype=np.float64), ref: List[float]=[0.,0.,0.,0.,0.,0.,0.], max_val:List[float] = [2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973], min_val:List[float] = [-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])->Tuple[bool, List[float]]:
    assert target.shape == (4,4)
    assert target.dtype == np.float64
    assert len(ref) == 7
    assert isinstance(ref, list)
    assert len(max_val) == 7
    assert isinstance(max_val, list)
    assert len(min_val) == 7
    assert isinstance(min_val, list)
    success, joint_values = bind_get_ik(target,ref, max_val, min_val)
    return success, np.array(joint_values)