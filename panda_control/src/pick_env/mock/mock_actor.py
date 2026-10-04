import numpy as np
import torch
from robot_grasping_sim.env.states import State, InternalState



class MockActor:
    def __init__(self, real_actor) -> None:
        self.real_actor = real_actor
        pass

    def get_register_func(self,func_name):
        return self.real_actor.get_register_func(func_name=func_name)
    
    def inference(self, s):
        real_a = self.real_actor.inference(s=s)

        # mock_a = torch.ones_like(real_a) * 0.001
        return real_a

    @property
    def dim_s(self):
        return self.real_actor.dim_s

    @property
    def max_sequence_length(self):
        return self.real_actor.max_sequence_length