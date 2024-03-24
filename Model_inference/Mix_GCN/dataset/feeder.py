import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import tools

coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]

class Feeder(Dataset):
    def __init__(self, data_path: str, data_split: str, p_interval: list=[0.95], window_size: int=64, bone: bool=False, vel: bool=False):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.load_data()
        
    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.data_split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        else:
            assert self.data_split == 'test'
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        data_numpy = self.data[idx] # T M V C
        label = self.label[idx]
        data_numpy = torch.from_numpy(data_numpy).permute(3, 0, 2, 1) # C,T,V,M
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if(valid_frame_num == 0): 
            return np.zeros((2, 64, 17, 2)), label, idx
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1)) # all_joint - 0_joint
        return data_numpy, label, idx # C T V M
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
# if __name__ == "__main__":
    # Debug
    # train_loader = torch.utils.data.DataLoader(
    #             dataset = Feeder(data_path = '/data-home/liujinfu/MotionBERT/pose_data/V1.npz', data_split = 'train'),
    #             batch_size = 4,
    #             shuffle = True,
    #             num_workers = 2,
    #             drop_last = False)
    
    # val_loader = torch.utils.data.DataLoader(
    #         dataset = Feeder(data_path = '/data-home/liujinfu/MotionBERT/pose_data/V1.npz', data_split = 'test'),
    #         batch_size = 4,
    #         shuffle = False,
    #         num_workers = 2,
    #         drop_last = False)
    
    # for batch_size, (data, label, idx) in enumerate(train_loader):
    #     data = data.float() # B C T V M
    #     label = label.long() # B 1
    #     print("pasue")