'''
@File    :   extract_2dpose.py
@Time    :   2024/03/23 3:00:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   extract 2d pose from dataset
'''

import os
import argparse
import numpy as np 

CS_train_V1 = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 
                61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 
                106, 110, 111, 112, 114, 115, 116, 117, 118]

CS_train_V2 = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 
                26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 
                49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 
                72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 
                92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 
                108, 109, 110, 111, 112, 113, 114, 115, 117, 118]

def extract_pose(ske_txt_path: str) -> np.ndarray:
    with open(ske_txt_path, 'r') as f: 
        num_frame = int(f.readline()) # the frame num
        joint_data = [] # T M V C
        for t in range(num_frame): # for each frame
            num_body = int(f.readline()) # the body num
            one_frame_data = np.zeros((num_body, 17, 2)) # M 17 2 
            for m in range(num_body): # for each body
                f.readline() # skip this line, e.g. 000 0 0 0 0 0 0 0 0 0
                num_joints = int(f.readline()) # the num joins, equal to 17
                assert num_joints == 17
                for v in range(num_joints): # for each joint
                    xy = np.array(f.readline().split()[:2], dtype = np.float64)
                    one_frame_data[m, v] = xy
            joint_data.append(one_frame_data)
        joint_data = np.array(joint_data)  
    return joint_data # T M 17 2 

def get_max_frame(root_Skeleton_path:str, samples_txt: list) -> int:
    max_frame = 0
    for idx, sample in enumerate(samples_txt):
        ske_path = root_Skeleton_path + '/' + sample
        with open(ske_path, 'r') as f:
            cur_frame = int(f.readline()) # the frame num
            if cur_frame > max_frame: max_frame = cur_frame
    return max_frame 

def main(root_Skeleton_path :str) -> None:
    samples_txt = sorted(os.listdir(root_Skeleton_path))
    max_frame = get_max_frame(root_Skeleton_path, samples_txt) # get the max frames
    # assert max_frame == 305
        
    # V1
    # CS_train_V1_data = []
    CS_test_V1_data = []
    # CS_train_V1_label = []
    CS_test_V1_label = []
    CS_test_V1_sample = []
    
    # V2
    # CS_train_V2_data = []
    CS_test_V2_data = []
    # CS_train_V2_label = []
    CS_test_V2_label = []
    CS_test_V2_sample = []
    
    max_body = 2
    tgt_shape = (max_frame, max_body, 17, 2) # T M V C
    
    for idx, sample in enumerate(samples_txt):
        print("process ", sample)
        subj_id = int(sample[1:4]) # get subject id
        label_id = int(sample.split('A')[1][:3])
        label = label_id
        
        ske_path = root_Skeleton_path + '/' + sample
        joint_data = extract_pose(ske_path)
        
        # padd to the same shape
        pad_width = [(0, tgt_shape[i] - joint_data.shape[i]) for i in range(len(tgt_shape))]
        pad_data = np.pad(joint_data, pad_width, mode = 'constant', constant_values = 0)
        assert pad_data[:joint_data.shape[0], :joint_data.shape[1], :, :].all() == joint_data.all()
        
        # V1
        # if subj_id in CS_train_V1:
        #     CS_train_V1_data.append(pad_data)
        #     CS_train_V1_label.append(label)
        # else:
        #     CS_test_V1_data.append(pad_data)
        #     CS_test_V1_label.append(label)
        
        # # V2    
        # if subj_id in CS_train_V2:
        #     CS_train_V2_data.append(pad_data)
        #     CS_train_V2_label.append(label)
        # else:
        #     CS_test_V2_data.append(pad_data)
        #     CS_test_V2_label.append(label)
        
        # V1
        if subj_id not in CS_train_V1:
            CS_test_V1_data.append(pad_data)
            CS_test_V1_label.append(label)
            CS_test_V1_sample.append(sample)
            
        # V2
        if subj_id not in CS_train_V2:
            CS_test_V2_data.append(pad_data)
            CS_test_V2_label.append(label)
            CS_test_V2_sample.append(sample)
     
    # V1
    # CS_train_V1_data = np.array(CS_train_V1_data)
    CS_test_V1_data = np.array(CS_test_V1_data)
    # CS_train_V1_label = np.array(CS_train_V1_label)
    CS_test_V1_label = np.array(CS_test_V1_label)
    
    # V2
    # CS_train_V2_data = np.array(CS_train_V2_data)
    CS_test_V2_data = np.array(CS_test_V2_data)
    # CS_train_V2_label = np.array(CS_train_V2_label)
    CS_test_V2_label = np.array(CS_test_V2_label)
        
    # np.savez('./pose_data/V1.npz', x_train = CS_train_V1_data, y_train = CS_train_V1_label, 
    #          x_test = CS_test_V1_data, y_test = CS_test_V1_label)
    # np.savez('./pose_data/V2.npz', x_train = CS_train_V2_data, y_train = CS_train_V2_label, 
    #          x_test = CS_test_V2_data, y_test = CS_test_V2_label)
    
    np.savez('./save_2d_pose/V1.npz', x_test = CS_test_V1_data, y_test = CS_test_V1_label)
    np.savez('./save_2d_pose/V2.npz', x_test = CS_test_V2_data, y_test = CS_test_V2_label)
    np.savetxt('./CS_test_V1.txt', CS_test_V1_sample, fmt = "%s")    
    np.savetxt('./CS_test_V2.txt', CS_test_V2_sample, fmt = "%s")  
    
    print("All done!")

def get_parser():
    parser = argparse.ArgumentParser(description = 'extract_2dpose_from_test_dataset') 
    parser.add_argument(
        '--test_dataset_path', 
        type = str,
        default = '../Test_dataset') # It's better to use absolute paths.
    return parser

# python extract_2dpose.py --test_dataset_path ../Test_dataset             
if __name__ == "__main__":
    # root_Skeleton_path = '../Test_dataset'
    parser = get_parser()
    args = parser.parse_args()
    root_Skeleton_path = args.test_dataset_path # It's better to use absolute paths.
    main(root_Skeleton_path)