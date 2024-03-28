import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--ctrgcn_J2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_B2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_JM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_BM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_J3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_B3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_JM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--ctrgcn_BM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_J2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_B2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_JM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl'),
    parser.add_argument(
        '--tdgcn_BM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_J2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_B2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_JM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl'),
    parser.add_argument(
        '--mstgcn_BM2d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl'),
    parser.add_argument(
        '--val_sample', 
        type = str,
        default = './Process_data/CS_test_V1.txt'),
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'V1')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype = str)
    for idx, name in enumerate(val_txt):
        label = int(name.split('A')[1][:3])
        true_label.append(label)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Mix_GCN Score File
    j_file = args.ctrgcn_J2d_Score
    b_file = args.ctrgcn_B2d_Score
    jm_file = args.ctrgcn_JM2d_Score
    bm_file = args.ctrgcn_BM2d_Score
    j3d_file = args.ctrgcn_J3d_Score
    b3d_file = args.ctrgcn_B3d_Score
    jm3d_file = args.ctrgcn_JM3d_Score
    bm3d_file = args.ctrgcn_BM3d_Score
    
    td_j_file = args.tdgcn_J2d_Score
    td_b_file = args.tdgcn_B2d_Score
    td_jm_file = args.tdgcn_JM2d_Score
    td_bm_file = args.tdgcn_BM2d_Score
    
    mst_j_file = args.mstgcn_J2d_Score
    mst_b_file = args.mstgcn_B2d_Score
    mst_jm_file = args.mstgcn_JM2d_Score
    mst_bm_file = args.mstgcn_BM2d_Score
    
    val_txt_file = args.val_sample

    File = [j_file, b_file, jm_file, bm_file, 
            j3d_file, b3d_file, jm3d_file, bm3d_file, 
            td_j_file, td_b_file, td_jm_file, td_bm_file,
            mst_j_file, mst_b_file, mst_jm_file, mst_bm_file]    
    if args.benchmark == 'V1':
        Numclass = 155
        Sample_Num = 6307
        Rate = [0.7, 0.7, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3] 
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label(val_txt_file)
    
    if args.benchmark == 'V2':
        Numclass = 155
        Sample_Num = 6599
        Rate = [0.7, 0.7, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3,
                0.05, 0.05, 0.05, 0.05] 
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label(val_txt_file)
    
    Acc = Cal_Acc(final_score, true_label)

    print('acc:', Acc)
