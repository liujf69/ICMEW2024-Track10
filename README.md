# ICMEW2024-Track10
This is the official repo of **HDBN** and our work is one of the **top solutions** in the Multi-Modal Video Reasoning and Analyzing Competition (**MMVRAC**) of **2024 ICME** Grand Challenge **Track10**. <br />
Our work "[HDBN: A Novel Hybrid Dual-branch Network for Robust Skeleton-based Action Recognition](https://ieeexplore.ieee.org/document/10645450)" is accepted by **2024 IEEE International Conference on Multimedia and Expo Workshop (ICMEW)**. <br />
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2404.15719) <br />
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hdbn-a-novel-hybrid-dual-branch-network-for/skeleton-based-action-recognition-on-uav)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-uav?p=hdbn-a-novel-hybrid-dual-branch-network-for)
# Framework
![image](https://github.com/liujf69/ICMEW2024-Track10/blob/main/framework.png)
Please install the environment based on **mix_GCN.yml**. <br />

# Dataset
**1. Put your test dataset into the ```Test_dataset``` folder.**
```
Please note that, the **naming** of the test dataset must comply with the validation datasets issued by the competition,
e.g. P000S00G10B10H10UC022000LC021000A000R0_08241716.txt. (This is a training data sample because subject_id is **000**)
e.g. P001S00G20B40H20UC072000LC021000A000R0_08241838.txt. (This is a **CSv1** validation data sample because subject_id is **001**)
e.g. P002S00G10B50H30UC062000LC092000A000R0_08250945.txt. (This is a **CSv2** validation data sample because subject_id is **002**)
```
**2. Extract 2d pose from the test dataset you provided, run the following code:**
```
cd Process_data
python extract_2dpose.py --test_dataset_path ../Test_dataset
```
Please note that, the path **../Test_dataset** is the path of the test dataset in the first step, and we recommend using an absolute path. <br />
After running this code, we will generate two files named **V1.npz** and **V2.npz** in the **Process_data/save_2d_pose** folder. <br />

**3. Estimate 3d pose from 2d pose.** <br />
First, you must download the 3d pose checkpoint from [here](https://drive.google.com/file/d/1citX7YlwaM3VYBYOzidXSLHb4lJ6VlXL/view?usp=sharing), and install the environment based on **pose3d.yml** <br />
Then, you must put the downloaded checkpoint into the **./Process_data/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite** folder. <br />
Finally, you must run the following code:
```
cd Process_data
python estimate_3dpose.py --test_dataset_path ../Test_dataset
```
After running this code, we will generate two files named **V1.npz** and **V2.npz** in the **Process_data/save_3d_pose** folder. <br />

# Model inference
## Run Mix_GCN
Copy the **Process_data/save_2d_pose** folder and the **Process_data/save_3d_pose** folder to **Model_inference/Mix_GCN/dataset**:
```
cp -r ./Process_data/save_2d_pose Model_inference/Mix_GCN/dataset
cp -r ./Process_data/save_3d_pose Model_inference/Mix_GCN/dataset
cd ./Model_inference/Mix_GCN
pip install -e torchlight
```

**1. Run the following code separately to obtain classification scores using different model weights.** <br />
**CSv1:**
```
python main.py --config ./config/ctrgcn_V1_J.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_J.pt --device 0
python main.py --config ./config/ctrgcn_V1_B.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_B.pt --device 0
python main.py --config ./config/ctrgcn_V1_JM.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_JM.pt --device 0
python main.py --config ./config/ctrgcn_V1_BM.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_BM.pt --device 0
python main.py --config ./config/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_J_3d.pt --device 0
python main.py --config ./config/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_B_3d.pt --device 0
python main.py --config ./config/ctrgcn_V1_JM_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_JM_3d.pt --device 0
python main.py --config ./config/ctrgcn_V1_BM_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V1_BM_3d.pt --device 0
###
python main.py --config ./config/tdgcn_V1_J.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V1_J.pt --device 0
python main.py --config ./config/tdgcn_V1_B.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V1_B.pt --device 0
python main.py --config ./config/tdgcn_V1_JM.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V1_JM.pt --device 0
python main.py --config ./config/tdgcn_V1_BM.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V1_BM.pt --device 0
###
python main.py --config ./config/mstgcn_V1_J.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V1_J.pt --device 0
python main.py --config ./config/mstgcn_V1_B.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V1_B.pt --device 0
python main.py --config ./config/mstgcn_V1_JM.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V1_JM.pt --device 0
python main.py --config ./config/mstgcn_V1_BM.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V1_BM.pt --device 0
```
**CSv2:**
```
python main.py --config ./config/ctrgcn_V2_J.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_J.pt --device 0
python main.py --config ./config/ctrgcn_V2_B.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_B.pt --device 0
python main.py --config ./config/ctrgcn_V2_JM.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_JM.pt --device 0
python main.py --config ./config/ctrgcn_V2_BM.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_BM.pt --device 0
python main.py --config ./config/ctrgcn_V2_J_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_J_3d.pt --device 0
python main.py --config ./config/ctrgcn_V2_B_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_B_3d.pt --device 0
python main.py --config ./config/ctrgcn_V2_JM_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_JM_3d.pt --device 0
python main.py --config ./config/ctrgcn_V2_BM_3d.yaml --phase test --save-score True --weights ./checkpoints/ctrgcn_V2_BM_3d.pt --device 0
###
python main.py --config ./config/tdgcn_V2_J.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V2_J.pt --device 0
python main.py --config ./config/tdgcn_V2_B.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V2_B.pt --device 0
python main.py --config ./config/tdgcn_V2_JM.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V2_JM.pt --device 0
python main.py --config ./config/tdgcn_V2_BM.yaml --phase test --save-score True --weights ./checkpoints/tdgcn_V2_BM.pt --device 0
###
python main.py --config ./config/mstgcn_V2_J.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V2_J.pt --device 0
python main.py --config ./config/mstgcn_V2_B.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V2_B.pt --device 0
python main.py --config ./config/mstgcn_V2_JM.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V2_JM.pt --device 0
python main.py --config ./config/mstgcn_V2_BM.yaml --phase test --save-score True --weights ./checkpoints/mstgcn_V2_BM.pt --device 0
```
**2. Verification report of the UAV dataset** <br />
To verify the correctness of your handling of the dataset, you can use the validation set from the original UAV-Human dataset to test the checkpoints above, and we provide the corresponding recognition accuracy below. <br />
**CSv1:**
```
ctrgcn_V1_J.pt: 43.52%
ctrgcn_V1_B.pt: 43.32%
ctrgcn_V1_JM.pt: 36.25%
ctrgcn_V1_BM.pt: 35.86%
ctrgcn_V1_J_3d.pt: 35.14%
ctrgcn_V1_B_3d.pt: 35.66%
ctrgcn_V1_JM_3d.pt: 31.08%
ctrgcn_V1_BM_3d.pt: 30.89%
###
tdgcn_V1_J.pt: 43.21%
tdgcn_V1_B.pt: 43.33%
tdgcn_V1_JM.pt: 35.74%
tdgcn_V1_BM.pt: 35.56%
###
mstgcn_V1_J.pt: 41.48%
mstgcn_V1_B.pt: 41.57%
mstgcn_V1_JM.pt: 33.82%
mstgcn_V1_BM.pt: 34.74%
```
**CSv2:**
```
ctrgcn_V2_J.pt: 69.00%
ctrgcn_V2_B.pt: 68.68%
ctrgcn_V2_JM.pt: 57.93%
ctrgcn_V2_BM.pt: 58.45%
ctrgcn_V2_J_3d.pt: 64.60%
ctrgcn_V2_B_3d.pt: 63.25%
ctrgcn_V2_JM_3d.pt: 55.80%
ctrgcn_V2_BM_3d.pt: 54.67%
###
tdgcn_V2_J.pt: 69.50%
tdgcn_V2_B.pt: 69.30%
tdgcn_V2_JM.pt: 57.74%
tdgcn_V2_BM.pt: 55.14%
###
mstgcn_V2_J.pt: 67.48%
mstgcn_V2_B.pt: 67.30%
mstgcn_V2_JM.pt: 54.43%
mstgcn_V2_BM.pt: 52.13%
```
## Run Mix_Former
Copy the **Process_data/save_2d_pose**  folder to **Model_inference/Mix_Former/dataset**:
```
cd ./Model_inference/Mix_Former
```
**1. Run the following code separately to obtain classification scores using different model weights.** <br />
**CSv1:** <br />
You have to change the corresponding **data-path** in the **config file**, just like：**data_path: dataset/save_2d_pose/V1.npz**. we recommend using an absolute path.
```
python main.py --config ./config/mixformer_V1_J.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_J.pt --device 0  
python main.py --config ./config/mixformer_V1_B.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_B.pt --device 0 
python main.py --config ./config/mixformer_V1_JM.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_JM.pt --device 0 
python main.py --config ./config/mixformer_V1_BM.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_BM.pt --device 0 
python main.py --config ./config/mixformer_V1_k2.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_k2.pt --device 0 
python main.py --config ./config/mixformer_V1_k2M.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V1_k2M.pt --device 0 
```
**CSv2:** <br />
You have to change the corresponding **data-path** in the **config file**, just like：**data_path: dataset/save_2d_pose/V2.npz**. we recommend using an absolute path.
```
python main.py --config ./config/mixformer_V2_J.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_J.pt --device 0 
python main.py --config ./config/mixformer_V2_B.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_B.pt --device 0 
python main.py --config ./config/mixformer_V2_JM.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_JM.pt --device 0 
python main.py --config ./config/mixformer_V2_BM.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_BM.pt --device 0 
python main.py --config ./config/mixformer_V2_k2.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_k2.pt --device 0 
python main.py --config ./config/mixformer_V2_k2M.yaml --phase test --save-score True --weights ./checkpoints/mixformer_V2_k2M.pt --device 0 
```

**2. Verification report of the UAV dataset** <br />
**CSv1:**
```
mixformer_V1_J.pt: 41.43%
mixformer_V1_B.pt: 37.40%
mixformer_V1_JM.pt: 33.41%
mixformer_V1_BM.pt: 30.24%
mixformer_V1_k2.pt: 39.21%
mixformer_V1_k2M.pt: 32.60%
```
**CSv2:**
```
mixformer_V2_J.pt: 66.03%
mixformer_V2_B.pt: 64.89%
mixformer_V2_JM.pt: 54.58%
mixformer_V2_BM.pt: 52.95%
mixformer_V2_k2.pt: 65.56%
mixformer_V2_k2M.pt: 55.01%
```

# Ensemble
## Ensemble Mix_GCN
**1.** After running the code of model inference, we will obtain classification score files corresponding to each weight in the **output folder** named **epoch1_test_score.pkl**. <br />
**2.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
python Ensemble_MixGCN.py \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V1.txt \
--benchmark V1
```
**3.** You can obtain the final classification accuracy of CSv2 by running the following code:
```
python Ensemble_MixGCN.py \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V2.txt \
--benchmark V2
```
Please note that when running the above code, you may need to carefully **check the paths** for each **epoch1_test_score.pkl** file and the **val_sample** to prevent errors. <br />
```
**CSv1:** Emsemble Mix_GCN: 46.73%
**CSv2:** Emsemble Mix_GCN: 74.06%
```
## Ensemble Mix_Former
**1.** After running the code of model inference, we will obtain classification score files corresponding to each weight in the **output folder** named **epoch1_test_score.pkl**. <br />
**2.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
pip install scikit-optimize
```
```
python Ensemble_MixFormer.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl \
--benchmark V1
```

**3.** You can obtain the final classification accuracy of CSv2 by running the following code:
```
python Ensemble_MixFormer.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V2_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V2_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V2_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V2_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2M/epoch1_test_score.pkl \
--benchmark V2
```
Please note that when running the above code, you may need to carefully **check the paths** for each **epoch1_test_score.pkl** file and the **val_sample** to prevent errors. <br />
```
**CSv1:** Emsemble MixFormer: 47.23%
**CSv2:** Emsemble MixFormer: 73.47%
```
## Ensemble Mix_GCN and Mix_Former

**1.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
python Ensemble.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V1.txt \
--benchmark V1
```
or
```
 python Ensemble2.py --benchmark V1 \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl \
--mixformer_JM_Score  ./Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl
```
**2.** You can obtain the final classification accuracy of CSv2 by running the following code:
```
python Ensemble.py \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V2_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V2_B/epoch1_test_score.pkl \
--mixformer_JM_Score ./Model_inference/Mix_Former/output/skmixf__V2_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V2_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl \
--val_sample ./Process_data/CS_test_V2.txt \
--benchmark V2
```
or
```
python Ensemble2.py --benchmark V2 \
--mixformer_J_Score ./Model_inference/Mix_Former/output/skmixf__V2_J/epoch1_test_score.pkl \
--mixformer_B_Score ./Model_inference/Mix_Former/output/skmixf__V2_B/epoch1_test_score.pkl \
--mixformer_JM_Score  ./Model_inference/Mix_Former/output/skmixf__V2_JM/epoch1_test_score.pkl \
--mixformer_BM_Score ./Model_inference/Mix_Former/output/skmixf__V2_BM/epoch1_test_score.pkl \
--mixformer_k2_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2/epoch1_test_score.pkl \
--mixformer_k2M_Score ./Model_inference/Mix_Former/output/skmixf__V2_k2M/epoch1_test_score.pkl \
--ctrgcn_J2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J/epoch1_test_score.pkl \
--ctrgcn_B2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B/epoch1_test_score.pkl \
--ctrgcn_JM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM/epoch1_test_score.pkl \
--ctrgcn_BM2d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM/epoch1_test_score.pkl \
--ctrgcn_J3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_J_3D/epoch1_test_score.pkl \
--ctrgcn_B3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_B_3D/epoch1_test_score.pkl \
--ctrgcn_JM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_JM_3D/epoch1_test_score.pkl \
--ctrgcn_BM3d_Score ./Model_inference/Mix_GCN/output/ctrgcn_V2_BM_3D/epoch1_test_score.pkl \
--tdgcn_J2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_J/epoch1_test_score.pkl \
--tdgcn_B2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_B/epoch1_test_score.pkl \
--tdgcn_JM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_JM/epoch1_test_score.pkl \
--tdgcn_BM2d_Score ./Model_inference/Mix_GCN/output/tdgcn_V2_BM/epoch1_test_score.pkl \
--mstgcn_J2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_J/epoch1_test_score.pkl \
--mstgcn_B2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_B/epoch1_test_score.pkl \
--mstgcn_JM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_JM/epoch1_test_score.pkl \
--mstgcn_BM2d_Score ./Model_inference/Mix_GCN/output/mstgcn_V2_BM/epoch1_test_score.pkl
```

```
**CSv1:** Emsemble: 47.95%
**CSv2:** Emsemble: 75.36%
```
# Suggestion
We recommend comprehensively considering the three ensemble results **Ensemble Mix_GCN**, **Ensemble Mix_Former**, and **Ensemble Mix_GCN and Mix_Former**.

# Model training
```
# Change the configuration file (.yaml) of the corresponding modality.
# Mix_GCN Example
cd ./Model_inference/Mix_GCN
python main.py --config ./config/ctrgcn_V1_J.yaml --device 0

# Mix_Former Example
cd ./Model_inference/Mix_Former
python main.py --config ./config/mixformer_V1_J.yaml --device 0
```

# Thanks
Our work is based on the [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [TD-GCN](https://github.com/liujf69/TD-GCN-Gesture), [MotionBERT](https://github.com/Walter0807/MotionBERT), [Ske-MixFormer](https://github.com/ElricXin/Skeleton-MixFormer).
# Citation
```
@inproceedings{liu2024HDBN,
  author={Liu, Jinfu and Yin, Baiqiao and Lin, Jiaying and Wen, Jiajun and Li, Yue and Liu, Mengyuan},
  title={HDBN: A Novel Hybrid Dual-branch Network for Robust Skeleton-based Action Recognition}, 
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo Workshop (ICMEW)}, 
  year={2024}
}
```
# Contact
For any questions, feel free to contact: liujf69@mail2.sysu.edu.cn
