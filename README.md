# ICMEW2024-Track10
ICMEW2024-Track #10: Skeleton-based Action Recognition. <br />
![image](https://github.com/liujf69/ICMEW2024-Track10/blob/main/framework.png)
Please install the environment based on **mix_GCN.yml**, **pose3d.yml**, **mix_former.yml**. <br />
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
```
**2. Verification report of the UAV dataset** <br />
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
#
tdgcn_V1_J.pt: 43.21%
tdgcn_V1_B.pt: 43.33%
tdgcn_V1_JM.pt: 35.74%
tdgcn_V1_BM.pt: 35.56%
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
ctrgcn_V2_BM_3d.pt: 53.98%
#
tdgcn_V2_J.pt: 69.50%
tdgcn_V2_B.pt: 69.30%
tdgcn_V2_JM.pt: 57.74%
tdgcn_V2_BM.pt:
```
## Run Mix_Former
...
# Ensemble
## Ensemble Mix_GCN
**1.** After running the code of model inference (i.e. **Run Mix_GCN**), we will obtain classification score files corresponding to each weight in the **output folder** named **epoch1_test_score.pkl**. <br />
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
--val_sample ./Process_data/CS_test_V2.txt \
--benchmark V2
```
Please note that when running the above code, you may need to carefully **check the paths** for each **epoch1_test_score.pkl** file and the **val_sample** to prevent errors. And the **CS_test_V1.txt** and **CS_test_V2.txt** are generated when running **python extract_2dpose.py**. <br />
```
**CSv1:** Emsemble Mix_GCN: 45.71%
**CSv2:** Emsemble Mix_GCN: 73.15%
```
## Ensemble Mix_Former
...
## Ensemble Mix_GCN and Mix_Former
...
# Suggestion
We recommend comprehensively considering the three ensemble results **Ensemble Mix_GCN**, **Ensemble Mix_Former**, and **Ensemble Mix_GCN and Mix_Former**.
# Contact
```
Author List: Jinfu Liu, Baiqiao Yin, Jiaying Lin, Jiajun Wen, Yue Li, Mengyuan Liu.
```
For any questions, feel free to contact: liujf69@mail2.sysu.edu.cn
