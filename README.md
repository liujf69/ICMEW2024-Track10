# ICMEW2024-Track10
ICMEW2024-Track #10: Skeleton-based Action Recognition

# Dataset
**1. Put your test dataset into the ```Test_dataset``` folder.**
```
Please note that, the **naming** of the test dataset must comply with the validation datasets issued by the competition,
e.g. P000S00G10B10H10UC022000LC021000A000R0_08241716.txt. (This is a training data sample because subject_id is **000**)
e.g. P001S00G20B40H20UC072000LC021000A000R0_08241838.txt. (This is a **CSv1** validation data sample because subject_id is **001**)
e.g. P002S00G10B50H30UC062000LC092000A000R0_08250945.txt. (This is a **CSv2** validation data sample because subject_id is **002**)
```
**2. Extract 2D Pose from the test dataset you provided, run the following code:**
```
cd Process_data
python extract_2dpose.py --test_dataset_path ../Test_dataset
```
Please note that, the path **../Test_dataset** is the path of the test dataset in the first step, and we recommend using an absolute path. <br />
After running this code, we will generate two files named **V1.npz** and **V2.npz** in the **Process_data/save_2d_pose** folder.

**3. Estimate 3d pose from 2d pose.** <br />
First, you must download the 3d pose checkpoint from [here](https://drive.google.com/file/d/1citX7YlwaM3VYBYOzidXSLHb4lJ6VlXL/view?usp=sharing). <br />
Then, you must put the downloaded checkpoint into the **./Process_data/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite** folder. <br />
Finally, you must run the following code:
```
cd Process_data
python estimate_3dpose.py --test_dataset_path ../Test_dataset
```
After running this code, we will generate two files named **V1.npz** and **V2.npz** in the **Process_data/save_3d_pose** folder.

# Model inference
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
ctrgcn_V1_JM_3d.pt:
ctrgcn_V1_BM_3d.pt:
```
**CSv2:**
```
ctrgcn_V2_J.pt: 69.00%
ctrgcn_V2_B.pt: 68.68%
ctrgcn_V2_JM.pt: 57.93%
ctrgcn_V2_BM.pt: 
ctrgcn_V2_J_3d.pt: 
ctrgcn_V2_B_3d.pt: 
ctrgcn_V2_JM_3d.pt:
ctrgcn_V2_BM_3d.pt:
```
# Ensemble

# Contact
For any questions, feel free to contact: liujf69@mail2.sysu.edu.cn
