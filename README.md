# DentalSeg: A Solution for ToothFairy2 Challenge
The proposed solution is based on [nnU-Net v2 framework](https://github.com/MIC-DKFZ/nnUNet). The inference code has been rewritten to enable the algorithm to run easily and rapidly.
## Usage
Following these steps to integrate DentalSeg with nn-UNet:
1. Download and install nnUNetv2 using the command
 ```
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
```
2. Copy the loss functions and network training code files to the corresponding directories in nnUNet using the following commands:
 ```
 cp DentalSeg/loss/* nnUNet/nnunetv2/training/loss/
 cp DentalSeg/nnUNetTrainer/* nnUNet/nnunetv2/training/nnUNetTrainer/
 ```
3. Our solution includes two stages, which means that we need to train 2 networks. Please follow the official commands of nnUNetv2.
   
   **Network1 Experiment Planning and Preprocessing**
    ```
    nnUNetv2_plan_and_preprocess -d [d1]  -c 3d_fullres -np 4
    ```
   **Network1 Training**
   ```
   nnUNetv2_train [d1]  3d_fullres [fold] 
   ```
   **Network2 Experiment Planning and Preprocessing**
   ```
   nUNetv2_plan_and_preprocess -d [d2]  -c 3d_fullres -np 4
   ```
   **Network2 Training**
   ```
   nnUNetv2_train [d2]  3d_fullres [fold] 
   ```
   **Network2 Finetuning**
   ```
   nnUNetv2_train [d2] 3d_fullres [fold] -tr nnUNetTrainer_Tversky_no_mirror
   ```
To inference by running the commond after modifying the file address in [inference.py](https://github.com/Mors20/DentalSeg/blob/main/inference.py):
```
python inference.py
```
