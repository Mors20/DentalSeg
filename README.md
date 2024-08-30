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
To inference by running the commond after modifying the file address in [inference.py](https://github.com/Mors20/DentalSeg/blob/main/inference.py):
```
python inference.py
```
