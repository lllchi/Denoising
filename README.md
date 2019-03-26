# Denoising

Code for  NTIRE 2019 Real Image Denoising Challenge

## Dependencies

- python = 3.6
- pytorch >= 1.0
- Numpy
- h5py


## Training Instruction

### generate training data:

python train.py    --prepocess True  --root 'Set the full path of /sRGB/SIDD_Medium_Srgb/'  --name RDN

example usage: python train.py --prepocess False  --root /data0/lichi/denoising/sRGB/SIDD_Medium_Srgb/ --name RDN

### start training:

python train.py    --prepocess False

## Validation Instruction

python validate.py  --name RDN_e40_16  --which_model final_net.pth  --test_path data/BenchmarkNoisyBlocksSrgb.mat
