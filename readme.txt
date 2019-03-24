Dependencies
python 3.6
Pytorch >= 1.0.0
numpy 
h5py


Training Instruction

1.generate training data:

python train.py    --prepocess True  --root 'Set the full path of /sRGB/SIDD_Medium_Srgb/'  --name RDN

example usage: python train.py --prepocess False  --root /data0/lichi/denoising/sRGB/SIDD_Medium_Srgb/ --name RDN

2. start training:

python train.py    --prepocess False


Validation Instruction

python validate.py  --name RDN_e40_16  --which_model final_net.pth  --validate_path data/ValidationNoisyBlocksSrgb.mat








