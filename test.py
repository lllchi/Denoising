import cv2
import os
import argparse
import glob

from models.RDN import *
from utils import *

import torchvision.transforms as transforms

from skimage.measure.simple_metrics import compare_psnr


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--name", type=str, default="RDN_e40_ssim", help='model name')

parser.add_argument("--add_BN", type=bool, default=True, help='Batch Normalization')


parser.add_argument("--input_c", type=int, default=1, help="input image channel")
parser.add_argument("--output_c", type=int, default=1, help="output image channel")
parser.add_argument("--n_feat", type=int, default=64, help="middle features number")
parser.add_argument("--n_resgroups", type=int, default=5, help="residule group numbers")
parser.add_argument("--n_resblocks", type=int, default=10, help="residule attention block numbers")
parser.add_argument("--reduction", type=int, default=16, help="channel reduction")
parser.add_argument("--res_scale", type=int, default=1, help="res_scale")

opt = parser.parse_args()

n_path = 'data/noise.txt'
gt_path = 'data/gt.txt'
root = '/data0/lichi/denoising/sRGB/SIDD_Medium_Srgb/'

with open(n_path, 'r') as f:
    lines = f.readlines()
    noise_list = [
        i.strip('\n') for i in lines
    ]
with open(gt_path, 'r') as f:
    lines = f.readlines()
    gt_list = [
        i.strip('\n') for i in lines
    ]

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    #net = RCAN(opt)
    #net = DnCNN(3, opt.num_of_layers)
    #net = IDNet(opt)
    #net = DnCNN_Attn(channels=1, num_of_layers=opt.num_of_layers)
    #net = DRRN(opt)
    #net = CASN(channels=1, num_of_layers=opt.num_of_layers)
    net = RDN(64, 3)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.name, '60_net.pth')))
    model.eval()
    # load data info
    # print('Loading data info ...\n')
    # files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    # files_source.sort()
    # process data
    psnr_test = 0
    cnt = 0
    index = np.arange(0, 100, 10)
    for f in index: #range(5):  #len(noise_list)
        # image
        ori = cv2.imread(os.path.join(root, noise_list[f]))
        gt = cv2.imread(os.path.join(root, gt_list[f]))
        Img = normalize(np.float32(ori[0:256,0:256,:]))
        #Img = np.expand_dims(Img, 0)

        [h, w, c] = Img.shape

        ISource = transforms.ToTensor()(Img)
        ISource = ISource.unsqueeze(0)

        Out = torch.Tensor(ISource.size()).cuda()
        # noisy image
        ISource = ISource.cuda()
        with torch.no_grad(): # this can save much memory
            k = 1
            n = int(w/k)
            for i in range(k):
                for j in range(k):
                    input1 = ISource[:, :, i*n:n+i*n, j*n:n+j*n]
                    #out1 = input1-model(ISource)
                    out1 = model(input1)
                    #out1 = torch.clamp(input1-model(input1), 0., 1.)
                    #out1 = torch.clamp(model(input1), 0., 1.)
                    Out[:, :, i*n:n+i*n, j*n:n+j*n] = out1
            #Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        Out = torch.clamp(Out, 0., 1.)*255
        img = Out.squeeze(0).cpu().numpy()
        img = img.astype('uint8')
        img = np.transpose(img, (1, 2, 0))

        show = np.concatenate((ori[0:256,0:256,:], img), 1)
        #cv2.imshow('ori', show)
        #cv2.imshow('result', img)
        #cv2.waitKey()

        cv2.imwrite('data/testing/' + str(cnt) + '_full' + '.png', show)
        cnt = cnt+1

        psnr = compare_psnr(img, gt[0:256,0:256,:], 255)
        print('psnr: %f', psnr)

        # psnr = batch_PSNR(Out, ISource, 1.)
        # psnr_test += psnr
        # print("%s PSNR %f" % (f, psnr))

if __name__ == "__main__":
    main()
