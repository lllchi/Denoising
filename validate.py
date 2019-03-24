import cv2
import os
import argparse
import glob
from models.RDN import *

from utils import *

import time

import scipy.io as scio
import torchvision.transforms as transforms


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--which_model", type=str, default='final_net.pth', help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/", help='path of log files')
parser.add_argument("--name", type=str, default="RDN_e40_16", help='model name')
parser.add_argument("--validate_path", type=str, default="data/ValidationNoisyBlocksSrgb.mat", help='path of val files')

parser.add_argument("--add_BN", type=bool, default=True, help='Batch Normalization')


opt = parser.parse_args()

validate_path = opt.validate_path

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = RDN(64, 3)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.name, opt.which_model)))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    # files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    # files_source.sort()
    mat_file = scio.loadmat(validate_path)
    data = mat_file['ValidationNoisyBlocksSrgb']
    # process data
    ave_time = 0
    cnt = 1
    for p in range(40):
        for q in range(32):
            # image
            img = data[p, q, :, :, :]
            disp_img = img

            img = normalize(np.float32(img))

            input = transforms.ToTensor()(img)
            input = input.unsqueeze(0)
            input = input.cuda()

            out = torch.Tensor(input.size()).cuda()
            with torch.no_grad():  # this can save much memory
                k = 1
                n = int(256 / k)
                for i in range(k):
                    for j in range(k):
                        input1 = input[:, :, i * n:n + i * n, j * n:n + j * n]
                        torch.cuda.synchronize()
                        start = time.time()
                        out1 = model(input1)
                        torch.cuda.synchronize()
                        end = time.time()
                        ave_time = ave_time + end - start
                        out[:, :, i * n:n + i * n, j * n:n + j * n] = out1

                out = torch.clamp(out, 0., 1.) * 255
                out_img = out.squeeze(0).cpu().numpy()
                out_img = out_img.astype('uint8')
                out_img = np.transpose(out_img, (1, 2, 0))

                cnt = cnt + 1
                data[p, q, :, :, :] = out_img

            print('cnt : %d', p*32+q)

    mat_file['ValidationNoisyBlocksSrgb'] = data
    scio.savemat('data/Resultsl116_full', {'results': data})

    ave_time = ave_time / (1280)
    ave_time = ave_time * (1000/256) * (1000/256)
    print('average time : %4f', ave_time)

if __name__ == "__main__":
    main()
