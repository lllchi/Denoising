import os
import time
import argparse
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader

from models.RDN import *

from dataset import prepare_data, Dataset, random_augumentation
from utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="RCAN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=10, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--weight_decay", type=float, default=0.00001, help="Initial learning rate")

parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--root", type=str, default="/data0/lichi/denoising/sRGB/SIDD_Medium_Srgb/", help='path of image files')
parser.add_argument("--name", type=str, default="RDN_e40_16", help='model name')

parser.add_argument("--add_BN", type=bool, default=True, help='Batch Normalization')

opt = parser.parse_args()

def main():

    model_dir = os.path.join(opt.outf, opt.name)
    print('create checkpoint directory %s...' % model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    num = len(dataset_train)
    # Build model
    net = RDN(64, 3)

    num_params = 0
    for parm in net.parameters():
        num_params += parm.numel()
    print(net)
    print('[Network %s] Total number of parameters : %.3f M' % (opt.name, num_params / 1e6))
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #model.load_state_dict(torch.load(os.path.join('logs/', opt.name, '40_net.pth')))  # !!!

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)  #weight_decay=opt.weight_decay
    # training
    step = 0
    for epoch in range(opt.epochs):
        # set learning rate
        current_lr = update_lr(opt.lr, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        model.train()
        start_time = time.time()
        ave_loss = 0
        ave_psnr = 0
        ave_ssim = 0
        for i, data in enumerate(loader_train, 0):
            # training step
            time1 = time.time()
            model.zero_grad()
            optimizer.zero_grad()

            noise_img = data[:, :3, :, :]
            gt_img = data[:, 3:, :, :]

            noise_img, gt_img = noise_img.cuda(), gt_img.cuda()
            res = noise_img - gt_img
            pred_res = model(noise_img)

            loss1 = torch.mean(torch.abs(pred_res - gt_img))
            loss2 = torch.mean(SSIM(pred_res, gt_img))
            loss = loss1 #0.75*loss1 + 0.25*loss2

            loss.backward()
            optimizer.step()

            # evaluate
            #result = torch.clamp(noise_img-pred_res, 0., 1.)
            result = torch.clamp(pred_res, 0., 1.)
            psnr_train = batch_PSNR(result, gt_img, 1.)

            ave_loss = (ave_loss*i + loss.item()) / (i+1)
            ave_psnr = (ave_psnr*i + psnr_train) / (i+1)
            ave_ssim = (ave_ssim*i + 1-loss2.item()*2) / (i+1)

            time2 = time.time()

            if i % 100 == 0:

                print("[epoch %d][%d/%d] time: %.3f t_time: %.3f loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                    (epoch+1, i, len(loader_train), (time2 - time1), (time2 - start_time), ave_loss, ave_psnr, ave_ssim))

            if step % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, 'latest_net.pth'))
            step += 1
        print('Time for the epoch is %f' % (time.time() - start_time))
        ## the end of each epoch

        # save model
        save_name = '%d_net.pth' % (epoch+1)
        torch.save(model.state_dict(), os.path.join(model_dir, save_name))


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(root=opt.root, data_path='data', patch_size=256, stride=200, aug_times=1)
	else:
        main()
