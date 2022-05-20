__author__ = 'ziao'
import os
import shutil
import argparse
import numpy as np
import torch
from data_provider import datasets_factory
from models.model_factory import Model
from utils import preprocess, logger
import trainer
import datetime
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定可见的gpu
parser = argparse.ArgumentParser(description='PyTorch video prediction model - TCTN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
train_path = ""
valid_path = ""

parser.add_argument('--dataset_name', type=str, default='mnist') # mnist, radar
parser.add_argument('--train_data_paths', type=str, default=train_path)
parser.add_argument('--valid_data_paths', type=str, default=valid_path)

# save_dir
parser.add_argument('--save_dir', type=str, default='moving-mnist/checkpoints/mnist_tctn')
parser.add_argument('--gen_frm_dir', type=str, default='moving-mnist/results/mnist_tctn')
parser.add_argument('--loss_dir', type=str, default='moving-mnist/loss/mnist_tctn')

# the print content save path, ect training loss
parser.add_argument('--print_path', type=str, default='moving-mnist/loss/mnist_tctn')

# input & output size
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--test_input_length', type=int, default=10)
parser.add_argument('--test_total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)  # 对应的图片宽度
parser.add_argument('--img_channel', type=int, default=1)

# optimization
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)  ## try 0.001, 0.01, 0.1
parser.add_argument('--n_steps', type=int, default=50000, help='number of iteration to update 2 times learning rate')
parser.add_argument('--T_0', type=int, default=5000)
parser.add_argument('--T_mult', type=int, default=2)
#parser.add_argument('--gamma', type=float, default=0.95, help='')
parser.add_argument('--batch_size', type=int, default=8)  # 8
parser.add_argument('--max_iterations', type=int, default=37500)  # 80000, 20000
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=1250)  # 5000
parser.add_argument('--snapshot_interval', type=int, default=2500)  # 5000, checkpoint save
parser.add_argument('--loss_interval', type=int, default=10000)
parser.add_argument('--num_save_samples', type=int, default=10)

#tctn
parser.add_argument('--model_name', type=str, default='TCTN')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--model_depth', type=int, default=128, help='depth of the model')
parser.add_argument('--de_layers', type=int, default=6, help='number of layers in decoder')
parser.add_argument('--n_layers', type=int, default=0, help='number of layers in encoder and decoder')
parser.add_argument('--n_heads', type=int, default=1, help='number of heads in conv mult-ihead attention')
parser.add_argument('--dec_frames', type=int, default=19, help='nummber of output frames')
parser.add_argument('--w_res', type=bool, default=True, help='using residual connect or not')
parser.add_argument('--w_pos', type=bool, default=True, help='using positional encoding or not')
parser.add_argument('--pos_kind', type=str, default='sine', help='kind of positional encoding,two choice: sine,learned')
parser.add_argument('--model_type', type=int, default=1, help='type of the model, 0 for interpoation model and 1 for extrapolation model')
parser.add_argument('--w_pffn', type=int, default=0)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--de_train_type', type=int, default=0, help='')
parser.add_argument('--test', type=int, default=0, help='')



args = parser.parse_args()
print(args)


def train_wrapper(model):
    # record all the print content to txt file
    logger.make_print_to_file(args.print_path)#logger?
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=True)

    ### save traning loss and test loss
    train_loss = []
    test_loss = []
    test_ssim = []
    test_psnr = []
    test_fmae = []
    test_sharp = []
    test_iter = []

    for itr in range(1, args.max_iterations + 1):
        # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)

        if train_input_handle.no_batch_left():#判断剩下数据够不够一个batch
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()#
        #ims = preprocess.reshape_patch_tensor(ims, args.patch_size)
        ims = preprocess.reshape_patch(ims, args.patch_size)

        tr_loss = trainer.train(model, ims, args, itr)
        train_loss.append(tr_loss)

        if itr % args.snapshot_interval == 0:#10000
            model.save(itr)

        if itr % args.test_interval == 0:#1000
            avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_input_handle, args, itr)
            test_iter.append(itr)
            test_loss.append(avg_mse)
            test_ssim.append(ssim)
            test_psnr.append(psnr)
            test_fmae.append(fmae)
            test_sharp.append(sharp)

            #x = range(len(train_loss))
            #y = range(len(test_loss)*args.test_interval)
            #plt.figure(1)
            #plt.title("these are losses of training and validation")
            #plt.plot(x, train_loss, label='loss of training',color='coral')
            #plt.plot(y, test_loss, label='loss of validation',color='black')
            #plt.legend()
            #plt.savefig(args.loss_dir + '/losses.png')
            #plt.close(1)

            x = range(len(train_loss))
            plt.figure(1)
            plt.title("this is losses of training")
            plt.plot(x, train_loss, label='loss of training')
            plt.legend()
            plt.savefig(args.loss_dir + '/training.png')
            plt.close(1)

           
            x = range(len(test_loss))
            plt.figure(1)
            plt.title("this is losses of validation")
            plt.plot(x, test_loss, label='loss')
            plt.legend()
            plt.savefig(args.loss_dir + '/valid_loss.png')
            plt.close(1)
            next

        if itr % args.loss_interval == 0:#10000
            fileName = "/loss iter{}".format(itr) + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
            np.savez_compressed(args.loss_dir + fileName, train_loss=np.array(train_loss),
                                test_iter=np.array(test_iter), test_loss=np.array(test_loss),
                                test_ssim=np.array(test_ssim), test_psnr=np.array(test_psnr),
                                test_fmae=np.array(test_fmae), test_sharp=np.array(test_sharp))

        train_input_handle.next()#？？

    fileName = "/loss all " + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
    np.savez_compressed(args.loss_dir + fileName, train_loss=np.array(train_loss), test_iter=np.array(test_iter),
                        test_loss=np.array(test_loss), test_ssim=np.array(test_ssim), test_psnr=np.array(test_psnr),
                        test_fmae=np.array(test_fmae), test_sharp=np.array(test_sharp))




def test_wrapper(model):
    # record all the print content to txt file
    logger.make_print_to_file(args.print_path)
    # test
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=False)
    avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_input_handle, args, 'test_result')
    fileName = "/test loss " + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
    np.savez_compressed(args.loss_dir + fileName, avg_mse=avg_mse, ssim=ssim, psnr=psnr, fmae=fmae, sharp=sharp)


if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

if not os.path.exists(args.loss_dir):
    os.makedirs(args.loss_dir)

# gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
print('Initializing models')

model = Model(args)
"""
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(args.device)
"""
if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
