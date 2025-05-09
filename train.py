#python train.py --path_to_ir_vis '../../dataset/LLVIP' --path_to_init_fus '/data/disk_A/chunyang/code_others_infrared_medical/DDc_GAN/outputsLLVIPTrain/'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
from utils import gradient, gradient2
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from testMat import showLossChart
from torch.autograd import Variable
import cv2
import utils
from net import ReconVISnet, ReconIRnet, ReconFuseNet
from args_fusion import args
from utils import sumPatch
import pytorch_msssim
import torchvision.models as models
import Myloss
import argparse

parser = argparse.ArgumentParser()

#default = './train_data/LLVIP/'
parser.add_argument('--path_to_ir_vis', type=str, required=True, help='root path to clean infrared and visible images (root/infrared/train, root/visible/train)')

#default = './train_data/outputsLLVIPTrain/'
parser.add_argument('--path_to_init_fus', type=str, required=True, help='root path to fusion results of an arbitrary method (root_init_fus/"ir or vis name.xxx")')

#parser.add_argument('--save_path', type=str, default='./outputs/', help='the fusion results will be saved here')

opt = parser.parse_args()



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #train_num = 40000
    train_num = 70000
    #train_num = 50;
    # for i in range(5):
    train()


def train():

    #创建模型和损失函数的保存地址。
    temp_path_model = os.path.join(args.save_model_dir)
    if os.path.exists(temp_path_model) is False:
        os.mkdir(temp_path_model)
        
    temp_path_loss = os.path.join(args.save_loss_dir)
    if os.path.exists(temp_path_loss) is False:
        os.mkdir(temp_path_loss)    
   
    #获取原始图像(红外&可见光) 为信息探针提供监督信号
    patchPrePathOriginal = opt.path_to_ir_vis
    
    #初始融合结果目录(e.g., DDCGAN)，目标是训练FBooster重构这个初始结果。
    patchPrePathLabel = opt.path_to_init_fus    
    
    PatchPaths = os.listdir(patchPrePathOriginal+"/visible/train/");

    batch_size = args.batch_size

    ReconVISnet_model = ReconVISnet();
    ReconIRnet_model = ReconIRnet();
    ReconFuseNet_model = ReconFuseNet();
    
    optimizerVIS = Adam(ReconVISnet_model.parameters(), args.lr)
    optimizerIR = Adam(ReconIRnet_model.parameters(), args.lr)
    optimizerFUSE = Adam(ReconFuseNet_model.parameters(), args.lr)
    
    mse_loss = torch.nn.MSELoss(reduction = "mean")
    ssim_loss = pytorch_msssim.msssim
    l1_loss = torch.nn.L1Loss(reduction = "mean");
    L_spa = Myloss.L_spa();
    loss_exp_pre = Myloss.L_exp(16,0.6);    

    device_ids = [0]

    if (args.cuda):
        ReconVISnet_model = torch.nn.DataParallel(ReconVISnet_model, device_ids = device_ids);
        ReconIRnet_model = torch.nn.DataParallel(ReconIRnet_model, device_ids = device_ids);
        ReconFuseNet_model = torch.nn.DataParallel(ReconFuseNet_model, device_ids = device_ids);
        
        ReconVISnet_model = ReconVISnet_model.cuda();
        ReconIRnet_model = ReconIRnet_model.cuda();
        ReconFuseNet_model = ReconFuseNet_model.cuda();

    tbar = trange(args.epochs)
    print('Start training.....')

    Loss_ill = []
    Loss_str = []
    Loss_final = []
    Loss_total = []
    all_ill_loss = 0.;
    all_str_loss = 0.;
    all_final_loss = 0.;
    all_total_loss =0.;
    pow2 = 0;
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        patchesPaths, batches = utils.load_datasetPair(PatchPaths,batch_size);
        ReconVISnet_model.train()
        ReconIRnet_model.train()
        ReconFuseNet_model.train()
        count = 0
        for batch in range(batches):
            image_paths = patchesPaths[batch * batch_size:(batch * batch_size + batch_size)]
            
            image_ir = utils.get_train_images_auto(patchPrePathOriginal+"/infrared/train",image_paths, mode="L");
            image_vi = utils.get_train_images_auto(patchPrePathOriginal+"/visible/train",image_paths, mode="L");
            img_fu = utils.get_train_images_auto(patchPrePathLabel,image_paths, mode="L",isPng = False);
            h = image_ir.shape[2];
            w = image_ir.shape[3];

            count += 1
            optimizerIR.zero_grad()
            optimizerVIS.zero_grad()
            optimizerFUSE.zero_grad()
            
            img_ir = Variable(image_ir, requires_grad=False)
            img_vi = Variable(image_vi, requires_grad=False)
            img_fu = Variable(img_fu, requires_grad=False)
            
            if args.cuda:
                img_ir = img_ir.cuda(args.device)
                img_vi = img_vi.cuda(args.device)        
                img_fu = img_fu.cuda(args.device)

            #损失函数  ------------开始
            
            #先更新红外/可见光图像的子网络。然后再更新重构融合图像的网络。
            recIR = ReconIRnet_model(fusion = img_fu);
            lossRecIR = l1_loss(recIR, img_ir);
            lossRecIR.backward();
            optimizerIR.step();

            
            recVIS = ReconVISnet_model(fusion = img_fu);
            lossRecVIS = l1_loss(recVIS, img_vi);            
            lossRecVIS.backward();
            optimizerVIS.step();            
            
            
            recIRe = ReconIRnet_model(fusion = img_fu);
            recVISe = ReconVISnet_model(fusion = img_fu);            
            out = ReconFuseNet_model(recIR = recIRe.detach(), recVIS = recVISe.detach());
            

            lossRecFuse = l1_loss(out, img_fu);
            lossRecFuse.backward();
            optimizerFUSE.step();                        
            
            loss_final_ill = 0;
            loss_final_str = 0;

            
            #损失函数  --------------结束
            

            all_ill_loss += lossRecIR.item();
            all_str_loss += lossRecVIS.item();
            all_final_loss += lossRecFuse.item();
            
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t ill loss: {:.6f}\t str loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_ill_loss / args.log_interval,
                                  all_str_loss / args.log_interval,
                                  all_total_loss / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_ill.append(all_ill_loss / args.log_interval)
                Loss_str.append(all_str_loss / args.log_interval)
                Loss_final.append(all_final_loss / args.log_interval)
                Loss_total.append(all_total_loss / args.log_interval)

                all_ill_loss = 0.;
                all_str_loss = 0.;
                all_final_loss = 0.;
                all_total_loss = 0.;
                
            if (batch + 1) % 30 == 0:
                # save model
                ReconIRnet_model.eval()
                ReconIRnet_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_ReconIR" + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(ReconIRnet_model.state_dict(), save_model_path)

                ReconVISnet_model.eval()
                ReconVISnet_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_ReconVIS" + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(ReconVISnet_model.state_dict(), save_model_path)                
                
                ReconFuseNet_model.eval()
                ReconFuseNet_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_ReconFuse" + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(ReconFuseNet_model.state_dict(), save_model_path)                
                
                # save loss data
                # ill loss
                loss_data_pixel = np.array(Loss_ill)
                loss_filename_path = "loss_ill_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+'/loss_ill.png')
                
                # str loss
                loss_data_pixel = np.array(Loss_str)
                loss_filename_path = "loss_str_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_"  + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+'/loss_str.png')                
                
                # final loss
                loss_data_pixel = np.array(Loss_final)
                loss_filename_path = "loss_final_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+'/loss_final.png')                                
                
                # total loss
                loss_data_pixel = np.array(Loss_total)
                loss_filename_path = "loss_total_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+"/loss_total.png");
                
                ReconIRnet_model.train()
                ReconVISnet_model.train()
                ReconFuseNet_model.train()
                if (args.cuda):
                    ReconIRnet_model.cuda();
                    ReconVISnet_model.cuda();
                    ReconFuseNet_model.cuda();
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
    # ill loss
    loss_data_pixel = np.array(Loss_ill)
    loss_filename_path = "Final_loss_ill.mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
    showLossChart(save_loss_path,args.save_loss_dir+'/Final_loss_ill.png')
    
    # str loss
    loss_data_pixel = np.array(Loss_str)
    loss_filename_path = "Final_loss_str.mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
    showLossChart(save_loss_path,args.save_loss_dir+'/Final_loss_str.png')                
    
    # final loss
    loss_data_pixel = np.array(Loss_final)
    loss_filename_path = "Final_loss_final.mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
    showLossChart(save_loss_path,args.save_loss_dir+'/Final_loss_final.png')                                
    
    # grad loss
    loss_data_pixel = np.array(Loss_total)
    loss_filename_path = "Final_loss_total.mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
    showLossChart(save_loss_path,args.save_loss_dir+"/Final_loss_total.png");
    # save model
    
    ReconIRnet_model.eval()
    ReconIRnet_model.cpu()
    save_model_filename =  "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_')  + "_IR.model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(ReconIRnet_model.state_dict(), save_model_path)    
    
    ReconVISnet_model.eval()
    ReconVISnet_model.cpu()
    save_model_filename =  "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_')  + "_VIS.model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(ReconVISnet_model.state_dict(), save_model_path)

    ReconFuseNet_model.eval()
    ReconFuseNet_model.cpu()
    save_model_filename =  "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_')  + "_Fuse.model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(ReconFuseNet_model.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    main()
