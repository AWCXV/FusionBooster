# test phase
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.autograd import Variable
from net import ReconVISnet, ReconIRnet, ReconFuseNet
import utils
from utils import sumPatch
import cv2
from PIL import Image
from args_fusion import args
import numpy as np
import time
import torchvision.models as models
from utils import gradient
torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.FloatTensor)
device_ids = [0]

def load_model_reconIR(path, input_nc, output_nc):

    ReconIRnet_model = ReconIRnet();
    ReconIRnet_model = torch.nn.DataParallel(ReconIRnet_model, device_ids = device_ids);    
    ReconIRnet_model.load_state_dict(torch.load(path))

    ReconIRnet_model.eval()
        
    if (args.cuda):
        
        ReconIRnet_model = ReconIRnet_model.cuda();

    return ReconIRnet_model

def load_model_reconVIS(path, input_nc, output_nc):
    ReconVISnet_model = ReconVISnet();
    ReconVISnet_model = torch.nn.DataParallel(ReconVISnet_model, device_ids = device_ids);
    ReconVISnet_model.load_state_dict(torch.load(path))

    ReconVISnet_model.eval()

    if (args.cuda):
            
        ReconVISnet_model = ReconVISnet_model.cuda();

    return ReconVISnet_model
    
def load_model_reconFuse(path, input_nc, output_nc):  
    
    ReconFuseNet_model = ReconFuseNet();
    ReconFuseNet_model = torch.nn.DataParallel(ReconFuseNet_model, device_ids = device_ids);            
    ReconFuseNet_model.load_state_dict(torch.load(path))

    ReconFuseNet_model.eval()
    if (args.cuda):

        ReconFuseNet_model = ReconFuseNet_model.cuda();

    return ReconFuseNet_model


def _generate_fusion_image(model, strategy_type, img1, img2):
    # encoder
    en_v = model.encoder(img2)
    en_r = model.encoder(img1)
    f = model.fusion(en_r, en_v, strategy_type=strategy_type)
    img_fusion = model.decoder(f);
    return img_fusion[0]

def run_demo(model_ReconFuse ,model_ReconIR ,model_ReconVIS , infrared_path, visible_path, output_path_root, fileName, input_methodX_dir, fusion_type, network_type, strategy_type, ssim_weight_str, mode):

    ir_img = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)
    vi_img = cv2.imread(visible_path, cv2.IMREAD_GRAYSCALE)
    fused_img = cv2.imread(input_methodX_dir+fileName, cv2.IMREAD_GRAYSCALE)
    
    ir_img=ir_img/255.0;
    vi_img=vi_img/255.0;
    fused_img = fused_img/255.0;
    h = vi_img.shape[0];
    w = vi_img.shape[1];
    
    ir_img_patches = [];
    vi_img_patches = [];
    fused_img_patches = [];
    
    ir_img = np.resize(ir_img, [1,h,w]);
    vi_img = np.resize(vi_img, [1,h,w]);
    fused_img = np.resize(fused_img, [1,h,w]);
    
    ir_img_patches.append(ir_img);  
    vi_img_patches.append(vi_img);
    fused_img_patches.append(fused_img);
    
    ps = args.PATCH_SIZE;
    
    ir_img_patches = np.stack(ir_img_patches,axis=0);
    vi_img_patches = np.stack(vi_img_patches,axis=0);
    fused_img_patches = np.stack(fused_img_patches,axis=0);
    
    ir_img_patches = torch.from_numpy(ir_img_patches);
    vi_img_patches = torch.from_numpy(vi_img_patches);
    fused_img_patches = torch.from_numpy(fused_img_patches);
    
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        fused_img_patches = fused_img_patches.cuda(args.device);
    
    recIR = model_ReconIR(fusion = fused_img_patches);
    recVIS = model_ReconVIS(fusion = fused_img_patches);

    #Booster Layer -- begin
    recIRb  = sumPatch(recIR,3);
    recVISb = sumPatch(recVIS,3);
    
    recIRe  = recIR + ir_img_patches - recIRb;
    recVISe = recVIS + vi_img_patches -  recVISb;    
    
    #Booster Layer -- end
    
    out = model_ReconFuse(recIR = recIRe, recVIS = recVISe);
    out = out * 255
    
    outputPath = "outputs/";
    
    outputFuse = output_path_root + fileName;
        
        
    cv2.imwrite(outputFuse, out[0,0,:,:].cpu().numpy())
    
    print(outputFuse);

def main():

    test_path = "./dataset/LLVIP/"

    network_type = 'densefuse'
    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['AVG', 'L1','SC']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    strategy_type = strategy_type_list[1]
    output_path = './outputs_enhancedDDcGAN_gray/';
    
    #based on this algorithm and enhance its result
    input_methodX_dir = './Origin_DDcGAN_gray/';

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = 2
    out_c = 1
    mode = 'L'
    model_path_ReconFuse = "./models/IVIF/ASE.model"
    model_path_ReconInfrared = "./models/IVIF/InformationProbe_ir.model"
    model_path_ReconVisible = "./models/IVIF/InformationProbe_vis.model"

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[2])
        ssim_weight_str = args.ssim_path[2]
        model_ReconFuse = load_model_reconFuse(model_path_ReconFuse, in_c, out_c)
        model_ReconIR = load_model_reconIR(model_path_ReconInfrared, in_c, out_c)
        model_ReconVIS = load_model_reconVIS(model_path_ReconVisible, in_c, out_c)
        files = os.listdir(test_path + "ir/");
        numFiles = len(files);
        for i in range(numFiles):
            infrared_path = test_path + 'ir/' + files[i];
            visible_path = test_path + 'vis/' + files[i];
            run_demo(model_ReconFuse ,model_ReconIR ,model_ReconVIS , infrared_path, visible_path, output_path, files[i], input_methodX_dir, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
    print('Done......')

if __name__ == '__main__':
    main()
