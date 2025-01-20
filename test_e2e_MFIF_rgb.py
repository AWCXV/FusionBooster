# test phase
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.autograd import Variable
from net import ReconVISnet, ReconIRnet, ReconFuseNet
import utils
from utils import sumPatch
from PIL import Image
import cv2
from args_fusion import args
import numpy as np
import time
import torchvision.models as models
from utils import gradient
torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_tensor_type(torch.FloatTensor)
device_ids = [0]

def load_model_MUFusion(path, input_nc = 2, output_nc = 1):

    from mufusion_net import TwoFusion_net
    nest_model = TwoFusion_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format("MUFusion", para / 1000/100))

    nest_model.eval()
    #nest_model.cuda()

    return nest_model

def load_model_reconIR(path, input_nc, output_nc):

    ReconIRnet_model = ReconIRnet();
    ReconIRnet_model = torch.nn.DataParallel(ReconIRnet_model, device_ids = device_ids);    
    ReconIRnet_model.load_state_dict(torch.load(path), strict=False)

    ReconIRnet_model.eval()
        
    if (args.cuda):        
        ReconIRnet_model = ReconIRnet_model.cuda();
        

    return ReconIRnet_model

def load_model_reconVIS(path, input_nc, output_nc):
    ReconVISnet_model = ReconVISnet();
    ReconVISnet_model = torch.nn.DataParallel(ReconVISnet_model, device_ids = device_ids);
    ReconVISnet_model.load_state_dict(torch.load(path), strict=False)

    ReconVISnet_model.eval()

    if (args.cuda):
            
        ReconVISnet_model = ReconVISnet_model.cuda();

    return ReconVISnet_model
    
def load_model_reconFuse(path, input_nc, output_nc):  
    
    ReconFuseNet_model = ReconFuseNet();
    ReconFuseNet_model = torch.nn.DataParallel(ReconFuseNet_model, device_ids = device_ids);            
    ReconFuseNet_model.load_state_dict(torch.load(path), strict=False)

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
 
def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image

def fuse_cb_cr(Cb1,Cr1,Cb2,Cr2):
    H, W = Cb1.shape
    Cb = np.ones((H, W))
    Cr = np.ones((H, W))

    for k in range(H):
        for n in range(W):
            if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                Cb[k, n] = 128
            else:
                middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                Cb[k, n] = middle_1 / middle_2

            if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                Cr[k, n] = 128
            else:
                middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                Cr[k, n] = middle_3 / middle_4
    return Cb, Cr

def run_demo_MUFusion(model, ir_img, vi_img):

    
    ir_img = ir_img/255.0;
    vi_img = vi_img/255.0;
    h = vi_img.shape[0];
    w = vi_img.shape[1];
    
    ir_img = np.resize(ir_img,[1,1,h,w]);
    vi_img = np.resize(vi_img,[1,1,h,w]);
    
    ir_img = torch.from_numpy(ir_img);
    vi_img = torch.from_numpy(vi_img);
    
    # dim = img_ir.shape
    if args.cuda:
        ir_img = ir_img.cuda(args.device)
        vi_img = vi_img.cuda(args.device)
        model = model.cuda(args.device);

    img = torch.cat([ir_img,vi_img],1);
    out = model(img);
    ############################ multi outputs ##############################################
    fuseImage = out[0][0].cpu().numpy();
           
    return fuseImage



def run_demo(model_ReconFuse , model_MUFusion, model_ReconIR ,model_ReconVIS , infrared_path, visible_path, output_path_root, fileName, fusion_type, network_type, strategy_type, ssim_weight_str, mode):    

    ir_img = Image.open(infrared_path).convert("RGB")
    ir_img, ir_img_cb, ir_img_cr = rgb_to_ycbcr(ir_img)
    
    vi_img = Image.open(visible_path).convert("RGB")
    vi_img_y, vi_img_cb, vi_img_cr = rgb_to_ycbcr(vi_img)
    
    vi_img_cb, vi_img_cr = fuse_cb_cr(vi_img_cb, vi_img_cr, ir_img_cb, ir_img_cr);

    print("Generating Fusion Results...")
    fused_img_y = run_demo_MUFusion(model_MUFusion, ir_img, vi_img_y)
    
    ir_img=ir_img/255.0;
    vi_img_y=vi_img_y/255.0;
    fused_img_y = fused_img_y/255.0;
    h = vi_img_y.shape[0];
    w = vi_img_y.shape[1];
    
    ir_img_patches = [];
    vi_img_patches = [];
    fused_img_patches = [];
    
    ir_img = np.resize(ir_img, [1,h,w]);
    vi_img_y = np.resize(vi_img_y, [1,h,w]);
    fused_img_y = np.resize(fused_img_y, [1,h,w]);
    
    ir_img_patches.append(ir_img);  
    vi_img_patches.append(vi_img_y);
    fused_img_patches.append(fused_img_y);
    
    ps = args.PATCH_SIZE;
    
    ir_img_patches = np.stack(ir_img_patches,axis=0);
    vi_img_patches = np.stack(vi_img_patches,axis=0);
    fused_img_patches = np.stack(fused_img_patches,axis=0);
    
    ir_img_patches = torch.from_numpy(ir_img_patches);
    vi_img_patches = torch.from_numpy(vi_img_patches);
    fused_img_patches = torch.from_numpy(fused_img_patches);
    
    #print(fused_img_patches);
    
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        fused_img_patches = fused_img_patches.cuda(args.device);
        

    recIR = model_ReconIR(fusion = fused_img_patches);
    recVIS = model_ReconVIS(fusion = fused_img_patches);
    
    #print(recIR);        

    #Booster Layer -- begin
    recIRb  = sumPatch(recIR,3);
    recVISb = sumPatch(recVIS,3);
    
    recIRe  = recIR + ir_img_patches - recIRb;
    recVISe = recVIS + vi_img_patches -  recVISb;    
    
    #Booster Layer -- end
    
    print("Enhancing Fusion Results...")
    out_y = model_ReconFuse(recIR = recIRe, recVIS = recVISe);
    out_y = out_y[0,0,:,:].cpu().numpy()
    out_y = out_y*255
    
    #vi_img_cb & vi_img_cr are equal to the fused_img_cb and fuesd_img_cr
    fuseImage = ycbcr_to_rgb(out_y, vi_img_cb, vi_img_cr);    

    outputFuse = output_path_root + fileName;
    fuseImage.save(outputFuse);             
    
    print(outputFuse);

def main():
    print("")
    print("************************************************")
    if (args.cuda):
        print("Trying to use GPU for FusionBooster inference...")
    else:
        print("Trying to use CPU for FusionBooster inference...")
    print("************************************************")
    print("")
    
    test_path = "./dataset/MFI-WHU/"

    network_type = 'densefuse'
    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['AVG', 'L1','SC']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    strategy_type = strategy_type_list[1]
    output_path = './outputs_enhancedMUFusion_mfif_rgb/';
    
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = 2
    out_c = 1
    mode = 'L'
    model_path_ReconFuse = "./models/MFIF/ASE.model"
    model_path_ReconInfrared = "./models/MFIF/InformationProbe_ir.model"
    model_path_ReconVisible = "./models/MFIF/InformationProbe_vis.model"
    model_path_MUFusion = "./models/MFIF/MUFusion.model"

    with torch.no_grad():
        ssim_weight_str = args.ssim_path[2]
        model_ReconFuse = load_model_reconFuse(model_path_ReconFuse, in_c, out_c)
        model_ReconIR = load_model_reconIR(model_path_ReconInfrared, in_c, out_c)
        model_ReconVIS = load_model_reconVIS(model_path_ReconVisible, in_c, out_c)
        model_MUFusion = load_model_MUFusion(model_path_MUFusion)
        files = os.listdir(test_path + "ir/");
        numFiles = len(files);
        for i in range(numFiles):
            infrared_path = test_path + 'ir/' + files[i];
            visible_path = test_path + 'vis/' + files[i];
            run_demo(model_ReconFuse , model_MUFusion, model_ReconIR ,model_ReconVIS , infrared_path, visible_path, output_path, files[i], fusion_type, network_type, strategy_type, ssim_weight_str, mode)
    print('Done......')

if __name__ == '__main__':
    main()
