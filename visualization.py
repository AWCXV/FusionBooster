# test phase
import torch
import os
from torch.autograd import Variable
from net import TwoFusion_net
import utils
from scipy.misc import imread, imsave, imresize
from args_fusion import args
import numpy as np
import time
torch.set_default_tensor_type(torch.DoubleTensor)

def load_model(path, input_nc, output_nc):

    nest_model = TwoFusion_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para*type_size / 1000/1000))

    nest_model.eval()
    #nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
    # encoder
    en_v = model.encoder(img2)
    en_r = model.encoder(img1)
    f = model.fusion(en_r, en_v, strategy_type=strategy_type)
    img_fusion = model.decoder(f);
    return img_fusion[0]

def run_demo(model, infrared_path, visible_path, output_path_root, index, network_type, mode):
    ir_img_patches = [];
    vi_img_patches = [];
    
    ir_img = imread(infrared_path,mode='L');
    vi_img = imread(visible_path,mode='L');
    ir_img=ir_img/255.0;
    vi_img=vi_img/255.0;
    
    h = vi_img.shape[0];
    w = vi_img.shape[1];
    
    ir_img = np.resize(ir_img,[1,h,w]);    
    vi_img = np.resize(vi_img,[1,h,w]);
    ir_img_patches.append(ir_img);
    vi_img_patches.append(vi_img);

    ir_img_patches = np.stack(ir_img_patches,axis=0);
    vi_img_patches = np.stack(vi_img_patches,axis=0);

    ir_img_patches = torch.from_numpy(ir_img_patches);
    vi_img_patches = torch.from_numpy(vi_img_patches);
    
    # dim = img_ir.shape
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        model = model.cuda(args.device);
    #ir_img_patches = Variable(ir_img_patches, requires_grad=False)
    #vi_img_patches = Variable(vi_img_patches, requires_grad=False)

    img = torch.cat([ir_img_patches,vi_img_patches],1);
    en = model.encoder(img);
    for h in range(6):
        featuremaps = en[h];
        for i in range(32):
            path = "visual/depth_"+str(h)+"_"+str(i)+"-th_channel.png";
            imsave("visual/depth_"+str(h)+"_"+str(i)+"-th_channel.png",featuremaps[0,i,:,:].cpu().numpy());
            print(path);
    out = model.decoder(en)[0];

def main():

    test_path = "images/IV_images/"

    network_type = 'densefuse'

    output_path = './outputs/';

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = 2
    out_c = 1
    mode = 'L'
    model_path = "./stage1.model"

    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)
        for i in range(1):
            index = i + 1
            infrared_path = "./IR" + str(index) + '.png'
            visible_path = "./VIS" + str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, network_type, mode)
    print('Done......')

if __name__ == '__main__':
    main()
