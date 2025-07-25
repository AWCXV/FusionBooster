

## FusionBooster (2025' IJCV)
This is the offical implementation for the paper titled "FusionBooster: A Unified Image Fusion Boosting Paradigm".

[Paper]([https://arxiv.org/abs/2305.05970](https://link.springer.com/article/10.1007/s11263-024-02266-6))


<div align="center">
  <img src="Figs/Result_1.png" width="1000px" />
  <p>"Visualisation of the proposed FusionBooster when applied to the infrared and visilble image fusion task."</p>
</div>

## <img width="40" src="Figs/environment.png"> Environment
```
python 3.7.3
torch 1.9.0
scipy 1.2.0
```
## <img width="32" src="Figs/test.png"> Test 

### <img width="25" src="Figs/set.png"> End to end

Use our pre-trained model to directly output enhanced fusion results based on two input images.

<img width="20" src="Figs/task.png"> IVIF task (Backbone: MUFusion):
```
python test_e2e_IVIF_rgb.py
```

<img width="20" src="Figs/task.png"> MFIF task (Backbone: MUFusion):

```
python test_e2e_MFIF_rgb.py
```

<img width="20" src="Figs/task.png"> MEIF task (Backbone: MUFusion):

```
python test_e2e_MEIF_rgb.py
```

## <img width="32" src="Figs/train.png"> Train

### <img width="20" src="Figs/task.png"> IVIF task (Backbone: DDcGAN)

<img width="15" src="Figs/dataset.png"> Dataset:

[Training Set (DDcGAN Results on LLVIP)](https://pan.baidu.com/s/1X58UeWpLSBiFMlRi6pFOLw?pwd=hokf) Password: hokf

[Training Set (Original LLVIP)](https://pan.baidu.com/s/1_I707esOlERfyMiUOzuZQg?pwd=jq15) Password: jq15

Put the above train data in the "train_data" folder and run the following prompt:

```
python train.py --path_to_ir_vis './train_data/LLVIP/' --path_to_init_fus './train_data/outputsLLVIPTrain/'
```

The trained model will be saved in the "models" folder automatically.


### <img width="25" src="Figs/set.png"> Test - Booster Only

To use our pre-trained FusionBooster to boost an arbitary method:

<img width="20" src="Figs/task.png"> IVIF task (Backbone: DDcGAN)

```
python test_booster_only_IVIF_rgb.py
```

You can modify the path in the "test_booster_only_xxxx.py" file, to enhance your own fusion results. 

## <img width="32" src="Figs/announcement.png"> Announcement
- 2025-7-9 The code for end-to-end boosting source images (MEIF) is now available. ("test_e2e_MEIF_rgb.py")
- 2025-5-9 The training code for IVIT task is now available. ("train.py")
- 2025-1-18 The code for end-to-end boosting source images (MFIF) is now available. ("test_e2e_MFIF_rgb.py")
- 2024-10-14 The code for end-to-end boosting source images (IVIF) is now available. ("test_e2e_IVIF_rgb.py")
- 2024-10-14 The code for boosting an arbitary method is available. ("test_booster_only.py")
- 2024-10-1 Because some of the fusion methods are realised using the tensorflow framework. Our FusionBooster demo will be implemented based on the [MUFusion](https://github.com/AWCXV/MUFusion). You can always use our "detached booster" to enhance your own fusion results. 
- 2024-9-30 This work has been accepted by IJCV.

## <img width="32" src="Figs/highlight.png"> Highlight
- We devise an image fusion booster by analysing the quality of the initial fusion results by means of a dedicated Information Probe.
- The proposed FusionBooster is a general enhancer, which can be applied to various image fusion methods, e.g., traditional or learning-based algorithms, irrespective of the type of fusion task.
- In a new divide-and-conquer image fusion paradigm, the results of the analysis performed by the Information Probe guide the refinement of the fused image.
- The proposed FusionBooster significantly enhances the performance of the SOTA fusion methods and downstream detection tasks, with only a slight increase in the computational overhead.

### <img width="32" src="Figs/citation.png"> Citation
If this work is helpful to you, please cite it as:
```
@article{cheng2025fusionbooster,
  title={FusionBooster: A Unified Image Fusion Boosting Paradigm},
  author={Cheng, Chunyang and Xu, Tianyang and Wu, Xiao-Jun and Li, Hui and Li, Xi and Kittler, Josef},
  journal={International Journal of Computer Vision},
  volume={133},
  pages={3041--3058},
  year={2025}
}
```

