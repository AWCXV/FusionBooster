<div align="center">
  <img src="Figs/Result_1.png" width="1000px" />
  <p>"Visualisation of the proposed FusionBooster when applied to the infrared and visilble image fusion approach task."</p>
</div>

## 1 FusionBooster
This is the offical implementation for the paper titled "FusionBooster: A Unified Image Fusion Boosting Paradigm".

[Paper](https://arxiv.org/abs/2305.05970)


## 2 Environment
```
python 3.7.3
torch 1.9.0
scipy 1.2.0
```
## 3 Test (End to end, you provide two source images, we directly output the enhanced fused image)


Backbone: MUFusion (If you want to report this result, please mark our mehtod as FusionBooster-MU)

### 3.1 For the IVIF task:
```
python test_e2e_rgb.py
```

### 3.2 For the MEIF task (to do):

```
python xxx.py
```

### 3.3 For the MFIF task (to do):

```
python xxx.py
```

## 4 Test (Booster Only, you provide the initial fused image, we enhance it)



To use our pre-trained FusionBooster to boost an arbitary method:

### 4.1 For the IVIF task (Backbone: DDcGAN)

```
python test_booster_only_rgb.py
```

### 4.2 For the MEIF task (to do)

```
python xxx.py
```

### 4.3 For the MFIF task (to do)

```
python xxx.py
```

You can modify the path in the "test_booster_only_xxxx.py" file, to enhance your own fusion results. 


## 5 Announcement
- 2024-10-14 The code for end-to-end boosting source images (IVIF) is now available. ("test_e2e_rgb.py").
- 2024-10-14 The code for boosting an arbitary method is available ("test_booster_only.py").
- 2024-10-1 Because some of the fusion methods are realised using the tensorflow framework. Our FusionBooster demo will be implemented based on the [MUFusion](https://github.com/AWCXV/MUFusion). You can always use our "detached booster" to enhance your own fusion results. 
- 2024-9-30 This work has been accepted by IJCV.

## 6 Highlight
- We devise an image fusion booster by analysing the quality of the initial fusion results by means of a dedicated Information Probe.
- The proposed FusionBooster is a general enhancer, which can be applied to various image fusion methods, e.g., traditional or learning-based algorithms, irrespective of the type of fusion task.
- In a new divide-and-conquer image fusion paradigm, the results of the analysis performed by the Information Probe guide the refinement of the fused image.
- The proposed FusionBooster significantly enhances the performance of the SOTA fusion methods and downstream detection tasks, with only a slight increase in the computational overhead.

### 7 Citation
If this work is helpful to you, please cite it as:
```
@article{cheng2024fusionbooster,
  title={FusionBooster: A Unified Image Fusion Boosting Paradigm},
  author={Cheng, Chunyang and Xu, Tianyang and Wu, Xiao-Jun and Li, Hui and Li, Xi and Kittler, Josef},
  journal={International Journal of Computer Vision},
  year={2024}
}
```

