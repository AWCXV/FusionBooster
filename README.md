# FusionBooster
This is the offical implementation for the paper titled "FusionBooster: A Unified Image Fusion Boosting Paradigm".

[Paper](https://arxiv.org/abs/2305.05970)

## Highlight
- We devise an image fusion booster by analysing the quality of the initial fusion results by means of a dedicated Information Probe.
- The proposed FusionBooster is a general enhancer, which can be applied to various image fusion methods, e.g., traditional or learning-based algorithms, irrespective of the type of fusion task.
- In a new divide-and-conquer image fusion paradigm, the results of the analysis performed by the Information Probe guide the refinement of the fused image.
- The proposed FusionBooster significantly enhances the performance of the SOTA fusion methods and downstream detection tasks, with only a slight increase in the computational overhead.

The code is coming soon.

## Announcement
- 2024-9-30 This work has been accepted by IJCV.
- 2024-10-1 Because some of the fusion methods are realised using the tensorflow framework. Our FusionBooster demo will be implemented based on the [MUFusion](https://github.com/AWCXV/MUFusion). You can always use our "detached booster" to enhance your own fusion results. 
