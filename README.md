# 6-DoF Object Pose from Semantic Keypoints
## Georgios Pavlakos, Xiaowei Zhou, Aaron Chan, Konstantinos G. Derpanis, Kostas Daniilidis

This is the code for the paper **6-DoF Object Pose from Semantic Keypoints**. Please follow the links to read the [paper](https://arxiv.org/abs/1703.04670) and visit the corresponding [project page](https://www.seas.upenn.edu/~pavlakos/projects/object3d).

We provide code to test our approach on [PASCAL3D](http://cvgl.stanford.edu/projects/pascal3d.html). Please follow the instructions below to setup and use our code. To run this code, make sure the following are installed:

- MATLAB
- [Torch7](https://github.com/torch/torch7)
- hdf5
- cudnn

### 1) Downloading model and PASCAL3D data

To get the pretrained model and [PASCAL3D](http://cvgl.stanford.edu/projects/pascal3d.html) data in the expected folders, we have included all the preprocessing steps in the following script:

```bash init.sh```

### 2) Training PASCAL3D models (optional - a [pretrained model](http://visiondata.cis.upenn.edu/object3d/models/pose-hg-pascal3d.t7) was downloaded in the previous script)

We have included the [Stacked Hourglass training code](https://github.com/anewell/pose-hg-train) with small adaptations in our package, so that you can run training on PASCAL3D dataset using the provided keypoint annotations. From terminal you need to run the following code so you can start training:

```
cd pose-hg/pose-hg-train/src
th main.lua -dataset pascal3d -expID test-run-stacked -netType hg-stacked -task pose-int -nStack 2 -LR 2.5e-4 -nEpochs 100 -snapshot 1
```

For more details on the command line options you can take a look at the pose-hg/pose-hg-train/src/opt.lua file or the original [Stacked Hourglass code](https://github.com/anewell/pose-hg-train). Here, we have applied only minimal changes to make training on PASCAL3D feasible.

### 3) Heatmap Predictions

First we need to run the ConvNet on the validation images to get heatmap predictions.

```
cd pose-hg/pose-hg-demo
th main.lua pascal3d valid pretrained
```

### 4) Pose Optimization

On Matlab, you can run the pose optimization on the whole PASCAL3D dataset by using the file:

```
pascal3d_eval.m
```

The results for the clean set of the dataset (no occlusions or truncations) are printed in the specified results folder. Optionally, you can also visualize the results (heatmaps and viewpoint predictions) by setting the flag ```vis = 1```.

### 5) Additional Demos

We provide sample precomputed heatmaps to check the pose optimization results for the weak perspective and the full perspective case respectively:

```
demoWP.m
demoFP.m
```

Additionally, if you want to use our code on your own custom images, the following demo describes the steps you need to follow (both for Keypoint Localization and Pose Optimization):

```
demoCustom.m
```

### Citing

If you find this code useful for your research, please consider citing the following paper:

	@Inproceedings{pavlakos17object3d,
	  Title          = {6-DoF Object Pose from Semantic Keypoints},
	  Author         = {Pavlakos, Georgios and Zhou, Xiaowei and Chan, Aaron and Derpanis, Konstantinos G and Daniilidis, Kostas},
	  Booktitle      = {International Conference on Robotics and Automation (ICRA)},
	  Year           = {2017}
	}

### Acknowledgements

This code follows closely the [released code](https://github.com/anewell/pose-hg-demo) for the Stacked Hourglass networks by Alejandro Newell. If you use this code, please consider citing the [respective paper](http://arxiv.org/abs/1603.06937).
