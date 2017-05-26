# Stacked Hourglass Networks for Human Pose Estimation (Demo Code)

This repository includes Torch code for evaluation and visualization of the network presented in:

Alejandro Newell, Kaiyu Yang, and Jia Deng,
**Stacked Hourglass Networks for Human Pose Estimation**,
[arXiv:1603.06937](http://arxiv.org/abs/1603.06937), 2016.

A pretrained model is available on the [project site](http://www-personal.umich.edu/~alnewell/pose). Include the model in the main directory of this repository to run the demo code.

**Check out the training and experimentation code now available at: [https://github.com/anewell/pose-hg-train](https://github.com/anewell/pose-hg-train)**

In addition, if you download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de) and replace this repository's `images` directory you can generate full predictions on the validation and test sets.

To run this code, the following must be installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- qlua (for displaying results)

For displaying the demo images:
`qlua main.lua demo`

For generating predictions:
`th main.lua predict-[valid or test]`

For evaluation on a set of validation predictions:
`th main.lua eval` 

## Testing your own images

To use the network off-the-shelf, it is critical that the target person is centered in the input image. There is some robustness to scale, but for best performance the person should be sized such that their full height is roughly three-quarters of the input height. Play around with different scale settings to see the impact it has on the network output. We offer a convenient function for generating an input image:

`inputImg = crop(img, center, scale, rot, res)`

`res` should be set to 256 for our network. `rot` is offered if you wish to rotate the image (in degrees). You can run the input image through the network, and get the (x,y) coordinates with:

`outputHm = m:forward(inputImg:view(1,3,256,256):cuda())`

`predsHm,predsImg = getPreds(outputHm, center, scale)`

The two outputs of `getPreds` are coordinates with respect to either the heatmap or the original image (using center and scale to apply the appropriate transformation back to the image space).

The MPII images come with center and scale annotations already. An important detail with regards to the annotations: we have modified their format slightly for ease of use with our code. In addition, we adjusted the original center and scale annotations uniformly across all images so as to reduce the chances of our function cropping out feet from the bottom of the image. This mostly involved moving the center down a fraction.

