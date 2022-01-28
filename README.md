## Edits made to this repo by Katherine Crowson

I have added several features to this repository for use in creating higher quality generative art (feature visualization probably also benefits):

* [Deformable convolutions](https://arxiv.org/abs/1703.06211) have been added.

* Higher quality non-learnable upsampling filters (bicubic, Lanczos) have been added, with matching downsampling filters. A bilinear downsampling filter which low pass filters properly has also been added.

* The nets can now optionally output to a fixed decorrelated color space which is then transformed to RGB and sigmoided. Deep Image Prior as originally written does not know anything about the correlations between RGB color channels in natural images, which can be disadvantageous when using it for feature visualization and generative art.

Example:

```python
from models import get_hq_skip_net

net = get_hq_skip_net(input_depth).to(device)
```

`get_hq_skip_net()` provides higher quality defaults for the skip net, using the added features, than `get_net()`. Deformable convolutions can be slow and if this is a problem you can disable them with `offset_groups=0` or `offset_type='none'`. The decorrelated color space can be turned off with `decorr_rgb=False`. The `upsample_mode` and `downsample_mode` defaults are now `'cubic'` for visual quality, I would recommend not going below `'linear'`. The default channel count and number of scales has been increased.

The default configuration is to use 1x1 convolution layers to create the offsets for the
deformable convolutions, because training can become unstable with 3x3. However to make full use of deformable convolutions you may want to use 3x3 offset layers and set their learning rate to around 1/10 of the normal layers:

```python
net = get_hq_skip_net(input_depth, offset_type='full')
params = [{'params': get_non_offset_params(net), 'lr': lr},
          {'params': get_offset_params(net), 'lr': lr / 10}]
opt = optim.Adam(params)
```

# Original README

**Warning!** The optimization may not converge on some GPUs. We've personally experienced issues on Tesla V100 and P40 GPUs. When running the code, make sure you get similar results to the paper first. Easiest to check using text inpainting notebook.  Try to set double precision mode or turn off cudnn. 

# Deep image prior

In this repository we provide *Jupyter Notebooks* to reproduce each figure from the paper:

> **Deep Image Prior**

> CVPR 2018

> Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky


[[paper]](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf) [[supmat]](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM) [[project page]](https://dmitryulyanov.github.io/deep_image_prior)

![](data/teaser_compiled.jpg)

Here we provide hyperparameters and architectures, that were used to generate the figures. Most of them are far from optimal. Do not hesitate to change them and see the effect.

We will expand this README with a list of hyperparameters and options shortly.

# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- numpy
- scipy
- matplotlib
- scikit-image
- jupyter

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```


or create an conda env with all dependencies via environment file

```
conda env create -f environment.yml
```

## Docker image

Alternatively, you can use a Docker image that exposes a Jupyter Notebook with all required dependencies. To build this image ensure you have both [docker](https://www.docker.com/) and  [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed, then run

```
nvidia-docker build -t deep-image-prior .
```

After the build you can start the container as

```
nvidia-docker run --rm -it --ipc=host -p 8888:8888 deep-image-prior
```

you will be provided an URL through which you can connect to the Jupyter notebook.

## Google Colab

To run it using Google Colab, click [here](https://colab.research.google.com/github/DmitryUlyanov/deep-image-prior) and select the notebook to run. Remember to uncomment the first cell to clone the repository into colab's environment.


# Citation
```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}
```
