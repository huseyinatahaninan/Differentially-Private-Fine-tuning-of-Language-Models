# Fine-tuning for Language Understanding Tasks

In this folder, we implement four methods to fine-tune the [RoBERTa](https://github.com/pytorch/fairseq/tree/main/examples/roberta) model with **differential privacy**.

The methods include: 
1. Update the full model with native DPSGD.

2. Update lightweight plug-in modules of three parameter-efficient methods.
    *   [Adapter](https://arxiv.org/abs/1902.00751)
    *   [Compactor](https://arxiv.org/abs/2106.04647)
    *   [LoRA](https://arxiv.org/abs/2106.09685)

Our implementation is based on [this repo](https://github.com/dayu11/Differentially-Private-Deep-Learning/tree/main/language). We evaluate our implementation on language understanding tasks from the [GLUE](https://gluebenchmark.com/) benchmark that have more than 10k training samples (MNLI, QQP, QNLI, and SST-2). 


## Organization 

This page introduces the environment we use. You need to install some packages before running a specific method.

We place the implementation of each method in the corresponding folder. To run a target method, you need to go to its folder and follow the instructions there.

## Preliminary Setup

Our implementation is tested on a Linux system with CUDA version 11.0. 

To run the source code, please first install the following packages:

```
python>=3.6
torch>=1.8
numpy
scipy
six
prv_accountant
apex
```

Most of the packages can be installed easily via `pip install`. If you have trouble installing the `apex` package, you can install it via [anaconda](https://www.anaconda.com/) following these [instructions](https://anaconda.org/conda-forge/nvidia-apex).

In addition to the classic [moments accountant](https://arxiv.org/abs/1607.00133), we also include an advanced tool [PRV accountant](https://github.com/microsoft/prv_accountant) to analyze the privacy loss.  You can use the flag `--accountant moments` or `--accountant prv`  to switch between these two accountants.

To install the `prv_accountant` package, go to the folder and run `pip install --editable . --user`.

By default, the experiments are run with half-precison, you need a Volta GPU, e.g., Titan V or Tesla V100, in this setting.

You can also switch to full-precision by adding `--fp32` when running `run_exp.py`.

## Pre-trained Models and Pre-processed Datasets

The data of four tasks are in the `glue_data` folder. They are processed using the instructions in [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md).

You can download the pre-trained RoBERTa-base and RoBERTa-large models at [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta). Then place the unzipped folders (roberta.base and roberta.large) in this folder.
