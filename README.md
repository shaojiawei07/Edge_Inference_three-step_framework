# Edge_Inference_three-step_framework
This code is for the [paper](https://arxiv.org/abs/2006.02166): "communication-computation tradeoffs in resource-constrained edge inference".



## Framework

We use a pair of CNNs as encoder and decoder. The communication channel is presented by its transfer function as a non-trainable layer.

![avatar](./encoder_and_decoder.png)

## Implementation

The three-step framework is implemented based on the ResNet18. The network is splitted at the end of each building block.

### Dependency

```
Pytorch 1.2.0
Torchvision 0.4.0
Numpy 1.6.12
```

### Dataset

```
CIFAR-10
```
### How to Run

1. `model_pruning.py` uses magnitude-based pruning method to compress the on-device model with parameter `-sp1`, `-sp2`, `-sp3`, `-sp4`, and `-sp5` to denote the _SP_asity ratios in different building blocks.


## Citation

```
@article{shao2020communication,
  title={Communication-Computation Trade-Off in Resource-Constrained Edge Inference},
  author={Shao, Jiawei and Zhang, Jun},
  journal={arXiv preprint arXiv:2006.02166},
  year={2020}
}
```






