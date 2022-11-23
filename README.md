# CRNet
Single-shot Global and Local Context Refinement Neural Network for Head Detection

### prerequisites

* Python 3.6
* Pytorch 1.10
* CUDA 8.0 or higher

### Data Preparation

* **Brainwash**: The dataset is in VOC format. Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare  datasets. Actually, you can refer to any others. After downloading the data, create softlinks in the folder data.
* **HollywoodHeads**:  Same as Brainwash. 

## Train and Test
We have provided train&test code for CRNet. Just run:

```
python train.py
```

## Citation
@inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}


