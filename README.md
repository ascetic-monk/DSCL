# Dual-Space Contrastive Learning for Open-World Semi-Supervised Classification
Pytorch implement of "Dual-Space Contrastive Learning for Open-World Semi-Supervised Classification" in TNNLS 2024 (under review). 
_________________



## Dependencies

To start up, you need to install some packages. Our implementation is based on [PyTorch](https://pytorch.org/). We recommend using `conda` to create the environment and install dependencies and all the requirements are listed in `./requirements.txt`.



## Usage


**Preparation for Dataset and Pretrained Parameters**


- The data preparation is the same as ORCA [link](https://github.com/snap-stanford/orca). For cifar100 and cifar10, they will be downloaded automantically and you don't need to configure the dataset. For the Imagenet100, you need to download the ImageNet dataset first and then generate corresponding splitting lists via ```gen_imagenet_list.py``` .
- For We use SimCLR for pretraining. The weights used in our paper can be downloaded in this [link](https://drive.google.com/file/d/19tvqJYjqyo9rktr3ULTp_E33IqqPew0D/view?usp=sharing).
- The pretrained checkpoints will be saved in ```./pretrained/``` while the datasets will be saved in ```./datasets/```. These settings are the same with the ORCA [link](https://github.com/snap-stanford/orca).
- **You also need to create a empty folder name ```./checkpoints/''' under the root path to store the parameters of the model.**

**Get Started**

- For training on a certain dataset, you can config the argument `dataset` as one of `cifar10,cifar100,imagenet100`. Note that the other argument like `labeled-num` and `labeled-ratio` are fixed in the code. If you want to config it, please refer to the code. 

- Our code automatically selects whether to run on a single GPU or multiple GPUs based on the number of devices specified in CUDA_VISIBLE_DEVICES.

- For example, to train on CIFAR-100 with 50 labeled samples and 0.5 labeled ratio on GPU 0, you can run the following command:
```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --dataset cifar100 --labeled-num 50 --labeled-ratio 0.5 --name 'cifar100_experiment'
```



## License

We use the MIT License. Details see the LICENSE file.



## Contact Us

If you have any questions, you can turn to the issue block or send emails to [1120220290@mail.nankai.edu.cn](mailto:1120220290@mail.nankai.edu.cn). Glad to have a discussion with you!
