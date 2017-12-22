# UAN
Code for <a href="https://arxiv.org/abs/1708.05207">Learning Universal Adversarial Perturbations with Generative Models</a>

![Alt text](figs/overview.png?raw=true "")

In this paper, we use generative models to compute universal adversarial perturbations. The generator is not conditioned on the images and so creates a perturbation that can be applied to any image to create an adversarial example.

We get pretty pictures like this:

![Alt text](figs/uap_example.png?raw=true "")

Clean Image              +          Perturbation          ==         Adversarial Image  

-------

Here is the output of a UAN throughout training:

![Alt text](figs/pert_evolution_1.gif?raw=true "")

-------

Data set-up

For ImageNet

  - Follow instructions https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data . The validation set should be in path `./imagenet/data/val/`. There should be 1000 directories, each with 50 images.
  
For CIFAR-10

  - Attack code will download if dataset does not exist.

-------

Target model training steps:

For ImageNet

  - Clone https://github.com/Cadene/pretrained-models.pytorch into this directoy and (for easy python importing) rename to "pretrained_models_pytorch"

For CIFAR-10

  - Train some models using https://github.com/kuangliu/pytorch-cifar
    
-------

To run the attack, choose between ImageNet and CIFAR-10 and specify the model.

e.g. `python main.py --cuda --dataset ImageNet --epochs 200 --batchSize 32 --shrink 0.00075 --shrink_inc 0.0001 --l2reg 0.00001 --restrict_to_correct_preds 1 --netClassifier resnet152 --imageSize 224 --outf resnet-results --every 100`


Note: For best results on ImageNet, batch size needs to be large. This takes up a lot of memory.
