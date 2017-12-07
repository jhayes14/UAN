# UAN
Code for Universal Adversarial Networks

Target model training steps:

For ImageNet

  - Clone https://github.com/Cadene/pretrained-models.pytorch into this directoy and (for easy python importing) rename to "pretrained_models_pytorch"

For CIFAR-10

  - Train some models using https://github.com/kuangliu/pytorch-cifar
  
  
-------

To run the attack, choose between ImageNet and CIFAR-10 and specify the model.

e.g. "python main.py --cuda --epochs 200 --batchSize 25 --shrink 0.00075 --shrink_inc 0.0001 --l2reg 0.00001 --restrict_to_correct_preds 1 --netClassifier resnet152 --imageSize 224 --outf resnet-results --every 100"


Note: For best results on ImageNet, batch size needs to be large. This takes up a lot of memory.
