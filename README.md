# About

This is a small reference repository consisting of standard experiments for machine learning systems research. 

The models include:
* VGG
* Resnet
* Densenet
* GPT-2
* BERT

The dataset used are:

* SST
* OpenWebText
* Imagenet-1k

All the training scripts are standard. They use Pytorch, Pytorch.Distributed for DDP, Gradio and Slurm. 

As mentioned in the slurm scripts, these scripts are configured to run on a 6 node cluster with A4000 GPUs. Feel free to 
tweak them as needed.