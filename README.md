# Project Overview

This repository contains various projects organized into different domains, including Computer Vision, Natural Language Processing, Reinforcement Learning, and more. Click on each section to expand and view the projects within.

## Computer Vision (CV)

<details>
<summary>Click to expand</summary>

### [Image Captioning with Attention](Computer-Vision(CV)/Image_Captioning_with_Attention.ipynb)
- **Description**: Implements an image captioning model using attention mechanisms.
- **Technologies**: TensorFlow, MS-COCO dataset.
- **Objective**: Generate descriptive captions for images by focusing on different parts of the image.

### [DebiasingVAE](Computer-Vision(CV)/Google-Lab-DebiasingVAE.ipynb)
- **Description**: Google Labs: Implements a Debiasing Variational AutoEncoder to address human gender bias.
- **Technologies**: TensorFlow.
- **Objective**: Mitigate bias in data representations. 

### [VAEs-DataVisualization](Computer-Vision(CV)/VAEs-DataVisualization.ipynb)
- **Description**: Focuses on data interpolation in Mnist and Cifar for Variational AutoEncoders.
- **Technologies**: TensorFlow.
- **Objective**: Visualize the latent space and model performance.

### [AutoEncoders-VAEs-GANs](Computer-Vision(CV)/AutoEncoders_VAEs_GANs.ipynb)
- **Description**: Explores AutoEncoders, Variational AutoEncoders (VAEs). Future work includes implementing more architectures like GANs and StyleGAN and exploring conditional generation.
- **Technologies**: TensorFlow.
- **Objective**: Understand the performance and transitions between these models.

### [VQ-VAE](Computer-Vision(CV)/VQ-VAE.ipynb)
- **Description**: Explores Vector Quantized Variational AutoEncoders (VQ-VAE).
- **Objective**: Build from scratch to better understand the quantization process, the discrete latent space and the learning representation encoded in the codebook.

### [StyleTransfer](Computer-Vision(CV)/SytleTransfer.py)
- **Description**: Implements style transfer using a pre-trained VGG19 model.
- **Technologies**: TensorFlow, VGG19.
- **Objective**: Blend content and style images by capturing style representations.


### [Image Segmentation](Computer-Vision(CV)/ImageSegmentation/ImageSegmentation.py)
- **Description**: Focuses on understanding and implementing image segmentation techniques.
- **Technologies**: TensorFlow.
- **Objective**: Segment images into meaningful parts.


</details>

## Natural Language Processing (NLP)

<details>
<summary>Click to expand</summary>

### [Image Captioning with Attention](path/to/NLP_Image_Captioning_with_Attention.ipynb)
- **Description**: Similar to the CV project, focuses on the textual aspect of image captioning with attention mechanisms.
- **Objective**: Generate accurate captions for images.

### [NMT-AttentionMechanisms](Natural-Language-Processing(NLP)/NMT-AttentionMechanisms.ipynb)
- **Description**: Neural Machine Translation using attention mechanisms.
- **Objective**: Translate text from one language to another (english -> spanish).

### [Shakespeare_Poesy_NLP](Natural-Language-Processing(NLP)/Shakespeare_Poesy_NLP.ipynb)
- **Description**: Generates Shakespearean poetry using a character-level LSTM.
- **Objective**: Generate poetry with a focus on Shakespearean style.

</details>

## Reinforcement Learning (RL)

<details>
<summary>Click to expand</summary>


### [CartPole](Reinforcement-Learning(RL)/CartPole.ipynb)
- **Description**: OpenAI Gym: CartPole environment.
- **Technologies**: OpenAI Gym, TensorFlow.
- **Objective**: Reinforcement learning experiment (DQN, A2C,  Policy Gradients).

This is the most optimized game in terms of training time. Easy to replicate. The rest of the games are more complex and require more time to train.


### [MountainCar](Reinforcement-Learning(RL)/MountainCar-v0/DQN-Tensorflow.ipynb)
- **Description**: OpenAI Gym: MountainCar environment.
- **Technologies**: OpenAI Gym, TensorFlow.
- **Objective**: Reinforcement learning experiment.

### [LunarLander](Reinforcement-Learning(RL)/LunarLander.ipynb)
- **Description**: OpenAI Gym: LunarLander environment.
- **Technologies**: OpenAI Gym, TensorFlow.
- **Objective**: Reinforcement learning experiment.

### [Pendulum](Reinforcement-Learning(RL)/Pendulum/Pendulum.ipynb)
- **Description**: OpenAI Gym: Pendulum environment.
- **Technologies**: OpenAI Gym, TensorFlow.
- **Objective**: Reinforcement learning experiment.

</details>

## Data Augmentation

<details>
<summary>Click to expand</summary>

### [DataAugmentationCodes](Data-Augmentation/DataAugmentation.md)
- **Description**: Contains code for various data augmentation techniques.
- **Technologies**: TensorFlow.
- **Objective**: Enhance dataset diversity through image translation and resizing.

</details>

## Tensorflow-ML

<details>
<summary>Click to expand</summary>

### [CustomCallback](Tensorflow-ML/CustomCallback.ipynb)
- **Description**: Explores the use of custom callbacks in TensorFlow.
- **Objective**: Enhance model training and evaluation processes.

</details>

## Miscellaneous

<details>
<summary>Click to expand</summary>

### [Contrastive-Loss](Tensorflow-ML/ContrastiveLoss.py)
- **Description**: Implements a Siamese network optimized with contrastive loss.
- **Objective**: Separate different classes in higher-dimensional space.

### [CustomTrainingLoop](Tensorflow-ML/CustomTrainingLoop.py)
- **Description**: Demonstrates how to implement a custom training loop.
- **Objective**: Customize the training process for specific needs.

</details>