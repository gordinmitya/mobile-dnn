Approaches to use neural networks efficiently on mobile devices can be separated into the following categories:

* Architecture of DNN
  * Decrease complexity: MobileNets, BlazeFace;
  * Keeping complexity, improve accuracy: Xception, ResNeXt;
* Compressing
  * Weights pruning: "Learning both Weights and Connections for Efficient Neural Networks";
  * Weights reusing: HashNets, Huffman coding;
* Matrix decomposition;
* Knowledge distillation
* Quantization
* Multitask learning
* Neural architecture search

I came up with quite similar division into categories as in [EfficientDNNs](https://github.com/MingSun-Tse/EfficientDNNs). This division does not mean that these methods are incompatible. The best results were obtained by applying a combination of several methods.

## Architecture

Architecture matters. Residual connections allow you to avoid vanishing gradient problem and train deeper CNN networks. Both spatial and depthwise separable convolutions allow you to reduce FLOPs while keeps acceptable accuracy. Also, it worth efforts to build and tune architecture for a particular task: BlazeFace, Prizma Portrait segmentation. MACE framework provides a great tool to analyze the performance of each layer, find a bottleneck,  tune hyperparameters (eg kernel size, number of channels, etc). Keep in mind that the same operation could perform differently depends on framework implementation, CPU/GPU, memory.

BlazeFace https://arxiv.org/abs/1907.05047
Prizma Portrait Segmentation https://blog.prismalabs.ai/39c84f1b9e66
EfficientNet https://arxiv.org/abs/1905.11946

## Compressing

There is a lot of redundancy in neural networks. Both in the way, models are stored as well as in unused connections. For instance, authors of the "DEEP COMPRESSION" paper achieve a compression rate of 49x ( from 552MB to 11.3MB) with VGG-16 without any accuracy loss on the ImageNet classification task.

## Knowledge distillation

Neural networks learn better from other networks than from ground truth data. There could be several reasons for that: 1) Soft class labels instead of hard ones provides more information for the students. For instance, a car looks more like a bus rather than a cat. Also, during the training penalty for misclassifying cars and buses will be less. 2) A teacher model can overcome errors in training data. 3) TODO

## Quantization

The idea of this approach is to convert floating-point weights into integer values with lower precision. Usually from FP32 to INT8. In this case, we get a boost due to the reduction of memory accesses and higher SIMD instruction utilization. To decrease quality degradation "quantization-aware training" can be applied.

The extreme case – train a network with binary weights.
XNOR-NET https://arxiv.org/abs/1603.05279

## Multitask learning

If a task requires to get a different kind of information from the same input data, we can try to combine this task to a single neural network. These tasks could be, for instance, get face embedding along with gender/age information or perform pose estimation along with face detection. First, we reduce the number of computations because we use a single feature extraction part of a network. Second, multitask learning may lead to improved quality of combined networks because more information is provided during training.


## Neural architecture search

Finding hyperparameters for an architecture of a neural network can be formulated as an optimization task. For instance, number of layers, number of filters and kernel size of CNN. It can be done either for designing and training network from scratch or for finetuning existing network.
https://arxiv.org/abs/1611.01578

## Related Repos

I want to credit authors and contributors to:
1) https://github.com/MingSun-Tse/EfficientDNNs;
2) https://github.com/EMDL/awesome-emdl;
3) https://github.com/he-y/Awesome-Pruning;
4) https://github.com/lhyfst/knowledge-distillation-papers.

## Purpose / about me

I am a Ph.D. student and starting research in the field of neural networks on mobile devices. Your suggestions, thoughts, and recommendations related to the theme are more than welcome!
