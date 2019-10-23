Approaches to use neural networks on mobile devices efficiently can be separated into the following categories:

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

## Frameworks

* [TensorflowLite](https://www.tensorflow.org/lite) Forced by Google a lot. Now offers [execution on mobile GPU](https://medium.com/tensorflow/e15797e6dee7) with OpenGL ES Compute Shaders on Android and Metal Compute Shaders on iOS.
* [CoreML by Apple](https://developer.apple.com/documentation/coreml) Has some disadvantages like no model protection, limits the GPU performance (for battery saving). (iOS ONLY)
* [OpenCV DNN](https://docs.opencv.org/master/d0/d6c/tutorial_dnn_android.html) Does not utilize mobile GPU but gets some boost from NEON instructions (Single Instruction Multiple Data for ARM). It works quite good for simple cases.
* [Tensorflow Mobile](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android) (Deprecated) Not specially designed for mobile devices, but in a case, with [MegaDepth](https://github.com/zhengqili/MegaDepth) on Android, it showed much better results than TFLite (at the beginning of 2019).
* [MACE by Xiaomi](https://github.com/XiaoMi/mace) Provides [a great tool for benchmarking](https://mace.readthedocs.io/en/latest/user_guide/benchmark.html) reports time consumption of each layer and helps to find a bottleneck. (Android ONLY)
* [SNPE by Qualcomm](https://developer.qualcomm.com/docs/snpe/overview.html) Developed, especially for Snapdragon processors. Among CPU and GPU supports [DSP (special coprocessor)](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor) execution. (Android/Snapdragon ONLY)

Frameworks I have not tried yet:

* [ncnn by Tencent](https://github.com/Tencent/ncnn) Uses Vulkan API to get GPU acceleration.
* [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) Inference engine and optimization tools. Supports different hardware: ARM CPU, Mali GPU, Adreno GPU, Huawei NPU, and FPGA.
* [HiAI by Huawei](https://developer.huawei.com/consumer/en/hiai) Developed for Kirin processors supports NPU execution. Also, it provides several solutions (eg. face detection, text recognition) out of the box. (Android+Kirin ONLY)

* [PyTorch Mobile](https://pytorch.org/mobile/home/) Was introduced recently. No insides yet.
* [Smelter by Prisma](https://github.com/prisma-ai/Smelter) Young framework, but used for production in [huge and popular applications]([https://prisma-ai.com](https://prisma-ai.com/)). It utilizes metal shaders for GPU acceleration. (iOS ONLY)

Not a frameworks, but worth paying attention to:

- [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/) Common interface for frameworks to access hardware acceleration. Android 8.1 (27+). Used in TFLite.

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
