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

* **A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.**

  The main idea – **depthwise separable convolution**. Convolution trick got good support from frameworks. Authors suggest to apply convolution to space and channels separately – it drastically decreases size and computation cost of the model while keeping good accuracy. There is improvement mobilenet v2 with residual connections, and also some experiments “in search for mobilenet v3

* **F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” arXiv preprint, pp. 1610–02 357, 2017.**

  The researcher applies ideas from mobilenet to Inception architecture. He used 60 k80 GPU for a month and archives improvement in 0.08% and 0.04% for top1 and top5 accuracy respectively. While marginally decrease speed. Also, he used a huge google private dataset, so we probably can’t even reproduce these results.

## Compressing

There is a lot of redundancy in neural networks. Both in the way, models are stored as well as in unused connections. For instance, authors of the "DEEP COMPRESSION" paper achieve a compression rate of 49x ( from 552MB to 11.3MB) with VGG-16 without any accuracy loss on the ImageNet classification task.

* **S. Han, J. Pool, J. Tran, and W. Dally, “Learning both weights and connections for efficient neural network,” in Advances in neural information processing systems, 2015, pp. 1135–1143.** <a href="learning_both_weights_and_connections"></a>

  Idea is to remove redundancy in networks by **pruning unnecessary weights** (as animal brain does during maturation), then fine-tune network to work with sparse matrices. It can be applied several times for further compression. Authors archives 9x compression and speedup from 3x to 5x. I think speed up very depends on software (PyTorch, TensorFlow sparse tensor) realization. And after some googling, it turns to have more effect when execution is done on CPU rather than GPU (for GPU is cheaper to multiply dense matrix, than check conditions). Interesting how it will show up on mobile devices.

* **S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding” arXiv preprint arXiv:1510.00149, 2015.**

  In continuation of <a href="#learning_both_weights_and_connections">the previous work</a> authors made huge work in testing combination of compressing techniques on different architectures/datasets/hardware. They achieve 49x compression for VGG network without accuracy loss. Along with storage benefits 200Mb vs 12Mb. It also gives a significant boost in performance and decreases energy consumption.

* **Y. Gong, L. Liu, M. Yang, and L. Bourdev, “Compressing deep convolutional network using vector quantization,” arXiv preprint arXiv:1412.6115, 2014.** 

  Decrease storage space of a model. Because of the number of parameters, 90% of disc space is taken by dense layers. It turns out that simple **k mean vector quantization** achieves good results (8-16 compression ratio with <0.5% accuracy loss). Using structured quantization, split matrix by rows/columns and apply k mean to each separately, shows better results, but requires experiments which structure is the best in particular case. An interesting fact that binarization (keep just sign of weight) also shows pretty good results.

* **E. L. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus, “Exploiting linear structure within convolutional networks for efficient evaluation,” in Advances in Neural Information Processing Systems, 2014, pp. 1269–1277.**

  The idea is to decrease the number of computation/multiplication by **approximating weight matrices with a more compact form**. Authors of the work noticed that the most computationally intensive layers in convolutional networks are first two convolution layers. I think it’s difficult to apply this technique (and authors also mentioned that) because it depends on an actual implementation.

  > "We consider several elementary tensor decompositions based on singular value decompositions, as well as filter clustering methods to take advantage of similarities between learned features."

* **W. Chen, J. Wilson, S. Tyree, K. Weinberger, and Y. Chen, “Compressing neural networks with the hashing trick,” in International Conference on Machine Learning, 2015, pp. 2285–2294.**

  It looks like a weird solution. Authors want to decrease space usage (storage and ram) by **using the same set of parameters randomly (by hash function)** located in a dense layer. They even achieve better accuracy compared with network with the same amount of actual parameters. But the problem is that they didn’t consider execution time. To achieve the same accuracy they increase the amount of computations! For instance, 3 layers network HashNet achieves 1.45% error rate while "classic" one achieves 1.69% with 8 times fewer computations.

  > “Each model is compared against a standard neural network with an equivalent number of stored parameters, Neural Network (Equivalent-Size) (NN). For example, for a network with a single hidden layer of 1000 units and a storage compression factor of 1/10, we adopt a size-equivalent baseline with a single hidden layer of 100 units.”

## Knowledge distillation

Neural networks learn better from other networks than from ground truth data. There could be several reasons for that: 1) Soft class labels instead of hard ones provides more information for the students. For instance, classes of cats and dogs are closer in terms of similarity than cats and cars. Also, during the training penalty for misclassifying cats and dogs will be less. 2) A teacher model can overcome errors in training data. 3) We can use additional data from unlabled part of dataset.



* **G.Hinton, O.Vinyals, and J.Dean,“Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015**

  The idea is to **combine ground truth labels from a training dataset and soft output of the teacher mode**l. The second idea is to use “specialist” models which learned a subset of classes and then distillate their knowledge to a single model.

* **J. Ba and R. Caruana, “Do deep nets really need to be deep?” in Advances in Neural Information Processing Systems 27, Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, Eds. Curran Associates, Inc., 2014, pp. 2654–2662. **

  Authors showed that a shallow net can learn complex functions from deeper models. // TODO

## Quantization

The idea of this approach is to convert floating-point weights into integer values with lower precision. Usually from FP32 to INT8. In this case, we get a boost due to the reduction of memory accesses and higher SIMD instruction utilization. To decrease quality degradation "quantization-aware training" can be applied.

The extreme case – train a network with binary weights.
XNOR-NET https://arxiv.org/abs/1603.05279

## Multitask learning

If a task requires to get a different kind of information from the same input data, we can try to combine this task to a single neural network. These tasks could be, for instance, get face embedding along with gender/age information or perform pose estimation along with face detection. First, we reduce the number of computations because we use a single feature extraction part of a network. Second, multitask learning may lead to improved quality of combined networks because more information is provided during training.


## Neural architecture search

Finding hyperparameters for an architecture of a neural network can be formulated as an optimization task. For instance, number of layers, number of filters and kernel size of CNN. It can be done either for designing and training network from scratch or for finetuning existing network.
https://arxiv.org/abs/1611.01578



// TODO https://ssnl.github.io/dataset_distillation/ – distilate dataset (MNIST from 60k to 10 images), so that nas can be performed a lot faster! (source https://t.me/gonzo_ML/143)

## Related Repos

I want to credit authors and contributors to:
1) https://github.com/MingSun-Tse/EfficientDNNs;
2) https://github.com/EMDL/awesome-emdl;
3) https://github.com/he-y/Awesome-Pruning;
4) https://github.com/lhyfst/knowledge-distillation-papers.

## Purpose / about me

I am a Ph.D. student and starting research in the field of neural networks on mobile devices. Your suggestions, thoughts, and recommendations related to the theme are more than welcome!
