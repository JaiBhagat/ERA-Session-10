# Custom ResNet Architecture for CIFAR10

We're going to construct a custom ResNet architecture for CIFAR10 as follows:

- **PrepLayer**: Conv 3x3 s1, p1) >> BN >> RELU [64k]
- **Layer1**:
  - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
  - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
  - Add(X, R1)
- **Layer 2**:
  - Conv 3x3 [256k]
  - MaxPooling2D
  - BN
  - ReLU
- **Layer 3**:
  - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
  - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  - Add(X, R2)
  - MaxPooling with Kernel Size 4
- **FC Layer**
- **SoftMax**

We use a **One Cycle Policy** with the following configuration:
- Total Epochs = 24
- Max at Epoch = 5
- LRMIN = FIND
- LRMAX = FIND
- No Annihilation

**Transforms** used are RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8).
Batch size is set to 512.
We use **ADAM** as the optimizer, and **CrossEntropyLoss** as the loss function.

Our target accuracy for this architecture is **90%**.

## OneCycleLR
OneCycleLR is an advanced strategy known as superconvergence, proposed by Leslie Smith. It fundamentally revolutionizes the way neural networks are trained, making the process almost an order of magnitude quicker than traditional methods. A standout feature of superconvergence is the application of one learning rate cycle and a high maximum learning rate.

A crucial understanding enabling superconvergence training is that greater learning rates naturally regulate the training process. Therefore, to maintain the optimal balance, we need to reduce all other forms of regulation.

Smith advocates for implementing a single cycle of learning rate composed of two equal-length steps. The maximum learning rate is decided based on a range test. Subsequently, a lower learning rate, usually one-fifth or one-tenth of the maximum learning rate, is chosen. The first step involves an escalation from a lower to a higher learning rate, followed by a decrease back to the lower learning rate in the second step.

The superconvergence technique of OneCycleLR demonstrated superior performance, delivering rapid and impressive accuracy results, fulfilling the anticipated outcomes at the beginning of the training process.

The architecture achieved 90% accuracy at the 10th Epoch. The One Cycle Policy offers the advantage of allowing us to train our network within fewer epochs and has been utilized effectively in this scenario.
