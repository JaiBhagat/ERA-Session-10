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

The architecture achieved 90% accuracy at the 10th Epoch. The One Cycle Policy offers the advantage of allowing us to train our network within fewer epochs and has been utilized effectively in this scenario.
