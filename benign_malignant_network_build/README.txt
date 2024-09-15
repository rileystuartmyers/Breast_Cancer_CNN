Attempt at a CNN model that performs binary classification tasks between XRAY images that have either 'benign' or 'malignant' calcifications.

Throughout training, the model hasn't been able to achieve above ~70% accuracy, which I believe can be attributed to one or more of the following...

  - Resolution of input images that the model can accept is too low (current max I can input before my PC runs out of memory is around 700x700). This is made worse by the fact that the original input images
    are around 3700x4000, which would be immensely difficult for any single computer to process, especially with our dataset numbers (we have around 1500 total). 

  - Too much noise in the input images, as many of them have signatures in the top corners or surrounding white/gray lines.

  - The actual parameters of the model may not be optimal (ex. using RELU instead of sigmoid, or choosing to use an unoptimal number of neurons for each layers, etc.)

  - The dataset just doesn't leave room for a model to derive from it :/
