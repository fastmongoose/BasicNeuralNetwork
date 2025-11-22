import numpy as np
import cv2
from numpynn import Model, LayerConvolution2D, LayerMaxPooling2D, LayerFlatten, \
                  LayerDense, ActivationLeakyRelu, ActivationSigmoid, \
                  LayerDropout, LossBinaryCrossentropy, OptimizerAdam, AccuracyCategorical



# 1. Data Loading - Keep the image dimensions (N, 1, 28, 28), load with a data loader
X, y, X_test, y_test = load_data_mnist('..path../fashion_mnist_images/')

# Shuffle
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Reshape to (N, Channels, Height, Width) and Normalize
# MNIST is grayscale, so Channels = 1
X = (X.reshape(X.shape[0], 1, 28, 28).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], 1, 28, 28).astype(np.float32) - 127.5) / 127.5

model = Model()

# Low level features
model.add(LayerConvolution2D(input_channels=1, n_filters=32, kernel_size=3, stride=1, padding=1))
model.add(ActivationLeakyRelu())

model.add(LayerConvolution2D(input_channels=32, n_filters=32, kernel_size=3, stride=1, padding=1))
model.add(ActivationLeakyRelu())

model.add(LayerMaxPooling2D(kernel_size=2, stride=2))

model.add(LayerDropout(0.25))

# High level features
model.add(LayerConvolution2D(input_channels=32, n_filters=64, kernel_size=3, stride=1, padding=1))
model.add(ActivationLeakyRelu())

model.add(LayerConvolution2D(input_channels=64, n_filters=64, kernel_size=3, stride=1, padding=1))
model.add(ActivationLeakyRelu())

model.add(LayerMaxPooling2D(kernel_size=2, stride=2))

model.add(LayerDropout(0.25))


# FLATTEN: Input (32, 6, 6) -> Output (32 * 6 * 6) = 1152
model.add(LayerFlatten())

# DENSE LAYERS
model.add(LayerDense(3136, 128))
model.add(ActivationLeakyRelu())
model.add(LayerDropout(0.5))
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())

# 3. Set Loss and Optimizer
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=OptimizerAdamW(decay=5e-4),
    accuracy=AccuracyCategorical()
)

model.finalize()

# 4. Train
model.train(X, y, validation_data=(X_test, y_test), 
            epochs=10, batch_size=128, print_every=100)

model.save('fashion_mnist'.model')