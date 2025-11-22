import numpy as np
import cv2
from numpynn import Model, LayerConvolution2D, LayerMaxPooling2D, LayerFlatten, \
                  LayerDense, ActivationLeakyRelu, ActivationSigmoid, \
                  LayerDropout, LossBinaryCrossentropy, OptimizerAdam, AccuracyCategorical



# 1. Data Loading - Keep the image dimensions (N, 3, 28, 28), load with a data loader
X, y, X_test, y_test = load_crack_data('...path.../crackImages/')
print(X,y, X_test, y_test)
# Shuffle
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

model = Model()

# --- LAYER 1 ---
# Input: (3, 64, 64) -> RGB Images
model.add(LayerConvolution2D(input_channels=3, n_filters=16, kernel_size=3, stride=1, padding=1))
model.add(ActivationLeakyRelu())
model.add(LayerMaxPooling2D(kernel_size=2, stride=2))
# Output: (16, 32, 32)

# --- LAYER 2 ---
model.add(LayerConvolution2D(input_channels=16, n_filters=32, kernel_size=3, stride=1, padding=1))
model.add(ActivationLeakyRelu())
model.add(LayerMaxPooling2D(kernel_size=2, stride=2))
# Output: (32, 16, 16)

# --- CLASSIFIER ---
model.add(LayerFlatten())
# 32 * 16 * 16 = 8192 inputs
# Dense layers
model.add(LayerDense(8192, 64)) 
model.add(ActivationLeakyRelu())
model.add(LayerDropout(0.5))

# --- BINARY OUTPUT ---
# 1 Neuron means "Probability of Crack"
model.add(LayerDense(64, 1)) 
model.add(ActivationSigmoid())


model.set(
    loss=LossBinaryCrossentropy(),
    optimizer=OptimizerAdam(learning_rate=0.001, decay=1e-4),
    accuracy=AccuracyCategorical(binary=True) 
)

model.finalize()

# 4. Train
model.train(X, y, validation_data=(X_test, y_test), 
            epochs=10, batch_size=128, print_every=100)
# 5. Save
model.save('crack_detection_cnn_numpy.model')