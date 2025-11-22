import numpy as np

class LayerDense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0,weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = (0.01 * np.random.randn(n_inputs, n_neurons)).astype(np.float32)
        self.biases = np.zeros((1, n_neurons), dtype=np.float32)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class LayerDropout:
    def __init__(self, rate):
        self.rate = 1 - rate
        
    def forward(self, inputs, training):
         self.inputs = inputs
         if not training:
             self.output = inputs.copy()
             return
         self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
         self.output = inputs * self.binary_mask
        
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
    

class LayerConvolution2D:
    def __init__(self, input_channels, n_filters, kernel_size, stride=1, padding=0,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.input_channels = input_channels
        
        # Regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        # He Initialization, different than glorot as glorot only works for DenseLayer
        fan_in = input_channels * kernel_size * kernel_size
        self.weights = (np.sqrt(2.0 / fan_in) * np.random.randn(n_filters, input_channels, kernel_size, kernel_size)).astype(np.float32)
        self.biases = np.zeros(n_filters, dtype=np.float32)

    def get_im2col_indices(self, x_shape):
        N, C, H, W = x_shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        i0 = np.repeat(np.arange(self.kernel_size), self.kernel_size)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(out_h), out_w)
        j0 = np.tile(np.arange(self.kernel_size), self.kernel_size * C)
        j1 = self.stride * np.tile(np.arange(out_w), out_h)
        
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), self.kernel_size * self.kernel_size).reshape(-1, 1)

        return k, i, j

    def im2col(self, x):
        self.x_shape = x.shape
        k, i, j = self.get_im2col_indices(x.shape)
        
        # Padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # Extract columns
        # (C * Kh * Kw, N * Oh * Ow)
        cols = x_padded[:, k, i, j] 
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_size * self.kernel_size * self.input_channels, -1)
        return cols

    def col2im(self, cols):
        N, C, H, W = self.x_shape
        H_padded, W_padded = H + 2 * self.padding, W + 2 * self.padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        
        k, i, j = self.get_im2col_indices(self.x_shape)
        cols_reshaped = cols.reshape(C * self.kernel_size * self.kernel_size, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        
        # np.add.at for safe accumulation of gradients
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        
        if self.padding == 0:
            return x_padded
        return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def forward(self, inputs, training):
        self.inputs = inputs
        
        # 1. Transform Input to Columns
        self.x_cols = self.im2col(inputs) # Shape: (Fan_in, Batch*Out_pixels)
        
        # 2. Flatten Weights
        # Shape: (n_filters, Fan_in)
        self.weights_flat = self.weights.reshape(self.n_filters, -1)
        
        # 3. Matrix Multiplication
        # Shape: (n_filters, Batch*Out_pixels)
        output = np.dot(self.weights_flat, self.x_cols) + self.biases.reshape(-1, 1)
        
        # 4. Reshape to Output format
        # (Batch, n_filters, Out_h, Out_w)
        N, C, H, W = inputs.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = output.reshape(self.n_filters, out_h, out_w, N)
        self.output = output.transpose(3, 0, 1, 2)

    def backward(self, dvalues):
        # dvalues shape: (N, n_filters, out_h, out_w)
        
        # 1. Reshape gradients for matrix multiplication
        dvalues_flat = dvalues.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)
        
        # 2. Gradient wrt Weights (dW = dy * x_cols.T)
        self.dweights = np.dot(dvalues_flat, self.x_cols.T).reshape(self.weights.shape)
        
        # 3. Gradient wrt Biases
        self.dbiases = np.sum(dvalues, axis=(0, 2, 3))
        
        # 4. Gradient wrt Inputs (dX_col = W.T * dy)
        dX_col = np.dot(self.weights_flat.T, dvalues_flat)
        
        # 5. Transform Columns back to Image (col2im)
        self.dinputs = self.col2im(dX_col)

        # Regularization gradients
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class LayerMaxPooling2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs, training):
        self.inputs = inputs
        N, C, H, W = inputs.shape
        
        # This speedy version assumes stride == kernel_size and that H, W are divisible by kernel_size.
        
        new_H = H // self.kernel_size
        new_W = W // self.kernel_size
        
        # Reshape: (N, C, H/2, 2, W/2, 2)
        inputs_reshaped = inputs.reshape(N, C, new_H, self.kernel_size, new_W, self.kernel_size)
        
        # Max over the two "2" dimensions (axis 3 and 5)
        self.output = np.max(inputs_reshaped, axis=(3, 5))
        
        # Store index of max value for backward pass (mask)
        # This is a simplified trick for the backward pass
        self.inputs_reshaped = inputs_reshaped

    def backward(self, dvalues):
        # Create an array of zeros matching the reshaped input
        dinputs_reshaped = np.zeros_like(self.inputs_reshaped)
        
        # Broadcast output gradients to match the kernel window shape
        # (N, C, new_H, 1, new_W, 1)
        out_grad_broadcast = np.expand_dims(np.expand_dims(dvalues, axis=3), axis=5)
        
        # Broadcast the output values to create the mask
        # (N, C, new_H, 1, new_W, 1)
        output_broadcast = np.expand_dims(np.expand_dims(self.output, axis=3), axis=5)
        
        # Create boolean mask where input == max
        mask = (self.inputs_reshaped == output_broadcast)
        
        # Apply gradient only to max locations
        # We divide by the sum of the mask to handle cases where multiple inputs are the max
        # (distribute gradient equally among ties)
        dinputs_reshaped = mask * out_grad_broadcast 
        
        # Reshape back to original input shape
        self.dinputs = dinputs_reshaped.reshape(self.inputs.shape)


class LayerFlatten:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.inputs.shape)

class LayerInput:
    def forward(self, inputs, training):
        self.output = inputs


# class LayerConvolution2DSlow:
#     def __init__(self, input_channels, n_filters, kernel_size, stride=1, padding=0,
#                 weight_regularizer_l1=0, weight_regularizer_l2=0,
#                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
#         fan_in = input_channels * kernel_size * kernel_size
        
#         # Scale = sqrt(2 / fan_in) for ReLU/Leaky ReLU
#         scale = np.sqrt(2.0 / fan_in)
        
#         self.weights = scale * np.random.randn(n_filters, input_channels, kernel_size, kernel_size)
#         self.biases = np.zeros(n_filters)
#         self.stride = stride
#         self.padding = padding

#         self.weight_regularizer_l1 = weight_regularizer_l1
#         self.weight_regularizer_l2 = weight_regularizer_l2
#         self.bias_regularizer_l1 = bias_regularizer_l1
#         self.bias_regularizer_l2 = bias_regularizer_l2

#     def forward(self, inputs, training):
#         self.inputs = inputs
#         if self.padding > 0:
#             self.inputs_padded = np.pad(inputs, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
#         else:
#             self.inputs_padded = inputs
#         (batch_size, inp_channels, inp_height, inp_width) = inputs.shape
#         (output_channels, kernel_channels, kernel_height, kernel_width) = self.weights.shape

#         padded_width = inp_width + (2 * self.padding)
#         padded_height = inp_height + (2 * self.padding)

#         output_height = (padded_height - kernel_height) // self.stride + 1
#         output_width = (padded_width - kernel_width) // self.stride + 1
        
#         self.output = np.zeros((batch_size, output_channels, output_height, output_width))

#         for b in range(batch_size):
#             for o in range(output_channels):
#                 for i in range(output_height):
#                     for j in range(output_width):
#                         height_start = i * self.stride
#                         height_end = height_start + kernel_height
#                         width_start = j * self.stride
#                         width_end = width_start + kernel_width
#                         self.output[b, o, i, j] = np.sum(self.inputs_padded[b, :, height_start:height_end, width_start:width_end] * self.weights[o, :, :, :]) + self.biases[0]
    
#     def backward(self, dvalues):
#         self.dweights = np.zeros_like(self.weights)
#         self.dinputs = np.zeros_like(self.inputs)
#         self.dbiases = np.sum(dvalues, axis=(0, 2, 3))

#         (batch_size, output_channels, output_height, output_width) = dvalues.shape
#         (output_channels, input_channels, kernel_height, kernel_width) = self.weights.shape

#         # 1. Calculate dweights
#         for oc in range(output_channels):
#             for ic in range(input_channels):
#                 for kh in range(kernel_height):
#                     for kw in range(kernel_width):
#                         for b in range(batch_size):
#                             for i in range(output_height):
#                                 for j in range(output_width):
#                                     in_row = i * self.stride + kh
#                                     in_col = j * self.stride + kw
#                                     self.dweights[oc, ic, kh, kw] += dvalues[b, oc, i, j] * self.inputs_padded[b, ic, in_row, in_col]

#         (_, _, H, W) = self.inputs.shape

#         for b in range(batch_size):
#             for oc in range(output_channels):
#                 for ic in range(input_channels):
#                     for i in range(H):
#                         for j in range(W):
#                             for kh in range(kernel_height):
#                                 for kw in range(kernel_width):
#                                     out_i_float = (i + self.padding - kh) / self.stride
#                                     out_j_float = (j + self.padding - kw) / self.stride

#                                     if (out_i_float >= 0 and out_i_float < output_height and
#                                        out_j_float >= 0 and out_j_float < output_width and
#                                        out_i_float.is_integer() and
#                                        out_j_float.is_integer()):
                                        
#                                         out_i = int(out_i_float)
#                                         out_j = int(out_j_float)
#                                         self.dinputs[b, ic, i, j] += self.weights[oc, ic, kh, kw] * dvalues[b, oc, out_i, out_j]

#         if self.weight_regularizer_l1 > 0:
#             dL1 = np.ones_like(self.weights)
#             dL1[self.weights < 0] = -1
#             self.dweights += self.weight_regularizer_l1 * dl1
#         if self.weight_regularizer_l2 > 0:
#             self.dweights += 2 * self.weight_regularizer_l2 * self.weights
#         if self.bias_regularizer_l1 > 0:
#             dL1 = np.ones_like(self.biases)
#             dL1[self.biases < 0] = -1
#             self.dbiases += self.bias_regularizer_l1 * dL1
#         if self.bias_regularizer_l2 > 0:
#             self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
    
#     def get_parameters(self):
#         return self.weights, self.biases

#     def set_parameters(self, weights, biases):
#         self.weights = weights
#         self.biases = biases

# class LayerMaxPooling2DSlow:
#     def __init__(self, kernel_size=2, stride=2):
#         self.kernel_size = kernel_size
#         self.stride = stride

#     def forward(self, inputs, training):
#         self.inputs = inputs
#         batch, channels, height, width = inputs.shape

#         output_height = (height - self.kernel_size) // self.stride + 1
#         output_width = (width - self.kernel_size) // self.stride + 1

#         self.output = np.zeros((batch, channels, output_height, output_width))

#         for b in range(batch):
#             for c in range(channels):
#                 for h in range(output_height):
#                     for l in range(output_width):
#                         height_start = h * self.stride
#                         height_end = height_start + self.kernel_size
#                         width_start = l * self.stride
#                         width_end = width_start + self.kernel_size
#                         self.output[b, c, h, l] = np.max(inputs[b, c, height_start:height_end, width_start:width_end])
    
#     def backward(self, dvalues):
#         self.dinputs = np.zeros_like(self.inputs)
#         batch, channels, height, width = self.inputs.shape
#         _, _, h_out, w_out = dvalues.shape

#         for b in range(batch):
#             for c in range(channels):
#                 for h in range(h_out):
#                     for l in range(w_out):
#                         height_start = h * self.stride
#                         height_end = height_start + self.kernel_size
#                         width_start = l * self.stride
#                         width_end = width_start + self.kernel_size

#                         window = self.inputs[b, c, height_start:height_end, width_start:width_end]
#                         max_val = np.max(window)

#                         mask = (window == max)
#                         self.dinputs[b, c, height_start:height_end, width_start:width_end] += mask * dvalues[b, c, h, l]
