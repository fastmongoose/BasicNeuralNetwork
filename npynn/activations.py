import numpy as np

class ActivationReLu:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

# Almost identical to ReLu for shallower networks
class ActivationLeakyRelu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(self.alpha * inputs, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] *= self.alpha

    def predictions(self, outputs):
        return outputs
        
        
# GeLU does nothing different to the final model. ReLU is enough for small models
class ActivationGeLu:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 0.5 * inputs * (1 + np.tanh((np.sqrt(2/np.pi) * (inputs + 0.044715 * (inputs ** 3)))))
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        x = self.inputs
        k = np.sqrt(2 / np.pi)
        c = 0.044715
        u = k * (x + c * (x**3))
        u_prime = k * (1 + 3 * c * (x**2))
        tanh_u = np.tanh(u)
        sech_u_squared = 1 - (tanh_u**2)
        term1 = 0.5 * (1 + tanh_u)
        term2 = 0.5 * x * sech_u_squared * u_prime
        deriv_GeLu = term1 +term2
        self.dinputs = self.dinputs * deriv_GeLu

    def predictions(self, outputs):
        return outputs


class ActivationSoftmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output)- np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class ActivationSigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    
class ActivationLinear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs