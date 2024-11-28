import numpy as np

class MLP:
    def __init__(self, input_size=9, hidden_size=18, output_size=9):
        self.weights_input_hidden = np.random.uniform(-2.0, 2.0, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-2.0, 2.0, (hidden_size, output_size))
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def activation(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.activation(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_output = self.activation(output_input)
        return output_output