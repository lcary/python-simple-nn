import numpy as np
import scipy.special


class NeuralNetwork(object):
    """ Simple 3-layer neural network """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, activation_func=None):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_hidden = self._get_weights(self.input_nodes, self.hidden_nodes)
        self.weights_hidden_output = self._get_weights(self.hidden_nodes, self.output_nodes)
        self.learning_rate = learning_rate

        if activation_func is None:
            self.activation_func = lambda x: scipy.special.expit(x)
        else:
            self.activation_func = activation_func

    @staticmethod
    def _get_weights(w1, w2):
        return np.random.normal(0.0, pow(w1, -0.5), (w2, w1))

    def train(self, inputs, targets):
        inputs = self.list_to_matrix(inputs)
        targets = self.list_to_matrix(targets)

        final_outputs, hidden_outputs = self._calculate_outputs(inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        self.backpropagate(final_outputs, hidden_errors, hidden_outputs, inputs, output_errors)

    def _calculate_outputs(self, inputs):
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        return final_outputs, hidden_outputs

    def backpropagate(self, final_outputs, hidden_errors, hidden_outputs, inputs, output_errors):
        self.weights_hidden_output += self.update_weights(output_errors, final_outputs, hidden_outputs)
        self.weights_input_hidden += self.update_weights(hidden_errors, hidden_outputs, inputs)

    def update_weights(self, next_layer_errors, next_layer_outputs, this_outputs):
        x = next_layer_errors * next_layer_outputs * (1 - next_layer_outputs)
        return self.learning_rate * np.dot(x, np.transpose(this_outputs))

    def list_to_matrix(self, inputs):
        return np.array(inputs, ndmin=2).T

    def test(self, inputs):
        inputs = self.list_to_matrix(inputs)
        final_outputs, _ = self._calculate_outputs(inputs)
        return final_outputs
