import os

import numpy as np

from network import NeuralNetwork

# training_file = 'mnist_train_100.csv'
training_file, training_len = 'mnist_train_50k.csv', 50000
# testing_file = 'mnist_test_10.csv'
testing_file, testing_len = 'mnist_test_10k.csv', 10000


def read_json_data(filename):
    with open(os.path.join('mnist_data', filename), 'r') as f:
        return f.readlines()


# 28x28 pixel images, one input node per pixel:
input_nodes = 784

# hidden layer should have less nodes than input to encourage
# finding patterns or features shorter than the input itself,
# but larger than the output to avoid restricting net freedom to find features
hidden_nodes = 200

# one output node for each digit value in the output range (0..9)
output_nodes = 10

learning_rate = 0.1

n = NeuralNetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate)


def print_progress(count, total, every=1000):
    if count % every == 0:
        prog = (float(count) / total) * 100.0
        print('Progress:\t{0:.1f}%'.format(prog), end='\r', flush=True)


def scale_values(data):
    # scale values from range 0 to 255 -> range 0.01 to 0.99
    return (np.asfarray(data) / 255.0 * 0.99) + 0.01


print('Training neural net')
epochs = 7
total_training_len = training_len * epochs
for e in range(epochs):
    for i, record in enumerate(read_json_data(training_file)):
        all_values = record.split(',')
        inputs = scale_values(all_values[1:])
        # target outputs should all be 0.01 except for target, which is 0.99:
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)  # no batching?
        print_progress(i + (e * training_len), total_training_len)
print()

print('Testing neural net')
results = []
for i, record in enumerate(read_json_data(testing_file)):
    all_values = record.split(',')
    expect = int(all_values[0])
    inputs = scale_values(all_values[1:])
    outputs = n.test(inputs)
    output = int(np.argmax(outputs))
    if output == expect:
        results.append(1)
    else:
        results.append(0)
    print_progress(i, testing_len)
print()

print('\nAccuracy: {}'.format((sum(results) / len(results)) * 100.0))
