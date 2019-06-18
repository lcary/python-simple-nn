import matplotlib.pyplot as plt
import numpy as np
import os


with open(os.path.join('mnist_data', 'mnist_train_100.csv'), 'r') as f:
    data_list = f.readlines()

values = data_list[0].split(',')
image_array = np.asfarray(values[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
filepath = os.path.join('out', 'mnist_train_eg0.png')
print(f'writing example mnist data to: {filepath}')
plt.savefig(filepath)
