import numpy as np
import matplotlib.pyplot as plt
import copy


with open('Toro.pgm', 'r') as f:
    # Read from pgm file
    A = f.readlines()
    column, row = np.array(A[2].split(' ')).astype('int')
    A = np.array(A[4:]).astype('int').reshape(row, column)

U, s, V = np.linalg.svd(A, full_matrices=True)  # Execute SVD

A_svd = np.zeros((row, column))
A_svd_arr = []
append_interval = np.array([10, 25, 50])

for i, sigma in enumerate(s):
    # Add terms in descending order
    A_svd += s[i] * U[:, i:i+1] * V[i:i+1, :]
    if np.any(append_interval == i):
        A_svd_arr.append(copy.deepcopy(A_svd))

print('Singular values: {}'.format(s[:5]))
# >> Singular values: [ 70325.90261891, 2522.70275689, 2038.5399065, 1889.48437463, 1787.74643761]

# Plot
plt.figure(figsize=(7.5, 5))
for i in range(3):
    plt.subplot(2, 2, i+1, xticks=[], yticks=[])
    plt.title('i = {0}, elements = {1}'.format(append_interval[i],
                                               append_interval[i] * (row + column)))
    plt.axis('off')
    plt.imshow(A_svd_arr[i], cmap='gray', vmin=50, vmax=255)

plt.subplot(2, 2, 4, xticks=[], yticks=[])
plt.title('i = {0}, elements = {1} (original)'.format(len(s), row * column))
plt.axis('off')
plt.imshow(A, cmap='gray', vmin=50, vmax=255)

plt.show()
