import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# for i in range(10):
#     plt.imshow(train[i:i+1].values[0][1:].reshape(28, 28), cmap='gray')
#     print(train[i:i+1].values[0][0])
#     plt.show()

y = train['label'].values
X = train.drop('label', axis=1).values
X_test = test.values

random_tree = RandomForestClassifier()
random_tree.fit(X, y)
ans = random_tree.predict(X_test)

with open('output.csv', 'w') as f:
    print('ImageId,Label', file=f)
    for image_id, label in enumerate(ans):
        print(image_id+1, label, sep=',', file=f)


