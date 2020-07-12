#%matplotlib inline
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
faces = datasets.fetch_olivetti_faces()
print(faces.DESCR)
print(faces.keys())


# this function is a utility to face images from the dataset
def display_faces(images, label, num2display):
    fig = plt.figure(figsize=(25, 25))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(num2display):
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)

        # p.text(0, 14, str(label[i]), color='red', fontsize=18)
        # p.text(0, 60, str(i))
    fig.show()

display_faces(faces.images[7:20], faces.target[7:20], 6)