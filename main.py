import numpy as np

class Centroid():
    def __init__(self, cl, acc):
        self.cl = cl
        self.acc = acc
        self.count = 1

    def append(self, data):
        for i, val in enumerate(self.acc):
            self.acc[i] += data[i]
            self.count += 1

def readDatabase(filename):
    dataset = np.loadtxt('databases\\{filename}'.format(filename = filename), delimiter = ',')

    # Shuffing the dataset, once sometimes the data are grouped by class
    np.random.shuffle(dataset)

    # Considering the last column being the class column
    classes = dataset[:, -1]

    # Remove the first column (ID) and the last column (class) from dataset
    dataset = np.delete(dataset, -1, axis = 1)
    dataset = np.delete(dataset, 0, axis = 1)

    return dataset, classes

def determineCentroids(dataset, classes):
    rows, cols = np.shape(dataset)

    stats = {}

    for i, row in enumerate(dataset):
        class_id = str(classes[i])
        if class_id in stats:
            stats[class_id].append(row)
        else:
            stats[class_id] = Centroid(classes[i], row)

    centroids = {}
    for key in stats:
        centroids[key] = stats[key].acc/stats[key].count

    return centroids