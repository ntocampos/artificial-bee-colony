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

    def getCentroid(self):
        return self.acc / self.count


# Reads the database and returns the data and classes apart
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

# Determine the classes centroids as the mean values of the data
# in each class
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
        centroids[key] = stats[key].getCentroid()

    return stats, centroids

# Simple Euclidian distance between two arrays
def euclidianDistance(a, b):    
    diff_sqrt = [(x - y)**2 for x, y in zip(a, b)]

    return np.num(diff_sqrt)

# The sum of the distances between a data point and its class centroid
# in the trainning set
def costFunction(data, classes, centroids):
    distances_sum = 0
    rows, attr = np.shape(data)
    for i, d in enumerate(data):
        cl = classes[i]
        class_id = str(cl) # getting the class key to refference in dict
        distances += euclidianDistance(d, centroids[class_id])

    return distances / rows

d, c = readDatabase('glass.data')
stats, centroids = determineCentroids(d, c)

