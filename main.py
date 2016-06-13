import numpy as np
import random

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


# Reads and normalize the database, returns the data and classes apart
def readDatabase(filename):
    dataset = np.loadtxt('databases\\{filename}'.format(filename = filename), delimiter = ',')

    # Shuffing the dataset, once sometimes the data are grouped by class
    np.random.shuffle(dataset)

    # Considering the last column being the class column
    classes = dataset[:, -1]

    # Remove the first column (ID) and the last column (class) from dataset
    dataset = np.delete(dataset, -1, axis = 1)
    dataset = np.delete(dataset, 0, axis = 1)

    # Normalizing the data in the [0 1] interval
    arr_max = np.max(dataset, axis = 0) # gets the max of each column
    arr_min = mp.min(dataset, axis = 0) # gets the min of each column

    rows, cols = np.shape(dataset)
    for i in range(rows):
        for j in range(cols):
            dataset[i][j] = (data[i][j] - arr_min[j]) / (arr_max[j] - arr_min[j])

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
def costFunction(dataset, classes, cl, centroid):
    # 'cl' will be the string representation of the class already
    distances_sum = 0
    count = 0
    for i, d in enumerate(dataset):
        if str(classes[i]) == cl: # limiting the search only in the specific class
            distances += euclidianDistance(d, centroids[cl])
            count += 1

    return distances / count

def fitnessFunction(dataset, classes, centroids):
    fitness = centroids.copy()
    for key in fitness:
        fitness[key] = 1/(1 + costFunction(dataset, classes, key, centroid[key]))
        # TODO: this function can be optimized by storing the costs for each class

    return fitness


# Artificial Bee Colony algorithm implementation
def ABC(dataset, classes, centroids):
    n_data, n_attr = np.shape(dataset) # Number of cases and number of attributes in each case
    n_bees = len(centroids) # Number of bees in the problem
    var_min = 0 # Minimum possible for each variable
    var_max = 1 # Maximum possible for each variable
    max_iter = 100 # Maximum number of iterations
    a_limit = 100 # Abandonment limit

    keys = [key for key in centroids] # centroid keys

    # Initialize the counter of rejections array
    C = centroids.copy()
    for key in F:
        C[key] = 0

    for it in range(max_iter):
        # Employed bees phase
        for cl in centroids:
            _keys = keys.copy() # copying to maintain the original dict
            del _keys[cl]
            k = random.choice(_keys) # getting a index k different from i

            # Define phi coefficient to generate a new solution
            phi = np.random.uniform(-1, 1, n_attr)

            # Generating new solution
            # centroids: numpy array
            # phi: numpy array
            # (centroids[cl] - centroids[j]): numpy array
            # The operation will be element by element given that all the operands
            # are numpy arrays
            new_solution = centroids[cl] + phi * (centroids[cl] - centroids[j])
            # TODO: ceil and floor of the new solution

            _centroids = centroids.copy()
            _centroids[cl] = new_solution

            # Calculate the cost of the dataset with the new centroid
            new_solution_cost = costFunction(dataset, classes, cl, _centroids[cl])

            # Greedy selection: comparing the new solution to the old one
            if new_solution_cost <= costFunction(dataset, classes, cl, centroids[cl]):
                centroids[cl] = new_solution
            else: 
                # Increment the counter for discarted new solutions
                C[cl] += 1

            F = fitness(dataset, classes, centroids) # calculate fitness of each class
            f_sum_arr = [val for key, val in F]
            f_sum = np.sum(f_sum_arr)
            P = {} # probabilities of each class
            for key in F:
                P[key] = F[key]/f_sum

            # Onlooker phase


d, c = readDatabase('glass.data')
stats, centroids = determineCentroids(d, c)

