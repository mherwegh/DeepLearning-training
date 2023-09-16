import numpy as np

def init_variables():
    """
        Init model variables (weigth and bias)
    """

    weights = np.random.normal(size=2)
    bias = 0
    print(weights)
    return weights, bias

def get_dataset():
    """
        Method used to generate the dataset
    """
    row_per_class = 5 # Number of rows per class
    sick = np.random.randn(row_per_class, 2) + np.array([-2,-2]) # Generate sicks person
    healthy = np.random.rand(row_per_class, 2) + np.array([2,2]) # Generate healthy person

    features = np.vstack([sick, healthy])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) +1))
    print(features)
    print(targets)

    return features, targets

def pre_activation(features, weights, bias):
    """
        Compute pre-activation
    """
    return np.dot(features,weights) + bias

def activation(z):
    """
        Compute activation (sigmoid function)
    """
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    features, targets = get_dataset() # Get dataset
    weights, bias = init_variables() # Variables
    z = pre_activation(features, weights, bias) # Pre-activation function
    a = activation(z) # Activation function
    print ("features:")
    print (features)
    print("weights:")
    print (weights)
    print ("bias:")
    print (bias)
    print("Pre-activation:")
    print (z)
    print ("activation:")
    print (a)
    print ("targets:")
    print (targets)
    pass
