import numpy as np

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

if __name__ == '__main__':
    get_dataset()
    pass
