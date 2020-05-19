import numpy as np

A = np.array(
    [[1,2,3],
    [4,5,6]]
)

# Transpose
AT = A.T

# Multiply two Matrices
# (2,3) * (3, 2) -> (2, 2)

A2 = A.dot(A.T)

A2 = A.dot(A, np.transpose(A))
np.transpose(A)

# tensors

A_zeros = np.zeros(shape= (2,3,2))

A_ones = np.ones(shape=(3,2,2))

B = np.arange(2, 49, 2)

B_t = B.reshape(2, 3, 4)

B_t.shape
B_t.ndim


from numpy import random as rnd

min_value = -1
max_value = 1
number_of_samples = 300

U = rnd.uniform(min_value, max_value, size=number_of_samples) # uniform samples in range (min_value, max_value)

f"Mean: {U.mean():.3f} Variance: {U.var():.2f} Std: {U.std():.2f}"


def simulate(node, T, flist):
    if len(flist) > 1999:
        return flist
    else:
        choice = random.choice(T[node])
        flist = flist.append(choice)
        simulate(choice,T, flist)


transition = {'A': 'BE', 'B': 'AFC',
              'C': 'BGD',
              'D': 'CH',
                'E': 'AF',
 'F': 'EBG',
 'G': 'FCH',
 'H': 'GD'}


# keys = np.fromiter(transition.keys(), dtype=np.float64)
# vals = np.fromiter(transition.values())


matrix = np.zeros((8,8))
keys = [*transition]
for key in keys:
    for value in transition[key]:
        row = keys.index(key)   # 0
        column = keys.index(value)  # 1
        matrix[row, column] = (1 / len(transition[key]))
        # matrix[column, row] += (1 / 8)



x = np.linspace(-2,6,100)