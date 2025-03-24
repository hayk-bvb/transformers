import numpy as np
from scipy.special import softmax

"""Step 1: Input : 3 inputs, d_model=4"""
x = np.array([[1.0, 0.0, 1.0, 0.0], # Input 1
              [0.0, 2.0, 0.0, 2.0], # Input 2
              [1.0, 1.0, 1.0, 1.0]  # Input 3
              ])


"""Step 2: weights 3 dimensions x d_model=4"""
w_query  = np.array([[1, 0, 1],
                     [1, 0, 0],
                     [0, 0, 1],
                     [0, 1, 1]
                    ])
w_key  = np.array([[0, 0, 1],
                   [1, 1, 0],
                   [0, 1, 0],
                   [1, 1, 0]
                    ])
w_value  = np.array([[0, 2, 0],
                     [0, 3, 0],
                     [1, 0, 3],
                     [1, 1, 0]
                    ])
# In this model, we will assume that there is one w_query, w_key, and w_value weight matrix for all inputs.
# Other approaches are possible

# First multiply the input vectors by the w_query weight matrix, followed by w_key and w_value
print("Query: x * w_query")
Q = np.matmul(x, w_query)
print(Q)


print("Key: x * w_key")
K = np.matmul(x, w_key)
print(K)

print("Value: x * w_value")
V = np.matmul(x, w_value)
print(V)

# Step 4: Scaled Attention Score
