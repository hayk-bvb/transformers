import numpy as np
from scipy.special import softmax
# Adjust print setting to display larger decimal places
np.set_printoptions(suppress=True, precision=8)

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
k_d = 1 # square root of k_d = 3 but rounded down to 1 for this example
attention_scores = (Q @ K.transpose())/k_d # NOTE: @ operator same as np.dot()
print("\nScaled Attention Scores")
print(attention_scores)


# Step 5: Apply Softmax to each intermediate attention score
attention_scores[0] = softmax(attention_scores[0]) # x_1
attention_scores[1] = softmax(attention_scores[1]) # x_2
attention_scores[2] = softmax(attention_scores[2]) # x_3
print("\nScaled Softmax Attention scores for each vector")
print(attention_scores)

# Step 6: Final attention representations
# attention value obtained by score1/k_d * V
# NOTE: To obtain Attention(Q, K, V) for x_1, we must multiply the intermediate attention score (attention_scores) by the 3 value vectors
attention1 = attention_scores[0].reshape(-1, 1)
attention1 = attention_scores[0][0] * V[0]
attention2 = attention_scores[0][1] * V[1]
attention3 = attention_scores[0][2] * V[2]
print("\n\nAttention 1 & 2 & 3")
print(attention1)
print(attention2)
print(attention3)

# Step 7: Summing up the results
# The 3 attention values of input_1 obtained will not be summed to obtain the first line of the ouptut matrix
attention_input1 = attention1 + attention2 + attention3
print("\nFirst line of ouput matrix for input #1 ")
print(attention_input1)
print("\n\n")

# Step 8: Steps 1 to 7 for all the inputs
# We assume we have 3 results with learned weights (they were not trained in this example)
# We also assume we are implementing the original Transformer paper. We will have 3 results of 64 dimensions each
print("Step 8: Step 1 to 7 for inputs 1 to 3")
attention_head1 = np.random.random((3, 64))
print(attention_head1)

# Step 9: The output of the heads of the attention sublayer
print("\nStep 9: We assume we have trained the 8 heads of the attention sublayer")
z0h1 = np.random.random((3, 64))
z1h2 = np.random.random((3, 64))
z2h3 = np.random.random((3, 64))
z3h4 = np.random.random((3, 64))
z4h5 = np.random.random((3, 64))
z5h6 = np.random.random((3, 64))
z6h7 = np.random.random((3, 64))
z7h8 = np.random.random((3, 64))
print("shape of one head", z0h1.shape, "dimension of 8 heads", 64*8)

# Step 10: Concatenation of the output of the heads
print("Step 10: Concatenation of heads 1 to 8 to obtain the original 8x64=512 output dimension of the model")
output_attention=np.hstack((z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8))
print(output_attention)