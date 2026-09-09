import pandas as pd
import torch

df = pd.DataFrame({
    "id":   [1, 1, 1, 1],
    "x1":   [0.2, 0.7, 0.4, 0.9],
    "time": [5.0, 8.0, 3.0, 10.0]
})
print(df)

# convert time to a tensor
time = torch.tensor(
    df["time"].values,
    dtype=torch.float32
)
print(time)

# pairwise absolute time differences
delta_t = torch.abs(
    time[:, None] - time[None, :]
)
print(delta_t)

# thresholding the time graph
cutoff = 5.0
print(cutoff)

# converting the time differences to an adjacency matrix
adj = (delta_t < cutoff).int()
print(adj)

# removing self edges
adj.fill_diagonal_(0)
print(adj)

# converting to an edge_index
edge_index = adj.nonzero().T
print(edge_index)

# now instead of cutoffs let's use edge weights
d = 10.0

# weights have the form exp( - ( t_1 - t_2 )^2 / d )
weighted_adj = torch.exp(
    -(delta_t ** 2) / d
)

# removing self edges from the weighted adjacency matrix
weighted_adj.fill_diagonal_(0)

print(weighted_adj)

# we can still create an edge matrix from this
edge_index = weighted_adj.nonzero().T
print(edge_index)

# but we also want to keep track of the weights
edge_weight = weighted_adj[
    edge_index[0],
    edge_index[1]
]
print(edge_weight)

# so overall:
# X           = node features
# edge_index  = which nodes are connected
# edge_weight = strength of each connection

