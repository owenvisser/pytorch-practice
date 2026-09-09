import numpy as np
import pandas as pd

# ============================================================
# 1. BASIC SIMULATION SETTINGS
# ============================================================

# Reproducibility:
# This makes the random numbers the same every time we run the code.
np.random.seed(123)

# Number of subjects:
# i = 1, ..., n_subjects
n_subjects = 10

# Minimum and maximum number of repeated observations per subject
min_times = 3
max_times = 10


# ============================================================
# 2. MODEL PARAMETERS
# ============================================================

# Longitudinal model:
#
#   X_ij = beta0 + beta1 * t_ij + beta2 * z_i + epsilon_ij
#
# where:
#
#   i = subject
#   j = repeated measurement
#   t_ij = observation time
#   z_i = latent subject-specific variable
#   epsilon_ij = random measurement error

beta0 = 1.0
beta1 = 0.5
beta2 = 1.5

# epsilon_ij ~ N(0, sigma^2)
sigma = 0.5


# Binary outcome model:
#
#   logit(p_i) = alpha0 + alpha1 * z_i
#
# where:
#
#   p_i = P(Y_i = 1 | z_i)
#
# and
#
#   Y_i ~ Bernoulli(p_i)

alpha0 = -0.5
alpha1 = 1.0


# ============================================================
# 3. GENERATE LATENT SUBJECT EFFECTS
# ============================================================

# Generate one latent value z_i for each subject:
#
#   z_i ~ N(0, 1)
#
# This latent variable is shared by BOTH:
#
#   1. the longitudinal measurements X_ij
#   2. the binary outcome Y_i
#
# This is what creates dependence between the longitudinal
# trajectory and the eventual binary outcome.

z = np.random.normal(
    loc=0,
    scale=1,
    size=n_subjects
)


# ============================================================
# 4. GENERATE LONGITUDINAL DATA
# ============================================================

rows = []

for i in range(n_subjects):

    # Generate a subject-specific number of observations:
    #
    #   n_i ~ Discrete Uniform{3, ..., 10}
    #
    # So different subjects can have different numbers
    # of longitudinal measurements.
    n_times_i = np.random.randint(
        low=min_times,
        high=max_times + 1
    )

    # Generate n_i irregular observation times:
    #
    #   t_ij ~ Uniform(0, 10)
    #
    # for j = 1, ..., n_i
    times = np.sort(
        np.random.uniform(
            low=0,
            high=10,
            size=n_times_i
        )
    )

    for t in times:

        error = np.random.normal(
            loc=0,
            scale=sigma
        )

        x = (
            beta0
            + beta1 * t
            + beta2 * z[i]
            + error
        )

        rows.append([
            i,
            t,
            x,
            z[i]
        ])
   


# ============================================================
# 5. CREATE LONGITUDINAL DATAFRAME
# ============================================================

# Each row represents ONE NODE / ONE REPEATED OBSERVATION.
#
# subject = subject identifier
# time    = observation time t_ij
# x       = observed longitudinal value X_ij
# z       = latent variable z_i
#
# IMPORTANT:
# z is included here because this is simulated data and we know
# the truth. We would NOT normally give z to the GCN as a feature.

df = pd.DataFrame(
    rows,
    columns=["subject", "time", "x", "z"]
)

print(df.head(10))


# ============================================================
# 6. GENERATE SUBJECT-LEVEL BINARY OUTCOME
# ============================================================

# First compute the linear predictor:
#
#   eta_i = alpha0 + alpha1 * z_i
#
# In our simulation:
#
#   eta_i = -0.5 + 1.0*z_i

logit_p = alpha0 + alpha1 * z


# Convert the logit into a probability using the logistic function:
#
#                  1
#   p_i = ---------------------
#          1 + exp(-eta_i)
#
# Therefore:
#
#   p_i = P(Y_i = 1 | z_i)

p = 1 / (1 + np.exp(-logit_p))

print(p)


# Generate the binary event:
#
#   Y_i ~ Bernoulli(p_i)
#
# Therefore each subject gets:
#
#   Y_i = 1 with probability p_i
#   Y_i = 0 with probability 1 - p_i

event = np.random.binomial(
    n=1,
    p=p
)


# ============================================================
# 7. CREATE SUBJECT-LEVEL DATAFRAME
# ============================================================

# Unlike df, which has MULTIPLE rows per subject,
# df_subject has exactly ONE row per subject.
#
# event_probability = true simulated P(Y_i = 1)
# event             = realized binary outcome
#
# Again, z is only retained because we know it in simulation.

df_subject = pd.DataFrame({
    "id": np.arange(n_subjects),
    "z": z,
    "event": event,
    "event_probability": p
})

print(df_subject.head(10))


# Now that I have the subject level data I'm going to store them all in a python "dictionary"
# it will have this form
#{
#    "subject": 0,
#    "x": ...,
#    "time": ...,
#    "adj": ...,
#    "edge_index": ...,
#    "edge_weight": ...,
#    "event": 1
#}

# This dictionary will be useful for organizing the data for each subject,
# especially when working with graph neural networks (GCN) where we need
# to represent the graph structure (adjacency matrix, edge indices, edge weights)
# along with the node features (x, time) and the binary outcome (event).

# First we'll need to loop over the subjects, Pandas has a very useful operator called `groupby` that allows us to iterate over each subject's data.
# For example:
df.groupby("subject")
# which returns a GroupBy object that we can iterate over.
# For example:
# for subject_id, group in df.groupby("subject"):
#     # group contains all rows for this subject
#     ...

# We will do
# for subject, subject_df in df.groupby("subject"):
#     print(subject)
#     print(subject_df)
# what this does is iterate over each subject in the dataframe,
# printing the subject ID and the corresponding subset of the dataframe
# that contains all rows for that subject.
# This is a common pattern when working with grouped data in Pandas.

# now let's turn this long data into a graph, here we go!
import torch

# Store all patient graphs here
graphs = []

d = 10

# Now within each subject we're doing to repeat what we did in the graph data example.
for subject, subject_df in df.groupby("subject"):

    # sort by time
    subject_df = subject_df.sort_values("time")

    # convert into a tensor
    time = torch.tensor(
        subject_df["time"].values,
        dtype=torch.float32
    )

    # create a tensor for the features
    x = torch.tensor(
        subject_df[["x"]].values,
        dtype=torch.float32
    )

    # calculate pairwise time differences
    delta_t = torch.abs(time[:, None] - time[None, :])

    # construct an adjacency matrix based on the pairwise time differences
    adj = torch.exp(
        -(delta_t ** 2) / d
    )

    # now we could apply some cutoff instead, that would look like this:
    # cutoff = value
    # adj = adj * (delta_t < cutoff).float()

    # remove self edges
    adj.fill_diagonal_(0)

    # convert adjacency matrix to edge indices and edge weights
    edge_index = adj.nonzero().T

    # get the weight corresponding to each edge
    edge_weight = adj[edge_index[0], edge_index[1]]

    # get the patients binary outcome
    event = df_subject.loc[
        df_subject["id"] == subject,
        "event"
    ].iloc[0]

    event = torch.tensor(event, dtype=torch.float32)

    # store everything belonging to its patient's graph
    graph = {
        "subject": subject,
        "x": x,
        "time": time,
        "adj": adj,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "event": event
    }

    graphs.append(graph)

# now that the loop is done over the patients, let's look at the graph for patient 0
print("this is subject:", graphs[0]["subject"])
print(graphs[0]["x"])
print(graphs[0]["time"])
print(graphs[0]["adj"])
print(graphs[0]["edge_index"])
print(graphs[0]["edge_weight"])
print(graphs[0]["event"])

print(graphs[0])

print(df.groupby("subject").size())