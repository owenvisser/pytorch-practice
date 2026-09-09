import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# ============================================================
# 1. USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------
#
# PyTorch and NumPy each have their own random-number generators.
#
# Setting the seeds helps make things such as:
#
#   - train/validation/test splitting
#   - neural-network initialization
#   - minibatch ordering
#
# more reproducible between runs.
#
RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ------------------------------------------------------------
# EDGE WEIGHT TYPE
# ------------------------------------------------------------
#
# Choose how the time difference between two observations
# determines the weight of the edge connecting them.
#
# Available options:
#
#   "none"
#       Every edge receives weight 1.
#
#   "linear"
#       Edge strength decreases linearly with time difference.
#
#   "exponential"
#       Edge strength decreases exponentially with time difference.
#
#   "gaussian"
#       Edge strength decreases according to a Gaussian-shaped
#       function.
#
EDGE_WEIGHT_TYPE = "none"


# Scale parameter used in the weighted edge functions.
#
# Mathematically, this corresponds to our tuning parameter d.
#
# We will eventually want to evaluate the model over multiple
# values of this parameter.
#
EDGE_SCALE = 2.0


# ------------------------------------------------------------
# SELF LOOPS
# ------------------------------------------------------------
#
# If False:
#
#     node j can only receive information through the temporal
#     edges we explicitly constructed.
#
# If True:
#
#     PyTorch Geometric also adds:
#
#         j -> j
#
#     for every node.
#
# This lets a node retain its own information while also
# receiving information from earlier observations.
#
# We are leaving this False for the first experiment.
#
ADD_SELF_LOOPS = False


# ------------------------------------------------------------
# MODEL SIZE
# ------------------------------------------------------------
#
# Number of learned features created by each GCN layer.
#
# Our original node has one observed feature x.
#
# The first GCN maps:
#
#     1 feature -> 16 features
#
# and the second GCN maps:
#
#     16 features -> 16 features
#
HIDDEN_CHANNELS = 16


# ------------------------------------------------------------
# DROPOUT
# ------------------------------------------------------------
#
# Dropout randomly sets some learned features to zero during
# training.
#
# This can help prevent overfitting.
#
# For example:
#
#     DROPOUT_RATE = 0.20
#
# means approximately 20% of features are randomly dropped
# during each training pass.
#
# Setting:
#
#     DROPOUT_RATE = 0.0
#
# completely disables dropout.
#
DROPOUT_RATE = 0.20


# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

# Number of complete passes through the training data.
N_EPOCHS = 100


# Learning rate used by Adam.
LEARNING_RATE = 0.001


# Number of subject graphs processed simultaneously.
#
# Because subjects have different numbers of nodes,
# PyTorch Geometric combines them into one large disconnected
# graph and keeps track of which node belongs to which subject.
#
BATCH_SIZE = 16


# ============================================================
# 2. GPU / CPU SETUP
# ============================================================

# torch.cuda.is_available() checks whether PyTorch can access
# an NVIDIA CUDA GPU.
#
# If a GPU is available:
#
#     device = "cuda"
#
# otherwise:
#
#     device = "cpu"
#
# Later we move both the model and each batch onto this device.
#
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

# Longitudinal data contains one row for every observation.
#
# Example:
#
# subject     time       x       z
#    0        ...       ...     ...
#    0        ...       ...     ...
#    1        ...       ...     ...
#
long_data = pd.read_csv("longitudinal_data.csv")


# Subject-level data contains one row for every subject.
#
# This includes the binary event outcome that we want
# the GCN to predict.
#
subject_data = pd.read_csv("subject_data.csv")


# Convert the subject-level outcomes into a dictionary.
#
# Example:
#
# events = {
#     0: 1,
#     1: 0,
#     2: 1
# }
#
# This lets us retrieve:
#
#     events[subject_id]
#
# rather than repeatedly searching the dataframe.
#
events = subject_data.set_index("subject")["event"].to_dict()


# ============================================================
# 4. ORGANIZE LONGITUDINAL DATA BY SUBJECT
# ============================================================

# Start with an empty dictionary.
#
subjects = {}


# groupby("subject") divides the dataframe according to the
# values in the "subject" column.
#
# On each iteration:
#
#     subject_id
#
# is one subject number, while:
#
#     subject_rows
#
# is a smaller dataframe containing only that subject's rows.
#
for subject_id, subject_rows in long_data.groupby("subject"):

    # Explicitly sort observations by time.
    #
    # This guarantees:
    #
    #     node 0 occurs before node 1
    #     node 1 occurs before node 2
    #     etc.
    #
    # Therefore, node index also represents temporal ordering.
    #
    subject_rows = subject_rows.sort_values("time")


    # Store the observation times and observed x values
    # for this subject.
    #
    subjects[subject_id] = {

        "time": subject_rows["time"].to_numpy(),

        "x": subject_rows["x"].to_numpy()
    }


# ============================================================
# 5. CREATE DIRECTED TEMPORAL EDGES
# ============================================================

# Each observation becomes one node.
#
# Suppose a subject has:
#
#     node 0
#     node 1
#     node 2
#     node 3
#
# ordered so that:
#
#     t0 < t1 < t2 < t3
#
# We construct:
#
#     0 -> 1
#     0 -> 2
#     0 -> 3
#     1 -> 2
#     1 -> 3
#     2 -> 3
#
# We DO NOT construct reverse edges.
#
# Therefore information can only be passed forward in time.
#
for subject_id in subjects:

    times = subjects[subject_id]["time"]


    # edges stores:
    #
    #     [source_node, destination_node]
    #
    edges = []


    # time_differences stores one delta_t for every edge.
    #
    # The entry in time_differences[k] corresponds to
    # the edge stored in edges[k].
    #
    time_differences = []


    # j is the source node.
    for j in range(len(times)):

        # k begins at j + 1.
        #
        # Therefore k always occurs later than j.
        #
        for k in range(j + 1, len(times)):

            # Directed edge:
            #
            #     j -> k
            #
            edges.append([j, k])


            # Time separating the two observations:
            #
            #     delta_t_jk = t_k - t_j
            #
            time_differences.append(
                times[k] - times[j]
            )


    subjects[subject_id]["edges"] = np.array(edges)

    subjects[subject_id]["time_difference"] = np.array(
        time_differences
    )


# ============================================================
# 6. DEFINE EDGE WEIGHT FUNCTION
# ============================================================

def make_edge_weights(time_differences):

    # --------------------------------------------------------
    # NO TIME WEIGHTING
    # --------------------------------------------------------
    #
    # Every existing edge receives:
    #
    #     w_jk = 1
    #
    # Therefore all temporal connections have equal strength.
    #
    if EDGE_WEIGHT_TYPE == "none":

        weights = np.ones_like(
            time_differences,
            dtype=float
        )


    # --------------------------------------------------------
    # LINEAR DECAY
    # --------------------------------------------------------
    #
    #     w_jk = 1 - delta_t / d
    #
    # where:
    #
    #     d = EDGE_SCALE
    #
    # We truncate values below zero.
    #
    elif EDGE_WEIGHT_TYPE == "linear":

        weights = (
            1 - time_differences / EDGE_SCALE
        )

        weights = np.clip(
            weights,
            0,
            None
        )


    # --------------------------------------------------------
    # EXPONENTIAL DECAY
    # --------------------------------------------------------
    #
    #     w_jk = exp(-delta_t / d)
    #
    elif EDGE_WEIGHT_TYPE == "exponential":

        weights = np.exp(
            -time_differences / EDGE_SCALE
        )


    # --------------------------------------------------------
    # GAUSSIAN DECAY
    # --------------------------------------------------------
    #
    #     w_jk = exp(-(delta_t)^2 / d)
    #
    elif EDGE_WEIGHT_TYPE == "gaussian":

        weights = np.exp(
            -(time_differences ** 2)
            / EDGE_SCALE
        )


    else:

        raise ValueError(
            "EDGE_WEIGHT_TYPE must be "
            "'none', 'linear', "
            "'exponential', or 'gaussian'."
        )


    return weights


# ============================================================
# 7. CONVERT SUBJECTS TO PYTORCH GEOMETRIC GRAPHS
# ============================================================

for subject_id in subjects:

    # --------------------------------------------------------
    # NODE FEATURES
    # --------------------------------------------------------
    #
    # Suppose x originally looks like:
    #
    #     [2.1, 3.4, 5.0, 6.2]
    #
    # PyTorch Geometric expects:
    #
    #     number_of_nodes x number_of_features
    #
    # Therefore:
    #
    #     view(-1, 1)
    #
    # converts the vector into:
    #
    #     [[2.1],
    #      [3.4],
    #      [5.0],
    #      [6.2]]
    #
    # This now represents:
    #
    #     4 nodes
    #     1 feature per node
    #
    x = torch.tensor(
        subjects[subject_id]["x"],
        dtype=torch.float32
    ).view(-1, 1)


    # --------------------------------------------------------
    # EDGE INDEX
    # --------------------------------------------------------
    #
    # Our edge list currently looks like:
    #
    #     [[0, 1],
    #      [0, 2],
    #      [1, 2]]
    #
    # which has shape:
    #
    #     E x 2
    #
    # PyTorch Geometric expects:
    #
    #     2 x E
    #
    # with:
    #
    #     first row  = source
    #     second row = destination
    #
    # Therefore .t() transposes the structure.
    #
    edge_index = torch.tensor(
        subjects[subject_id]["edges"],
        dtype=torch.long
    ).t().contiguous()


    # --------------------------------------------------------
    # EDGE WEIGHTS
    # --------------------------------------------------------

    time_differences = subjects[
        subject_id
    ]["time_difference"]


    weights = make_edge_weights(
        time_differences
    )


    edge_weight = torch.tensor(
        weights,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # SUBJECT OUTCOME
    # --------------------------------------------------------
    #
    # y contains the observed binary event:
    #
    #     0 = no event
    #     1 = event
    #
    y = torch.tensor(
        [events[subject_id]],
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # CREATE GRAPH OBJECT
    # --------------------------------------------------------
    #
    # Each Data object represents ONE subject.
    #
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )


    # We can also save the subject ID inside the graph.
    #
    # This is not required for GCN training, but it can be
    # useful later if we want to trace predictions back to
    # individual subjects.
    #
    graph.subject_id = torch.tensor(
        [subject_id],
        dtype=torch.long
    )


    subjects[subject_id]["graph"] = graph


# ============================================================
# 8. CREATE TRAIN / VALIDATION / TEST SETS
# ============================================================

# Convert the dictionary of graphs into an ordinary list.
#
# Each element of graph_list is one complete subject graph.
#
graph_list = [
    subjects[subject_id]["graph"]
    for subject_id in subjects
]


# Get one binary outcome for each graph.
#
# We use these values only to stratify the split.
#
# Stratification attempts to preserve approximately the same
# proportion of event = 0 and event = 1 subjects in each set.
#
graph_labels = [
    int(graph.y.item())
    for graph in graph_list
]


# ------------------------------------------------------------
# FIRST SPLIT
# ------------------------------------------------------------
#
# Keep 70% for training.
#
# The remaining 30% will later be divided equally between:
#
#     validation = 15%
#     test       = 15%
#
train_graphs, temp_graphs = train_test_split(
    graph_list,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=graph_labels
)


# Outcomes for the temporary 30% set.
#
temp_labels = [
    int(graph.y.item())
    for graph in temp_graphs
]


# ------------------------------------------------------------
# SECOND SPLIT
# ------------------------------------------------------------
#
# Divide the remaining 30% in half:
#
#     15% validation
#     15% test
#
validation_graphs, test_graphs = train_test_split(
    temp_graphs,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_labels
)


# ============================================================
# 9. CREATE PYTORCH GEOMETRIC DATA LOADERS
# ============================================================

# PyTorch Geometric's DataLoader allows multiple subject
# graphs to be processed simultaneously.
#
# Suppose:
#
#     subject A has 5 nodes
#     subject B has 8 nodes
#     subject C has 3 nodes
#
# PyTorch Geometric combines these into one larger disconnected
# graph containing:
#
#     5 + 8 + 3 = 16 nodes
#
# It automatically creates a batch vector that remembers:
#
#     which nodes belong to subject A
#     which nodes belong to subject B
#     which nodes belong to subject C
#
# This is what allows global_mean_pool() to later recover
# one representation per subject.
#

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True
)


validation_loader = DataLoader(
    validation_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 10. DEFINE THE GCN
# ============================================================

class SimpleGCN(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # FIRST GCN LAYER
        # ----------------------------------------------------
        #
        # Input:
        #
        #     1 observed feature per node
        #
        # Output:
        #
        #     HIDDEN_CHANNELS learned features per node
        #
        self.gcn1 = GCNConv(
            in_channels=1,
            out_channels=HIDDEN_CHANNELS,
            add_self_loops=ADD_SELF_LOOPS
        )


        # ----------------------------------------------------
        # SECOND GCN LAYER
        # ----------------------------------------------------
        #
        # Input:
        #
        #     HIDDEN_CHANNELS features
        #
        # Output:
        #
        #     HIDDEN_CHANNELS features
        #
        self.gcn2 = GCNConv(
            in_channels=HIDDEN_CHANNELS,
            out_channels=HIDDEN_CHANNELS,
            add_self_loops=ADD_SELF_LOOPS
        )


        # ----------------------------------------------------
        # OUTPUT LAYER
        # ----------------------------------------------------
        #
        # After graph pooling, every subject is represented
        # by one HIDDEN_CHANNELS-dimensional vector.
        #
        # This layer converts that vector into one scalar.
        #
        # The scalar is a LOGIT.
        #
        self.output = nn.Linear(
            HIDDEN_CHANNELS,
            1
        )


    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        batch
    ):

        # ----------------------------------------------------
        # FIRST GRAPH CONVOLUTION
        # ----------------------------------------------------
        #
        # edge_index determines:
        #
        #     WHO communicates with whom
        #
        # edge_weight determines:
        #
        #     HOW STRONGLY each edge contributes
        #
        x = self.gcn1(
            x,
            edge_index,
            edge_weight=edge_weight
        )


        # ReLU introduces nonlinearity.
        x = F.relu(x)


        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------
        #
        # During training, a fraction DROPOUT_RATE of features
        # are randomly set to zero.
        #
        # During evaluation:
        #
        #     self.training = False
        #
        # so dropout automatically turns itself off.
        #
        x = F.dropout(
            x,
            p=DROPOUT_RATE,
            training=self.training
        )


        # ----------------------------------------------------
        # SECOND GRAPH CONVOLUTION
        # ----------------------------------------------------

        x = self.gcn2(
            x,
            edge_index,
            edge_weight=edge_weight
        )


        x = F.relu(x)


        # Apply dropout again after the second GCN.
        x = F.dropout(
            x,
            p=DROPOUT_RATE,
            training=self.training
        )


        # ----------------------------------------------------
        # GRAPH-LEVEL POOLING
        # ----------------------------------------------------
        #
        # Before pooling:
        #
        #     one 16-dimensional vector per NODE
        #
        # After pooling:
        #
        #     one 16-dimensional vector per SUBJECT
        #
        # DataLoader created batch automatically.
        #
        x = global_mean_pool(
            x,
            batch
        )


        # ----------------------------------------------------
        # SUBJECT-LEVEL LOGIT
        # ----------------------------------------------------

        x = self.output(x)


        return x


# ============================================================
# 11. INITIALIZE MODEL
# ============================================================

model = SimpleGCN()


# Move all model parameters onto the selected device.
#
# If CUDA is available, the model now lives on the GPU.
#
model = model.to(device)


# ============================================================
# 12. DEFINE LOSS FUNCTION
# ============================================================

# BCEWithLogitsLoss combines:
#
#     sigmoid
#
# and:
#
#     binary cross entropy
#
# into one numerically stable operation.
#
# Therefore the model itself should return the raw logit.
#
loss_function = nn.BCEWithLogitsLoss()


# ============================================================
# 13. DEFINE OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 14. STORAGE FOR TRAINING HISTORY
# ============================================================

# These lists will contain one loss value per epoch.
#
# Later we will plot:
#
#     epoch vs training loss
#
# and:
#
#     epoch vs validation loss
#
train_loss_history = []
validation_loss_history = []


# ============================================================
# 15. TRAIN THE MODEL
# ============================================================

for epoch in range(N_EPOCHS):

    # --------------------------------------------------------
    # TRAINING MODE
    # --------------------------------------------------------
    #
    # This tells PyTorch that we are training.
    #
    # Importantly, dropout is ACTIVE in this mode.
    #
    model.train()


    total_train_loss = 0.0
    n_train_subjects = 0


    # --------------------------------------------------------
    # LOOP THROUGH TRAINING BATCHES
    # --------------------------------------------------------

    for batch in train_loader:

        # Move all information stored in this batch to the GPU
        # if CUDA is being used.
        #
        batch = batch.to(device)


        # Clear gradients from the previous optimization step.
        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_weight,
            batch.batch
        )


        # Flatten:
        #
        #     [batch_size, 1]
        #
        # into:
        #
        #     [batch_size]
        #
        logits = logits.view(-1)


        labels = batch.y.view(-1)


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = loss_function(
            logits,
            labels
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # PARAMETER UPDATE
        # ----------------------------------------------------

        optimizer.step()


        # Multiply the average batch loss by the number of
        # subjects in the batch.
        #
        # This allows us to calculate the true average loss
        # over all subjects at the end of the epoch.
        #
        number_in_batch = batch.num_graphs

        total_train_loss += (
            loss.item() * number_in_batch
        )

        n_train_subjects += number_in_batch


    average_train_loss = (
        total_train_loss / n_train_subjects
    )


    train_loss_history.append(
        average_train_loss
    )


    # ========================================================
    # VALIDATION LOSS
    # ========================================================

    # Switch to evaluation mode.
    #
    # This turns dropout OFF.
    #
    model.eval()


    total_validation_loss = 0.0
    n_validation_subjects = 0


    # torch.no_grad() tells PyTorch that we do not need
    # gradients.
    #
    # This saves memory and computation during evaluation.
    #
    with torch.no_grad():

        for batch in validation_loader:

            batch = batch.to(device)


            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch
            ).view(-1)


            labels = batch.y.view(-1)


            loss = loss_function(
                logits,
                labels
            )


            number_in_batch = batch.num_graphs


            total_validation_loss += (
                loss.item() * number_in_batch
            )


            n_validation_subjects += (
                number_in_batch
            )


    average_validation_loss = (
        total_validation_loss
        / n_validation_subjects
    )


    validation_loss_history.append(
        average_validation_loss
    )


# ============================================================
# 16. GENERAL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, loader):

    # Turn off dropout.
    model.eval()


    all_labels = []
    all_logits = []


    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)


            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch
            ).view(-1)


            # Move the results back to the CPU before
            # converting them to NumPy arrays.
            #
            all_logits.extend(
                logits.cpu().numpy()
            )


            all_labels.extend(
                batch.y.view(-1).cpu().numpy()
            )


    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)


    # --------------------------------------------------------
    # CONVERT LOGITS TO PROBABILITIES
    # --------------------------------------------------------
    #
    # The GCN returns logits:
    #
    #     -infinity < logit < infinity
    #
    # Sigmoid converts them to:
    #
    #     0 < probability < 1
    #
    probabilities = 1 / (
        1 + np.exp(-all_logits)
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------
    #
    # Use 0.5 as the initial classification threshold.
    #
    # If:
    #
    #     p >= 0.5
    #
    # predict event = 1.
    #
    # Otherwise:
    #
    #     event = 0.
    #
    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        predictions
    )


    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------
    #
    # AUC measures the model's ability to rank subjects
    # according to event risk across every possible threshold.
    #
    # AUC requires both outcome classes to be present.
    #
    if len(np.unique(all_labels)) == 2:

        auc = roc_auc_score(
            all_labels,
            probabilities
        )

    else:

        auc = np.nan


    return {
        "labels": all_labels,
        "logits": all_logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "accuracy": accuracy,
        "auc": auc
    }


# ============================================================
# 17. VALIDATION AND TEST PERFORMANCE
# ============================================================

validation_results = evaluate_model(
    model,
    validation_loader
)


test_results = evaluate_model(
    model,
    test_loader
)


# The following values are now available:
#
# validation_results["accuracy"]
# validation_results["auc"]
#
# test_results["accuracy"]
# test_results["auc"]
#
# test_results["probabilities"]
#
# etc.


# ============================================================
# 18. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

# Epoch numbers:
#
#     1, 2, ..., N_EPOCHS
#
epochs = range(
    1,
    N_EPOCHS + 1
)


plt.figure()

plt.plot(
    epochs,
    train_loss_history,
    label="Training loss"
)

plt.plot(
    epochs,
    validation_loss_history,
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title("GCN Training")
plt.legend()

plt.show()


# ============================================================
# 19. PLOT TEST ROC CURVE
# ============================================================

# Only construct an ROC curve if the test set contains
# both outcome classes.
#
if len(
    np.unique(test_results["labels"])
) == 2:

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            test_results["labels"],
            test_results["probabilities"]
        )
    )


    plt.figure()

    plt.plot(
        false_positive_rate,
        true_positive_rate
    )


    # Reference line corresponding to random classification.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )


    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"Test ROC Curve "
        f"(AUC = {test_results['auc']:.3f})"
    )

    plt.show()