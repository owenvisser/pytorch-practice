import os
import torch

from sim_data import generate_sim_data
from graph_my_data import create_graphs
from my_training import split_graphs

# ============================================================
# 1. GENERAL SETTINGS
# ============================================================

# Random seed used for the simulated data and train/validation/test split.
seed = 100
sample_size = 5000

# Proportion of patients assigned to each data set.
train_prop = 0.70
val_prop = 0.15
test_prop = 0.15

os.makedirs(
    "data",
    exist_ok=True
)

# ============================================================
# 3. GENERATE SIMULATED DATA
# ============================================================
# here p is the number of features; i've coded strictly for 5, see sim_data.py

long_data, patient_data = generate_sim_data(
    N=sample_size,
    min_visits=3,
    max_visits=20,
    p=5,
    seed=seed
)

# ============================================================
# 4. CREATE PATIENT GRAPHS
# ============================================================

graphs = create_graphs(
    long_data=long_data,
    patient_data=patient_data
)

print("Number of patient graphs:", len(graphs))

# ============================================================
# 5. SPLIT INTO TRAIN, VALIDATION, AND TEST SETS
# ============================================================

# We do this ONCE before running the different model setups.
# This ensures every model is compared using exactly the same
# patients in the training, validation, and test sets.

train_graphs, val_graphs, test_graphs = split_graphs(
    graphs=graphs,
    train_prop=train_prop,
    val_prop=val_prop,
    test_prop=test_prop,
    seed=seed
)

print("Training patients:", len(train_graphs))
print("Validation patients:", len(val_graphs))
print("Test patients:", len(test_graphs))

# ============================================================
# SAVE DATA
# ============================================================

long_data.to_csv(
    "data/long_data.csv",
    index=False
)

patient_data.to_csv(
    "data/patient_data.csv",
    index=False
)

torch.save(
    list(graphs),
    "data/graphs.pt"
)

torch.save(
    list(train_graphs),
    "data/train_graphs.pt"
)

torch.save(
    list(val_graphs),
    "data/val_graphs.pt"
)

torch.save(
    list(test_graphs),
    "data/test_graphs.pt"
)

print("Data generation and splitting complete.")