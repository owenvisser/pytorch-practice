import torch
import pandas as pd
import matplotlib.pyplot as plt

from sim_data import generate_sim_data
from graph_data import create_graphs
from model import PatientGCN
from training import (
    split_graphs,
    create_loaders,
    train_model,
    evaluate_model
)


# ============================================================
# 1. GENERAL SETTINGS
# ============================================================

# Random seed used for the simulated data and train/validation/test split.
seed = 100

# Training settings.
n_epochs = 1000
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0

# Proportion of patients assigned to each data set.
train_prop = 0.60
val_prop = 0.20
test_prop = 0.20


# ============================================================
# 2. CHOOSE DEVICE
# ============================================================

# Use the GPU if CUDA is available.
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 3. GENERATE SIMULATED DATA
# ============================================================
# here p is the number of features; i've coded strictly for 5, see sim_data.py

long_data, patient_data = generate_sim_data(
    N=5000,
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
# 6. CREATE DATA LOADERS
# ============================================================

train_loader, val_loader, test_loader = create_loaders(
    train_graphs=train_graphs,
    val_graphs=val_graphs,
    test_graphs=test_graphs,
    batch_size=batch_size
)


# ============================================================
# 7. DEFINE MODEL SETTINGS TO COMPARE
# ============================================================

# Edge weighting methods.
weighting_options = [
    None,
    "exponential",
    "gaussian"
]

# Values of d to test for the weighted models.
d_options = [
    0.5,
    1.0,
    2.0,
    5.0
]

# Graph pooling methods.
pooling_options = [
    "mean",
    "sum",
    "max"
]


# ============================================================
# 8. CREATE EXPERIMENT CONFIGURATIONS
# ============================================================

# Each dictionary describes one model that we want to fit.

experiments = []


for weighting in weighting_options:

    for pooling in pooling_options:

        # If there is no edge weighting, d does nothing.
        # Therefore we only fit one model for weighting=None.

        if weighting is None:

            experiments.append({
                "weighting": None,
                "d": None,
                "pooling": pooling
            })


        # If an edge weighting function is being used,
        # fit one model for each value of d.

        else:

            for d in d_options:

                experiments.append({
                    "weighting": weighting,
                    "d": d,
                    "pooling": pooling
                })


print("Number of models to fit:", len(experiments))


# ============================================================
# 9. CREATE OBJECTS TO STORE RESULTS
# ============================================================

# Summary statistics for each model will be stored here.
results = []

# Full loss histories will be stored separately so that
# we can later plot the training and validation curves.
loss_histories = []


# ============================================================
# 10. RUN EACH MODEL
# ============================================================

for model_number, experiment in enumerate(experiments, start=1):


    # Pull the settings out of the experiment dictionary.

    weighting = experiment["weighting"]
    d = experiment["d"]
    pooling = experiment["pooling"]


    print()
    print("============================================================")
    print(
        f"MODEL {model_number} OF {len(experiments)}"
    )
    print("============================================================")

    print("Weighting:", weighting)
    print("d:", d)
    print("Pooling:", pooling)


    # When weighting=None, d is not actually used.
    # PatientGCN still requires a numerical value, so we
    # simply use d=1.0 internally.

    model_d = (
        1.0
        if d is None
        else d
    )


    # CREATE MODEL

    model = PatientGCN(
        weighting=weighting,
        d=model_d,
        pooling=pooling
    )


    # TRAIN MODEL
    # The returned model has already been restored to the
    # parameter values from the epoch with the lowest
    # validation loss.

    (
        model,
        training_losses,
        validation_losses,
        best_validation_loss,
        best_epoch

    ) = train_model(

        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )


    # TEST MODEL
    # The test set is only used after training and model
    # selection are complete.

    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )


    # SAVE SUMMARY RESULTS

    results.append({

        "weighting": weighting,
        "d": d,
        "pooling": pooling,

        "best_epoch": best_epoch,

        "best_validation_loss":
            best_validation_loss,

        "balanced_accuracy":
            test_results["balanced_accuracy"],

        "sensitivity":
            test_results["sensitivity"],

        "specificity":
            test_results["specificity"],

        "TP":
            test_results["TP"],

        "TN":
            test_results["TN"],

        "FP":
            test_results["FP"],

        "FN":
            test_results["FN"]
    })


    # SAVE COMPLETE LOSS HISTORY

    loss_histories.append({

        "weighting": weighting,
        "d": d,
        "pooling": pooling,

        "training_losses":
            training_losses,

        "validation_losses":
            validation_losses
    })


    print(
        "Best validation loss:",
        round(best_validation_loss, 4)
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Balanced accuracy:",
        round(
            test_results["balanced_accuracy"],
            4
        )
    )


# ============================================================
# 11. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print()
print("FINAL RESULTS")
print(results_df)


# ============================================================
# 12. SORT MODELS BY BALANCED ACCURACY
# ============================================================

results_df = results_df.sort_values(
    "balanced_accuracy",
    ascending=False
)

print()
print("RESULTS SORTED BY BALANCED ACCURACY")
print(results_df)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_results.csv",
    index=False
)


# ============================================================
# 14. PLOT VALIDATION LOSS CURVES
# ============================================================

plt.figure(
    figsize=(10, 6)
)


for history in loss_histories:

    weighting = history["weighting"]
    d = history["d"]
    pooling = history["pooling"]


    # Create a label describing the model.
    label = (
        f"{weighting}, "
        f"d={d}, "
        f"{pooling}"
    )


    plt.plot(
        history["validation_losses"],
        label=label
    )


plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss by Model")

plt.legend()

plt.tight_layout()
plt.show()