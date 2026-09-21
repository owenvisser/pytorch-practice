import torch
import pandas as pd
import matplotlib.pyplot as plt

from my_model import PatientGCN
from my_training import (
    create_loaders,
    train_model,
    evaluate_model
)

# ============================================================
# 1. DEFINE MODEL SETTINGS
# ============================================================

# Training settings.
n_epochs = 1000
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0


weighting_options = [
    #None,
    "exponential"
    #"gaussian"
]

d_options = [
    #0.25,
    #0.5,
    0.75
    #2.0,
    #5.0
]

pooling_options = [
    #"mean",
    #"sum",
    "max"
    #"mean_max"
]

seed = 100

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. LOAD SAVED GRAPH DATA
# ============================================================

train_graphs = torch.load(
    "data/train_graphs.pt",
    weights_only=False
)

val_graphs = torch.load(
    "data/val_graphs.pt",
    weights_only=False
)

test_graphs = torch.load(
    "data/test_graphs.pt",
    weights_only=False
)

# ============================================================
# 3. CHOOSE DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 4. CREATE DATA LOADERS
# ============================================================

train_loader, val_loader, test_loader = create_loaders(
    train_graphs=train_graphs,
    val_graphs=val_graphs,
    test_graphs=test_graphs,
    batch_size=batch_size
)

# ============================================================
# 5. CREATE EXPERIMENT CONFIGURATIONS
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
# 6. CREATE OBJECTS TO STORE RESULTS
# ============================================================

# Summary statistics for each model will be stored here.
results = []

# Full loss histories will be stored separately so that
# we can later plot the training and validation curves.
loss_histories = []


# ============================================================
# 7. RUN EACH MODEL
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
# 8. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print()
print("FINAL RESULTS")
print(results_df)


# ============================================================
# 9. SORT MODELS BY BALANCED ACCURACY
# ============================================================

results_df = results_df.sort_values(
    "balanced_accuracy",
    ascending=False
)

print()
print("RESULTS SORTED BY BALANCED ACCURACY")
print(results_df)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results/model_results.csv",
    index=False
)


# ============================================================
# 11. PLOT VALIDATION LOSS CURVES
# ============================================================

# Get each unique value of d that was actually used.
plot_d_values = [
    None
] + d_options


# Number of subplot columns.
n_cols = 2

# Calculate how many rows are needed.
n_rows = (
    len(plot_d_values)
    + n_cols
    - 1
) // n_cols


# Create the grid of plots.
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(14, 5 * n_rows)
)

# Flatten the grid so we can loop through the plots easily.
axes = axes.flatten()


# MAKE ONE PLOT FOR EACH VALUE OF D

for i, d_value in enumerate(plot_d_values):

    ax = axes[i]


    for history in loss_histories:

        history_d = history["d"]

        # Only plot models belonging to this value of d.
        if history_d == d_value:

            weighting = history["weighting"]
            pooling = history["pooling"]

            label = (
                f"{weighting}, "
                f"{pooling}"
            )

            ax.plot(
                history["validation_losses"],
                label=label
            )


    # Give the no-weighting models a clearer title.
    if d_value is None:

        ax.set_title(
            "No Edge Weighting"
        )

    else:

        ax.set_title(
            f"d = {d_value}"
        )


    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Validation Loss"
    )


    # Put the legend outside each individual plot.
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )


# REMOVE UNUSED PLOTS
# For example, if we create a 3 x 2 grid but only need 5 panels.

for j in range(
    len(plot_d_values),
    len(axes)
):

    fig.delaxes(
        axes[j]
    )


plt.tight_layout()
plt.show()