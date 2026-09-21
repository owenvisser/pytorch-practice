import copy
import torch

from torch_geometric.loader import DataLoader


# ============================================================
# SPLIT GRAPHS INTO TRAIN, VALIDATION, AND TEST SETS
# ============================================================

def split_graphs(
    graphs,
    train_prop=0.60,
    val_prop=0.20,
    test_prop=0.20,
    seed=100
):

    # Make sure the proportions add up to 1.
    if abs(
        train_prop
        + val_prop
        + test_prop
        - 1.0
    ) > 1e-8:

        raise ValueError(
            "train_prop, val_prop, and test_prop must add to 1."
        )


    # Set the random seed for the split.
    generator = torch.Generator().manual_seed(
        seed
    )


    # Number of graphs/patients.
    n = len(graphs)


    # Calculate the number of patients in each set.
    n_train = int(
        train_prop * n
    )

    n_val = int(
        val_prop * n
    )

    # Give the remaining patients to the test set.
    n_test = (
        n
        - n_train
        - n_val
    )


    # Randomly split the graph list.
    train_graphs, val_graphs, test_graphs = torch.utils.data.random_split(
        graphs,
        [
            n_train,
            n_val,
            n_test
        ],
        generator=generator
    )


    return (
        train_graphs,
        val_graphs,
        test_graphs
    )


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_loaders(
    train_graphs,
    val_graphs,
    test_graphs,
    batch_size=32
):

    # Shuffle the training set because the model is learning
    # from these observations.

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True
    )


    # Validation and test sets do not need to be shuffled.

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# CALCULATE VALIDATION LOSS
# ============================================================

def calculate_validation_loss(
    model,
    val_loader,
    loss_function,
    device
):

    # Put the model into evaluation mode.
    model.eval()

    total_loss = 0.0
    total_graphs = 0


    # No gradients are needed during validation.
    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(
                device
            )


            # Make predictions.
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_times,
                batch.batch
            )


            # Calculate loss.
            loss = loss_function(
                logits.view(-1),
                batch.y.view(-1)
            )


            # Keep track of the total loss weighted by
            # the number of graphs in the batch.

            total_loss += (
                loss.item()
                * batch.num_graphs
            )

            total_graphs += batch.num_graphs


    # Average validation loss per patient.
    average_loss = (
        total_loss
        /
        total_graphs
    )


    return average_loss


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs=1000,
    learning_rate=0.001,
    weight_decay=0.0
):

    # Move the model onto the selected device.
    model = model.to(
        device
    )


    # Binary classification loss.
    # The model returns logits, so we use BCEWithLogitsLoss.
    loss_function = torch.nn.BCEWithLogitsLoss()


    # Adam optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


    # Save the loss from every epoch.
    training_losses = []
    validation_losses = []


    # Start with no best model.
    best_validation_loss = float("inf")
    best_epoch = None
    best_model_state = None


    # TRAINING LOOP

    for epoch in range(n_epochs):

        # Put the model into training mode.
        model.train()

        total_train_loss = 0.0
        total_train_graphs = 0


        for batch in train_loader:

            batch = batch.to(
                device
            )


            # Clear gradients from the previous update.
            optimizer.zero_grad()


            # Forward pass.
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_times,
                batch.batch
            )


            # Calculate training loss.
            loss = loss_function(
                logits.view(-1),
                batch.y.view(-1)
            )


            # Calculate gradients.
            loss.backward()


            # Update model parameters.
            optimizer.step()


            # Keep track of average training loss.
            total_train_loss += (
                loss.item()
                * batch.num_graphs
            )

            total_train_graphs += batch.num_graphs


        # Average training loss for this epoch.
        train_loss = (
            total_train_loss
            /
            total_train_graphs
        )


        # Calculate validation loss after the epoch.
        val_loss = calculate_validation_loss(
            model=model,
            val_loader=val_loader,
            loss_function=loss_function,
            device=device
        )


        # Save both losses.
        training_losses.append(
            train_loss
        )

        validation_losses.append(
            val_loss
        )


        # SAVE BEST MODEL
        # If validation loss improved, keep a copy of the
        # model parameters from this epoch.

        if val_loss < best_validation_loss:

            best_validation_loss = val_loss

            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )


    # RESTORE BEST MODEL
    # The final model returned is not necessarily the model from
    # the last epoch. It is the model with the lowest validation loss.

    model.load_state_dict(
        best_model_state
    )


    return (
        model,
        training_losses,
        validation_losses,
        best_validation_loss,
        best_epoch
    )


# ============================================================
# EVALUATE MODEL ON TEST SET
# ============================================================

def evaluate_model(
    model,
    test_loader,
    device
):

    # Put the model into evaluation mode.
    model.eval()


    true_values = []
    predicted_probabilities = []
    predicted_classes = []


    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(
                device
            )


            # Get model logits.
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_times,
                batch.batch
            )


            # Convert logits into probabilities.
            probabilities = torch.sigmoid(
                logits
            )


            # Convert probabilities into binary predictions.
            predictions = (
                probabilities >= 0.5
            ).int()


            # Save results.
            true_values.extend(
                batch.y.view(-1).cpu().tolist()
            )

            predicted_probabilities.extend(
                probabilities.view(-1).cpu().tolist()
            )

            predicted_classes.extend(
                predictions.view(-1).cpu().tolist()
            )


    # CALCULATE CONFUSION MATRIX VALUES

    TP = 0
    TN = 0
    FP = 0
    FN = 0


    for y_true, y_pred in zip(
        true_values,
        predicted_classes
    ):

        if y_true == 1 and y_pred == 1:
            TP += 1

        elif y_true == 0 and y_pred == 0:
            TN += 1

        elif y_true == 0 and y_pred == 1:
            FP += 1

        elif y_true == 1 and y_pred == 0:
            FN += 1


    # Sensitivity:
    #
    #     TP / (TP + FN)

    sensitivity = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )


    # Specificity:
    #
    #     TN / (TN + FP)

    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else 0.0
    )


    # Balanced accuracy:
    #
    #     (sensitivity + specificity) / 2

    balanced_accuracy = (
        sensitivity
        + specificity
    ) / 2


    results = {
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "true_values": true_values,
        "predicted_probabilities": predicted_probabilities,
        "predicted_classes": predicted_classes
    }


    return results