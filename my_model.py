import torch
import torch.nn as nn

from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
    global_max_pool
)

from graph_my_data import get_edge_weights


# ============================================================
# GRAPH CONVOLUTION LAYER
# ============================================================

class GraphLayer(MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        weighting=None,
        d=1.0
    ):

        super().__init__(aggr="add")

        # Linear transformation applied to the node features.
        self.linear = nn.Linear(
            in_channels,
            out_channels
        )

        # Save the edge weighting choices.
        self.weighting = weighting
        self.d = d


    def forward(
        self,
        x,
        edge_index,
        edge_times
    ):

        # Create one weight for each edge using its time difference.
        edge_weight = get_edge_weights(
            edge_times=edge_times,
            weighting=self.weighting,
            d=self.d
        )

        # Transform the node features.
        x = self.linear(x)

        # Perform message passing across the graph.
        x = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_weight=edge_weight
        )

        return x


    def message(
        self,
        x_j,
        edge_weight
    ):

        # Weight each incoming message by the corresponding edge weight.
        message = (
            edge_weight.view(-1, 1)
            * x_j
        )

        return message


# ============================================================
# GRAPH-LEVEL LOGISTIC OUTPUT
# ============================================================

class LogisticOutput(nn.Module):

    def __init__(
        self,
        in_channels,
        pooling="mean"
    ):

        super().__init__()

        self.pooling = pooling


        # Mean + max doubles the number of features.
        if pooling == "mean_max":

            output_channels = (
                in_channels * 2
            )

        else:

            output_channels = in_channels


        self.linear = nn.Linear(
            output_channels,
            1
        )


    def forward(
        self,
        x,
        batch
    ):

        if self.pooling == "mean":

            x = global_mean_pool(
                x,
                batch
            )


        elif self.pooling == "sum":

            x = global_add_pool(
                x,
                batch
            )


        elif self.pooling == "max":

            x = global_max_pool(
                x,
                batch
            )


        elif self.pooling == "mean_max":

            x_mean = global_mean_pool(
                x,
                batch
            )

            x_max = global_max_pool(
                x,
                batch
            )

            x = torch.cat(
                [
                    x_mean,
                    x_max
                ],
                dim=1
            )


        else:

            raise ValueError(
                "pooling must be 'mean', 'sum', 'max', or 'mean_max'"
            )


        x = self.linear(x)

        return x


# ============================================================
# COMPLETE PATIENT GCN
# ============================================================

class PatientGCN(nn.Module):

    def __init__(
        self,
        weighting=None,
        d=1.0,
        pooling="mean"
    ):

        super().__init__()


        # GRAPH CONVOLUTION LAYERS
        # The number of nodes/features in each hidden layer can
        # still be manually changed here if we want to change
        # the architecture.

        self.gcn1 = GraphLayer(
            in_channels=5,
            out_channels=8,
            weighting=weighting,
            d=d
        )

        self.gcn2 = GraphLayer(
            in_channels=8,
            out_channels=8,
            weighting=weighting,
            d=d
        )

        self.gcn3 = GraphLayer(
            in_channels=8,
            out_channels=8,
            weighting=weighting,
            d=d
        )

        self.gcn4 = GraphLayer(
            in_channels=8,
            out_channels=8,
            weighting=weighting,
            d=d
        )


        # ACTIVATION FUNCTION

        self.relu = nn.ReLU()


        # GRAPH-LEVEL OUTPUT

        self.logistic = LogisticOutput(
            in_channels=8,
            pooling=pooling
        )


    def forward(
        self,
        x,
        edge_index,
        edge_times,
        batch
    ):

        # GRAPH LAYER 1

        x = self.gcn1(
            x,
            edge_index,
            edge_times
        )

        x = self.relu(x)


        # GRAPH LAYER 2

        x = self.gcn2(
            x,
            edge_index,
            edge_times
        )

        x = self.relu(x)


        # GRAPH LAYER 3

        x = self.gcn3(
            x,
            edge_index,
            edge_times
        )

        x = self.relu(x)


        # GRAPH LAYER 4

        x = self.gcn4(
            x,
            edge_index,
            edge_times
        )

        x = self.relu(x)


        # POOL GRAPH AND CREATE ONE PATIENT PREDICTION

        x = self.logistic(
            x,
            batch
        )

        return x