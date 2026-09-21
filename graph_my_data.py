import torch

from torch_geometric.data import Data


# ============================================================
# CREATE PATIENT GRAPHS
# ============================================================

def create_graphs(long_data, patient_data):

    graphs = []

    # Loop through one patient at a time.
    for subject, visits in long_data.groupby("subject"):

        # Sort visits by time so the graph moves forward chronologically.
        visits = visits.sort_values("time")

        # Convert the five longitudinal features into a tensor.
        # Each row is one visit and each column is one feature.
        x = torch.tensor(
            visits[["x1", "x2", "x3", "x4", "x5"]].to_numpy(),
            dtype=torch.float32
        )

        # Convert visit times into a tensor.
        times = torch.tensor(
            visits["time"].to_numpy(),
            dtype=torch.float32
        )


        # CREATE DIRECTED TEMPORAL EDGES
        # Each earlier visit points toward every later visit.
        # We also include a self-loop for every node.

        edges = []
        edge_times = []

        for j in range(len(times)):

            # Self-loop j -> j has a time difference of 0.
            edges.append([j, j])
            edge_times.append(
                torch.tensor(0.0)
            )

            # j + 1 makes the remaining edges forward-only.
            for k in range(j + 1, len(times)):

                edges.append([
                    j,
                    k
                ])

                edge_times.append(
                    times[k] - times[j]
                )


        # CONVERT EDGES INTO TENSORS
        # PyTorch Geometric expects edge_index to have shape:
        #
        #     2 x number_of_edges
        #
        # Our list is currently number_of_edges x 2,
        # so we transpose it.

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

        # Stack all of the individual time-difference tensors
        # into one vector.
        edge_times = torch.stack(
            edge_times
        )


        # GET PATIENT OUTCOME
        # Find the binary y value for this patient.

        y = patient_data.loc[
            patient_data["subject"] == subject,
            "y"
        ].iloc[0]

        y = torch.tensor(
            [y],
            dtype=torch.float32
        )


        # CREATE PYTORCH GEOMETRIC GRAPH
        # x = node features
        # edge_index = directed edges
        # edge_times = time difference for each edge
        # y = patient-level binary outcome

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_times=edge_times,
            y=y
        )

        graphs.append(graph)


    return graphs


# ============================================================
# CREATE EDGE WEIGHTS
# ============================================================

def get_edge_weights(
    edge_times,
    weighting=None,
    d=1.0
):

    # Exponential weighting:
    #
    #     w = exp(-|delta t| / d)

    if weighting == "exponential":

        weights = torch.exp(
            -abs(edge_times) / d
        )


    # Gaussian weighting:
    #
    #     w = exp(-(delta t)^2 / d^2)

    elif weighting == "gaussian":

        weights = torch.exp(
            -(edge_times ** 2) / (d ** 2)
        )


    # If no weighting function is chosen,
    # every edge receives weight 1.

    else:

        weights = torch.ones_like(
            edge_times
        )


    return weights