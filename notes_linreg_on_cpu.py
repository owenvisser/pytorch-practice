import torch
import torch.nn as nn

# Choose device
device = torch.device("cpu")

print("Using device:", device)

# making some data

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0]
])

# defining the linear regression model

model = nn.Linear(
    in_features=1,
    out_features=1
)

print(model)
print("starting coefficient:", model.weight)
print("starting intercept:", model.bias)

# running the model through y = bx + c

y_hat = model(x)

print("predicted y:", y_hat)
print("coef weight:", model.weight)
print("intercept:", model.bias)

# Define the loss function
loss_function = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# This does 1000 steps of backpropagation
for epoch in range(1000):

    # 1. Make predictions
    y_hat = model(x)

    # 2. Calculate the loss
    loss = loss_function(y_hat, y)

    # 3. Clear old gradients
    optimizer.zero_grad()

    # 4. Calculate gradients
    loss.backward()

    # 5. Update the weight and intercept
    optimizer.step()

# Now we see how the model has changed.
print("Final coefficient:", model.weight)
print("Final intercept:", model.bias)
print("Final loss:", loss)