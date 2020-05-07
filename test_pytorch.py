import torch
import torch.nn as nn

# f = w * X

# Real function: f = 3 * X
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[3], [6], [9], [12]], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)
class LinearRegression(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dimension, output_dimension)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_size)

X_test = torch.tensor([5], dtype=torch.float32)
print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# Training
learning_rate = 0.11
N = 75

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(N):
    # prediction = forward pass
    y_hat = model(X)

    # loss
    l = loss(Y, y_hat)

    # gradients = backward pass
    l.backward() #dloss/dw

    # update weights
    optimizer.step()

    # reset gradient
    optimizer.zero_grad()

    # if i % 2 == 0:
    [w, b] = model.parameters()
    print(f"epoch {i+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")






# numpy code:

# import numpy
#
# # f = w * X
#
# # Real function: f = 3 * X
# X = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
# Y = numpy.array([3, 6, 9, 12], dtype=numpy.float32)
#
# w = 0.0
#
# # model prediction
# def forward(X):
#     return (w * X)
#
# # loss = Means Squared Error
# # MSE = 1/N * (w*X - Y)^2
# def loss(Y, y_hat):
#     return ((y_hat - Y)**2).mean()
#
# # gradient
# # dJ/dw = 1/N 2*x(w*x - y)
# def gradient(X, Y, y_hat):
#     return numpy.dot(2*X, y_hat - Y).mean()
#
# print(f"Prediction before training: f(5) = {forward(5):.3f}")
#
# # Training
# learning_rate = 0.01
# N = 20
#
# for i in range(N):
#     # prediction = forward pass
#     y_hat = forward(X)
#
#     # loss
#     l = loss(Y, y_hat)
#
#     # gradients
#     dw = gradient(X, Y, y_hat)
#
#     # update weights
#     w = w - learning_rate * dw
#
#     print(f"epoch {i+1}: w = {w:.3f}, loss = {l:.8f}")
#
# print(f"Prediction after training: f(5) = {forward(5):.3f}")
