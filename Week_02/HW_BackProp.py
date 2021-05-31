import numpy as np
import torch

dtype = torch.float
device = torch.device("cpu")

batch_size = 64
input_size = 3
hidden_size = 2
output_size = 1

# Create random input and output data
x = torch.randn(batch_size, input_size, device=device, dtype=dtype)
y = torch.randn(batch_size, output_size, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(input_size, hidden_size, device=device, dtype=dtype,
                 requires_grad=True)
w2 = torch.randn(hidden_size, output_size, device=device, dtype=dtype,
                 requires_grad=True)

learning_rate = 1e-6

for t in range(50):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum() / batch_size / 2
    loss.backward()
    loss_value = loss.item()

    grad_w1 = w1.grad
    grad_w2 = w2.grad

    grad_w2_manual = ((y_pred - y).T @ h_relu).T / batch_size
    grad_h_relu_manual = (y_pred - y) * w2.T
    grad_h_manual = grad_h_relu_manual * h_relu
    grad_w1_manual = (grad_h_manual.T @ x).T / batch_size

    diff_w1 = (grad_w1 - grad_w1_manual).abs().sum()
    diff_w2 = (grad_w2 - grad_w2_manual).abs().sum()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        w1.grad.zero_()
        w2.grad.zero_()

    print(f'[{t:05}] Loss {loss_value:10.5f} Diff-W1 {diff_w1:10.5f} Diff-W2 {diff_w2:10.5f}')
