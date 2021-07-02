import torch

dtype = torch.float
device = torch.device("cpu")

batch_size = 64
input_size = 3
hidden_size = 2
output_size = 1

# Create random input and output data
torch.random.manual_seed(888)
x = torch.randn(batch_size, input_size, device=device, dtype=dtype)
y = torch.randn(batch_size, output_size, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(input_size, hidden_size, device=device, dtype=dtype,
                 requires_grad=True)
b1 = torch.zeros(1, hidden_size, device=device, dtype=dtype,
                 requires_grad=True)
w2 = torch.randn(hidden_size, output_size, device=device, dtype=dtype,
                 requires_grad=True)
b2 = torch.zeros(1, output_size, device=device, dtype=dtype,
                 requires_grad=True)

learning_rate = 0.01

for t in range(50):
    # Forward pass: compute predicted y
    h = x.mm(w1) + b1
    h.retain_grad()
    h_relu = torch.relu(h)
    h_relu.retain_grad()
    y_pred = h_relu.mm(w2) + b2
    y_pred.retain_grad()

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum() / batch_size / 2
    loss.retain_grad()
    loss.backward()
    loss_value = loss.item()

    if t % 5 == 0:
        print(f'--------------- Step {t:5} ---------------')
        grad_y_pred_manual = (y_pred - y) / batch_size
        diff_y_pred = (y_pred.grad - grad_y_pred_manual).abs().sum()
        print('diff_y_pred', diff_y_pred)

        grad_w2_manual = (grad_y_pred_manual.T @ h_relu).T
        diff_w2 = (w2.grad - grad_w2_manual).abs().sum()
        print('diff_w2    ', diff_w2)

        grad_b2_manual = grad_y_pred_manual.sum(dim=0, keepdim=True)
        diff_b2 = (b2.grad - grad_b2_manual).abs().sum()
        print('diff_b2    ', diff_b2)

        grad_h_relu_manual = grad_y_pred_manual @ w2.T
        diff_h_relu = (h_relu.grad - grad_h_relu_manual).abs().sum()
        print('diff_h_relu', diff_h_relu)

        grad_h_manual = grad_h_relu_manual * (h_relu > 0)
        diff_h = (h.grad - grad_h_manual).abs().sum()
        print('diff_h     ', diff_h)

        grad_w1_manual = (grad_h_manual.T @ x).T
        diff_w1 = (w1.grad - grad_w1_manual).abs().sum()
        print('diff_w1    ', diff_w1)

        grad_b1_manual = grad_h_manual.sum(dim=0, keepdim=True)
        diff_b1 = (b1.grad - grad_b1_manual).abs().sum()
        print('diff_b1    ', diff_b1)

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * grad_w1_manual
        w2 -= learning_rate * grad_w2_manual
        b1 -= learning_rate * grad_b1_manual
        b2 -= learning_rate * grad_b2_manual

        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()

    print(f'[{t:05}] Loss {loss_value:10.5f}')
