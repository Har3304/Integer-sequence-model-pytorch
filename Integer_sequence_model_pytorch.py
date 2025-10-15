import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.arange(1, 10, 1, dtype=torch.float32).to(device)
x = x.reshape(-1, 1)
y = (x**5).to(device)
rounding_param=1
class SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1, device=device), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(1, device=device), requires_grad=True)
        # self.w3 = nn.Parameter(torch.randn(1, device=device), requires_grad=True)
        self.b1 = nn.Parameter(torch.randn(1, device=device), requires_grad=True)
    def forward(self, x):
        return ((x ** self.w1)) * self.w2 + self.b1



model = SequenceModel().to(device)

train_set = int(0.8 * len(x))
x_train, y_train = x[:train_set], y[:train_set]
x_test, y_test = x[train_set:], y[train_set:]

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred)
    return acc

loss_fn = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100000
round_interval = 1000
total_train_loss, total_test_loss, accuracy = 0, 0, 0

for e in range(epochs):
    prev_total_train_loss = total_train_loss
    prev_total_test_loss = total_test_loss
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    total_train_loss += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % round_interval == 0 and e != 0:
        with torch.no_grad():
            for param in model.parameters():
                param.copy_(torch.round(param * (rounding_param)) / (rounding_param))
    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_fn(test_pred, y_test)
        total_test_loss += test_loss
        acc = accuracy_fn(y_test, test_pred)
    if e % 10000 == 0:
        print(f"Epoch: {e} Loss: {loss} Test Loss: {test_loss} Accuracy: {acc}")
    if test_loss == 0. or loss == 0.:
        break

model.eval()
with torch.inference_mode():
    y_pred = model(x_train)

plt.figure(figsize=(10, 7))
plt.scatter(x_train.cpu(), y_train.cpu(), label='Train', c='b')
plt.scatter(x_test.cpu(), y_test.cpu(), label="Train", c='g')
plt.scatter(x_test.cpu(), test_pred.cpu(), label="Test", c='r')
plt.legend()
plt.show()

model.state_dict()
