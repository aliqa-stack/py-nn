import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

X = Variable(torch.Tensor([[2.0], [3.0], [4.0]])) 
Y = Variable(torch.Tensor([[4.0], [6.0], [8.0]])) 

class Neural(nn.Module):
    def __init__(self):
        super(Neural, self).__init__()
        self.out = nn.Linear(1,1)
    def forward(self,x):
        x = self.out(x)
        return x

model = Neural()
Err = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoc in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = Err(y_pred, Y)
    loss.backward()
    optimizer.step()

    if(epoc + 1) % 20 == 0:
        print(f"epoch {epoc + 1}, loss: {loss.item():.4f}")

new_var = Variable(torch.Tensor([[2.0]]))
pred_y = model(new_var)
print("predict (after training)", 4, model(new_var).item())

    