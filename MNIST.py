import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn

batch_size = 64
lr=0.1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)

model = nn.Linear(784, 392)
model_2 = nn.Linear(392, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=lr)
relu = nn.ReLU()

for batch_idx, (data, targets) in enumerate(train_loader):
    data = data.view((-1, 28*28))
    print(data.size())
    outputs = model(data)
    outputs = relu(outputs)
    outputs = model_2(outputs)

    log_softmax = F.log_softmax(outputs, dim=1)
    loss = F.nll_loss(log_softmax, targets)
    # print("\nLoss shape: {}".format(loss), end="")
    
    loss.backward()
    with torch.no_grad():
        optimizer.step()
        optimizer_2.step()
    optimizer.zero_grad()
    optimizer_2.zero_grad()


import matplotlib.pyplot as plt
for i in range(100):

    batch_idx, (data, target) = next(enumerate(test_loader))
    data = data.view((-1, 28*28))

    outputs = model(data)
    outputs = model_2(outputs)
    softmax = F.softmax(outputs, dim=1)
    pred = softmax.argmax(dim=1, keepdim=True)

    plt.imshow(data[0].view(28, 28), cmap="gray")
    plt.title("Predicted class {}".format(pred[0]))
    plt.show()