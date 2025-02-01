import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)
x = torch.randn(1, 10).to(device)
y = model(x)
print(y)
