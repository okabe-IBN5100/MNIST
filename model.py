import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(8, 16, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(288, 10),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        x1 = self.conv(X)
        x2 = self.linear(torch.flatten(x1, start_dim=1))
        return x2
        
if __name__ == "__main__":
    model = Classifier().to('cuda')

    x = torch.rand((4, 1, 28, 28), device='cuda')
    print(x.shape)
    
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))