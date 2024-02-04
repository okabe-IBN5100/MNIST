import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(784, 128, bias=True),
                                     nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(128, 10),
                                     nn.Softmax(dim=1))
        
    def forward(self, X):
        l1 = self.linear1(X)
        l2 = self.linear2(l1)
        return l2
        
if __name__ == "__main__":
    model = Classifier().to('cuda')
    t = torch.randn(1, 784, device='cuda')
    print(model(t).sum(dim=1))