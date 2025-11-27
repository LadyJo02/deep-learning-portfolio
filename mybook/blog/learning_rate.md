---
title: "How Learning Rate Shapes the Journey of a Neural Network"
author: "Joanna Reyda D. Santos"
---

# How Learning Rate Shapes the Journey of a Neural Network
### Deep Learning Blog Post

---

## Introduction

Training a neural network involves gradually adjusting model parameters to minimize error.  
However, the speed at which these adjustments occur is controlled by a critical hyperparameter known as the **learning rate**.

Choosing the right learning rate is essential:
- too slow, and the model learns inefficiently,  
- too fast, and the model becomes unstable.

This blog explores the concept of learning rate through explanation and experimentation.

---

## What is Learning Rate?

During training, model weights are updated using gradient descent:

\[
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)
\]

Where:

- \( \theta \): model weights  
- \( L \): loss function  
- \( \nabla_\theta L \): gradient  
- **\( \eta \)**: learning rate  

### Effects of Different Learning Rates

**Small Learning Rate (0.0001)**
- Very stable  
- Slow progress  
- May not converge in reasonable time  

**Medium Learning Rate (0.001)**
- Typically optimal  
- Smooth and consistent convergence  
- Good accuracy  

**Large Learning Rate (0.01)**
- Unstable  
- Loss may oscillate  
- Training may diverge entirely  

---

## Experiment Overview

To observe the impact of learning rate, a simple CNN was trained on the **MNIST** digit recognition dataset using three learning rates:

| Learning Rate | Expected Behavior |
|---------------|------------------|
| 0.0001 | Slow, stable, low accuracy |
| 0.001 | Ideal, smooth convergence |
| 0.01 | Unstable or diverging |

**Dataset:** MNIST (28×28 grayscale images)  
**Model:** Simple CNN  
**Epochs:** 5  
**Optimizer:** Adam  
**Comparison:** Loss curves and validation accuracy trends  

---

# CNN Model Code

```python
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
```

---

# Dataset Loading

```python
transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

---

# Training Function

```python
def train_model(lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accs = []

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                correct += (preds.argmax(1) == y).sum().item()

        acc = correct / len(test_set)
        val_accs.append(acc)

        print(f"LR={lr} | Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Acc={acc:.4f}")

    return train_losses, val_accs
```

---

# Running the Experiment

```python
lrs = [0.0001, 0.001, 0.01]
results = {}

for lr in lrs:
    print("="*60)
    print(f"Training with LR = {lr}")
    results[lr] = train_model(lr)
```

---

# Plotting the Results

```python
plt.figure(figsize=(14,5))

# Loss
plt.subplot(1,2,1)
for lr in lrs:
    plt.plot(results[lr][0], label=f"LR={lr}")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
for lr in lrs:
    plt.plot(results[lr][1], label=f"LR={lr}")
plt.title("Validation Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
```

---

# Results and Interpretation

### Learning Rate = **0.001** (Best Performance)
- Consistent decrease in loss  
- High validation accuracy (≈ 97–98%)  
- Stable convergence  
- Smooth training curve  

### Learning Rate = **0.0001** (Too Slow)
- Loss decreases very gradually  
- Accuracy improves but remains lower compared to 0.001  
- More epochs would be required to reach optimal performance  

### Learning Rate = **0.01** (Unstable)
- Loss fluctuates or increases  
- Accuracy stagnates or declines  
- Can cause divergence in training  

These observations confirm that the learning rate has a dramatic impact on performance, stability, and convergence speed.

---

# Conclusion

The learning rate is one of the most influential hyperparameters when training deep learning models. This experiment demonstrated that:

- A learning rate that is **too small** slows learning and delays convergence.  
- A learning rate that is **too large** destabilizes training and may prevent convergence entirely.  
- An appropriately chosen learning rate (here, **0.001**) achieves the best balance between speed and stability.

Understanding the behavior of learning rates is essential for effective model tuning and should be among the first hyperparameters evaluated in any deep learning workflow.

---

# References

- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html  
- **Deep Learning (Goodfellow, Bengio, Courville):** https://www.deeplearningbook.org/  
- **MNIST Dataset Paper:** http://yann.lecun.com/exdb/mnist/  
- **DS413 Lecture Notes** (Instructor-provided materials)  

