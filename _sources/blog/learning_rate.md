---
title: "How Learning Rate Shapes the Journey of a Neural Network"
author: "Joanna Reyda D. Santos"
---

# How Learning Rate Shapes the Journey of a Neural Network
### A Deep Learning Blog Post by Joanna Reyda D. Santos

---

## Introduction

Training a neural network is similar to teaching a student.  
If the pace is too slow, learning becomes unproductive.  
If the pace is too fast, the student becomes overwhelmed.

In deep learning, this “pace of learning” is controlled by a single but influential hyperparameter: the **learning rate**.

---

# What is Learning Rate?

A neural network updates its weights through gradient descent:

\[
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)
\]

Where:

- \( \theta \): model weights  
- \( L \): loss  
- \( \nabla_\theta L \): gradient  
- \( \eta \): learning rate  

### Small Learning Rate (0.0001)
- learns slowly  
- stable but may get stuck in local minima

### Ideal Learning Rate (0.001)
- balanced  
- allows smooth convergence  
- often yields the best overall accuracy

### Large Learning Rate (0.01)
- unstable  
- may overshoot minima  
- risk of divergence

---

## Experiment Overview

To visualize how learning rate affects model behavior, I trained a simple CNN on the **MNIST** dataset using three different learning rates:

| Learning Rate | Expected Behavior |
|--------------|------------------|
| **0.0001** | slow learning |
| **0.001** | optimal |
| **0.01** | unstable |

**Dataset:** MNIST (28×28 grayscale images)  
**Model:** Simple CNN  
**Epochs:** 5  
**Optimizer:** Adam  

---

# CNN Model Code

```python
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

# Running the Experiments

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

# Loss Curves
plt.subplot(1,2,1)
for lr in lrs:
    plt.plot(results[lr][0], label=f"LR={lr}")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy Curves
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

# Results & Interpretation

### Learning Rate = 0.001 (Optimal)
- The training curve is stable and smooth.
- Achieves the highest accuracy (~98%).
- Converges efficiently.

### Learning Rate = 0.0001 (Too Small)
- Training is very slow.
- Loss decreases gradually.
- Needs more epochs to reach high accuracy.

### Learning Rate = 0.01 (Too Large)
- Highly unstable.
- Accuracy oscillates and may not improve.
- Sometimes diverges entirely.

---

# Reflections

From this experiment, I learned that even small changes in learning rate dramatically affect:

1. **Training stability**  
2. **Speed of convergence**  
3. **Final performance**

A poorly chosen learning rate cannot be compensated by simply training longer.  
Because of this, tuning the learning rate is often the **first and most important** hyperparameter decision.

---

# Conclusion

The learning rate is one of the most influential hyperparameters in deep learning.  
A well-chosen learning rate enables efficient and stable learning, while a poor choice can prevent learning altogether.

This experiment demonstrated how different learning rates affect the behavior of a CNN model on MNIST and highlighted why selecting the right value is crucial for model success.

---

# References

- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html  
- **Deep Learning (Goodfellow, Bengio, Courville):** https://www.deeplearningbook.org/  
- **MNIST Dataset Paper:** http://yann.lecun.com/exdb/mnist/  
- **DS413 Lecture Notes** (Instructor-provided materials)

