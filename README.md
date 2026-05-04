# ✍️ MNIST Digit Classifier — Neural Network from Scratch

Handwritten digit recognition (0-9) built using a fully 
connected neural network in PyTorch. No CNNs — pure 
dense layers trained from scratch.

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 98.5%+ |
| Epochs | 10 |
| Parameters | ~580,000 |
| Training time | ~3 mins (Colab GPU) |

## Architecture
Input (784)           ← 28×28 flattened image
↓
Dense(512) + ReLU + Dropout(0.2)
↓
Dense(256) + ReLU + Dropout(0.2)
↓
Dense(128) + ReLU
↓
Output(10) + Softmax  ← digits 0-9


## Dataset
MNIST — 70,000 handwritten digit images
- Training: 60,000 images
- Test: 10,000 images
- Image size: 28×28 grayscale
- Classes: 10 (digits 0-9)

## Key findings from confusion matrix
- Digit 1 — easiest (1131/1135 correct)
- Digit 5 — hardest (874/892 correct)
- Most confused pairs: 9↔4, 2↔3, 8↔5
- Same digits humans mix up when handwriting is messy

## What's different vs previous projects
- First project using PyTorch (not sklearn)
- Written training loop — not just model.fit()
- Backpropagation happening explicitly
- GPU acceleration
- Dropout regularisation

## New concepts
- Neural network architecture design
- CrossEntropyLoss + Adam optimizer
- Dropout — preventing overfitting
- Softmax — converting outputs to probabilities
- Confusion matrix for multiclass problems

## Tech Stack
Python · PyTorch · torchvision
NumPy · Matplotlib · Seaborn · Sklearn

## Files
- `mnist_classifier.ipynb` — full notebook
- `mnist_samples.png` — sample digit images
- `mnist_training.png` — accuracy + loss curves
- `mnist_confusion.png` — confusion matrix
- `mnist_predictions.png` — predictions on test images


<img width="1430" height="877" alt="image" src="https://github.com/user-attachments/assets/a598ff2b-c595-404a-817f-357235b32a8e" />


# ✍️ MNIST Digit Classifier — Dense Network + CNN Comparison

Handwritten digit recognition built TWO ways:
1. Fully connected Dense Network (from scratch)
2. CNN (Convolutional Neural Network)
Both trained in PyTorch and compared side by side.

## Results
| Model | Test Accuracy | Parameters | Best for |
|-------|-------------|------------|---------|
| Dense Network | 98.5% | ~580,000 | Simple, fast |
| **CNN** | **99.3%** | ~820,000 | Spatial patterns |
| **CNN wins by** | **+0.8%** | | |

## Why CNN beats Dense on images
Dense network sees pixels as unrelated numbers.
CNN understands spatial relationships — neighbouring
pixels form edges, edges form shapes, shapes form digits.

## Architectures

### Dense Network

Input (784) → Dense(512)+ReLU+Dropout →
Dense(256)+ReLU+Dropout → Dense(128)+ReLU → Output(10)

### CNN
Input (28×28×1)
↓
Conv2d(32)+ReLU → MaxPool    (28×28 → 14×14)
↓
Conv2d(64)+ReLU → MaxPool    (14×14 → 7×7)
↓
Conv2d(128)+ReLU             (7×7×128)
↓
Flatten → Dense(256)+ReLU+Dropout → Output(10)

## Dataset
MNIST — 70,000 handwritten digit images
- Training: 60,000 · Test: 10,000
- Image size: 28×28 grayscale · Classes: 10 (0-9)

## Key findings
- CNN crosses 99% from epoch 3 onwards
- Dense plateaus at ~98.5% after epoch 6
- Hardest digit: 5 (confused with 3 and 8)
- Easiest digit: 1 (most distinct shape)
- Same pairs humans confuse: 9↔4, 2↔3, 8↔5

## New concepts (Week 5 + 6)
- PyTorch nn.Module — base class for all neural nets
- DataLoader — mini-batch training (64 images/batch)
- Training loop — forward → loss → backward → update
- Dropout — randomly disable neurons, prevent overfitting
- CrossEntropyLoss — loss for multiclass problems
- Adam optimizer — adaptive learning rates
- Conv2d — sliding filter detects spatial patterns
- MaxPool2d — shrink feature maps, keep strongest signal
- Feature maps — where each pattern appears in image
- model.train() vs model.eval() — toggle dropout


## Author
Yukthi N ·
