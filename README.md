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





## Author
Yukthi N ·
