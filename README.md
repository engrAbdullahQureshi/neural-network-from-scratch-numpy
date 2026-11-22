# Neural Network from Scratch in NumPy (Airbnb Price Classes)

A multi-layer perceptron built **without any deep learning framework**: forward propagation, backpropagation, cross-entropy loss and gradient descent are all written by hand in NumPy. It classifies NYC Airbnb listings into four price classes (Budget, Moderate, Premium, Luxury).

## What's inside

| Notebook | Contents |
|---|---|
| `01_eda_preprocessing.ipynb` | EDA on 41,348 listings: class balance, missing-value handling (mode for categorical, median for numeric), one-hot encoding, scaling, correlations |
| `02_mlp_from_scratch.ipynb` | MLP implementation, Sigmoid vs ReLU comparison, gradient-magnitude analysis, gradient-based feature attribution, test evaluation and generalisation analysis |

## Model

```
inputs → Dense(32) → Dense(16) → Dense(4) → softmax
```

- Hand-written `forward_propagation`, `backward_propagation`, `compute_loss` (cross-entropy) and `compute_accuracy`
- Two variants: **Sigmoid** with small random init, and **ReLU** with He initialisation
- Features: neighbourhood group, room type, minimum nights, amenity score, number of reviews, availability

## Findings

**1. Sigmoid vs ReLU.** Tracking average gradient magnitudes per layer shows Sigmoid squashing gradients as they flow backwards (vanishing gradients), so early layers learn slowly. ReLU keeps gradients healthy and trains faster and more stably.

**2. Feature attribution.** Gradients of the output with respect to the inputs identify **amenity score** as the most influential predictor.

**3. A generalisation failure, investigated.** Training accuracy reached ~82% and validation ~81%, but accuracy on the separate test file collapsed to ~13%. Comparing train vs test histograms feature by feature showed that the test distribution differs sharply from training (distribution shift), so patterns learned on training data, especially around amenity score, did not transfer.

Lessons: always evaluate on data that reflects real deployment, compare feature distributions before trusting a model, and verify that the test preprocessing pipeline exactly matches training (encoder columns, scaler, column order), since a drop this large can also come from a preprocessing mismatch.

## Tech

Python, NumPy, Pandas, scikit-learn (preprocessing only), Matplotlib, Seaborn

## Run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run `01_eda_preprocessing.ipynb` first: it writes the encoded dataset and fitted scaler into `data/`, which `02_mlp_from_scratch.ipynb` uses.
