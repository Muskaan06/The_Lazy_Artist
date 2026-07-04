# The Lazy Artist — Shortcut Learning in CNNs

A study on how image classification models exploit spurious features (like color) instead of learning meaningful ones (like shape), and methods to detect, visualize, and mitigate this behavior.

Built as part of the **Precog (IIIT-H) research assignment**.

## Overview

A 3-layer CNN was trained on **ColoredMNIST** — a modified MNIST dataset where each digit is assigned a specific color. The model achieves $\sim 100\%$ train accuracy but drops to $\sim 3\%$ on test data where colors are shuffled. The rest of the project investigates why this happens and what can be done about it.

## Tasks

### Task 0 — Biased Canvas (Dataset Creation)

Created a custom `ColoredMNIST` dataset from standard MNIST ($60{,}000$ samples):

- **Train split** ($\sim 57{,}000$, majority): Each digit gets a fixed color (e.g., $0 \rightarrow$ Orange, $1 \rightarrow$ Red, $2 \rightarrow$ Blue, ...) with a clean black background.
- **Test split** ($\sim 3{,}000$, minority): Digits get a *different* color than their assigned one, with a noisy textured background.

The majority fraction is controlled by a parameter $\rho = 0.95$. Binary masks $M \in \{0, 1\}^{28 \times 28}$ were generated from the grayscale images to separate digit pixels from background, used later in Task 4.

### Task 1 — The Cheater (Training the Biased Model)

Architecture: 3 conv layers (encoder $\rightarrow$ middle $\rightarrow$ decoder) with BatchNorm + ReLU, followed by a 2-layer FC classifier. Input: $\mathbb{R}^{3 \times 28 \times 28}$, Output: $\mathbb{R}^{10}$.

- **Loss:** Cross-Entropy $\mathcal{L}_{CE} = -\sum_{c} y_c \log(\hat{y}_c)$
- **Optimizer:** Adam with $\text{lr} = 10^{-3}$
- **Epochs:** 20

| Split | Loss | Accuracy |
|-------|------|----------|
| Train | $0.00$ | $100\%$ |
| Val | $0.0002$ | $99.99\%$ |
| Test | $47.35$ | $3.27\%$ |

**Observation:** The model learned to classify digits entirely by color. When colors were changed at test time, it failed completely.

### Task 2 — The Prober (Neuron Analysis via Activation Maximization)

Used activation maximization to visualize what the encoder's 16 feature maps respond to. For a given channel $c$, a random input $x$ is iteratively updated to maximize the mean activation:

$$x^* = \arg\max_{x} \; \frac{1}{H \cdot W} \sum_{i,j} A_c(x)_{i,j}$$

where $A_c(x)$ is the feature map of channel $c$. The update rule is:

$$x_{t+1} = x_t + \eta \cdot \nabla_x \bar{A}_c(x_t)$$

with $\eta = 0.05$ and $50{,}000$ iterations, clamping $x \in [-1, 1]$.

Two setups were tried:
- **Without regularization:** Noisy results, but several channels clearly responded to specific colors (uniform yellow, green, blue, cyan). Some showed stripe/edge-like patterns.
- **With regularization** ($L_2$ decay + Gaussian blur): Cleaner outputs. Multiple channels showed solid uniform colors, confirming the model's neurons are primarily encoding color rather than shape.

### Task 3 — The Interrogation (Grad-CAM)

Implemented Grad-CAM from scratch. For target class $c$ and feature maps $A^k$ from the encoder:

1. Compute importance weights:

$$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{ij}}$$

2. Compute the class activation map:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$

**Observation:** On biased (train) images, the heatmap focuses on the digit area — but since color and shape overlap spatially, Grad-CAM alone can't distinguish which feature is being used. Combined with Task 2's results, it supports the conclusion that color is the dominant signal. On conflicting (test) images, the heatmap appears smeared due to the noisy background and mismatched colors.

### Task 4 — The Intervention (Retraining to Remove Bias)

Two gradient-based retraining methods were applied without modifying the dataset or converting inputs to grayscale.

**Method 1 — Right for Right Reasons:**

Penalizes the gradient magnitude on background pixels using the binary mask $M$:

$$\mathcal{L}_{\text{penalty}} = \frac{1}{N} \sum_{n} \left\| \frac{\partial s_c}{\partial x_n} \odot M_n \right\|_2^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \lambda_1 \cdot \mathcal{L}_{\text{penalty}}$$

with $\lambda_1 = 100$, $\lambda_2 = 10^{-4}$ (weight decay).

| Split | Loss | Accuracy |
|-------|------|----------|
| Test | $40.58$ | $5.37\%$ |

Did not help much — since color and shape share the same spatial region (the digit), penalizing background gradients doesn't address the actual shortcut.

**Method 2 — Color Deviation Penalty:**

Instead of penalizing all gradients, this penalizes the *deviation* between R, G, B channel gradients on digit pixels. Let $g = \nabla_x s_c \odot M$ be the masked gradient and $\bar{g}$ be its per-channel mean:

$$\mathcal{L}_{\text{color}} = \frac{1}{N} \sum_n \left\| g_n - \bar{g}_n \right\|_2^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{CE} + \lambda_1 \cdot \mathcal{L}_{\text{color}}$$

with $\lambda_1 = 0.5$. The model is allowed to look at the digit but forced to treat all color channels equally.

| Split | Loss | Accuracy |
|-------|------|----------|
| Test | $9.32$ | $17.67\%$ |

Better than Method 1 but still not satisfactory. The fundamental difficulty is that color and shape are spatially entangled in this dataset.

### Task 5 — The Invisible Cloak (Targeted Adversarial Attack)

Used PGD (Projected Gradient Descent) to craft an adversarial example: take an image of digit $7$ and make the retrained model classify it as $3$.

At each step $t$:

$$x_{t+1} = \Pi_{x + \mathcal{S}} \left( x_t - \alpha \cdot \text{sign}(\nabla_x \mathcal{L}(f(x_t), y_{\text{target}})) \right)$$

where $\Pi_{x + \mathcal{S}}$ projects back into the $\ell_\infty$ ball of radius $\varepsilon = 0.1$ around the original image, and $\alpha = 0.001$ with $300$ steps.

**Observation:** The model's $P(y=3)$ increased slightly but remained low-confidence, showing some robustness of the retrained model against targeted attacks.

### Task 6 — Decomposition using Sparse Autoencoder

Trained a Sparse Autoencoder (SAE) on the classifier layer's activations to decompose dense polysemantic representations into sparse monosemantic features.

The SAE maps $h \in \mathbb{R}^{64}$ to a latent $z \in \mathbb{R}^{32768}$:

$$z = \text{ReLU}(W_e h + b_e), \quad \hat{h} = W_d z + b_d$$

Training objective:

$$\mathcal{L}_{\text{SAE}} = \| h - \hat{h} \|_2^2 + \lambda_s \cdot \| z \|_1$$

with $\lambda_s = 0.001$, optimized using Adam ($\text{lr} = 10^{-3}$, weight decay $= 0.001$).

**Feature identification:**
- **Color features:** SAE latent neurons that activate consistently across samples of the same color but different digits.
- **Shape features:** Neurons that activate across samples of the same digit but different colors.

**Validation via activation patching:**
- Zeroing out identified color features caused a noticeable drop in accuracy, confirming those neurons encode color.
- Zeroing out shape features did not decrease accuracy, likely because the model never properly learned shape in the first place.
- $\sim 20$ polysemantic neurons were found that respond to both color and shape.

## Key Takeaways

- Models will exploit the easiest available signal (color here) even when it doesn't generalize.
- Activation maximization and Grad-CAM together provide evidence of what features the model relies on.
- Gradient-based retraining helps but is limited when the spurious feature spatially overlaps with the real one.
- Sparse autoencoders can decompose model internals into interpretable features and validate them via targeted perturbation.

## Setup

```bash
pip install torch torchvision matplotlib numpy pillow
```

Trained on Google Colab with GPU. The notebook expects a Google Drive mount for saving/loading model checkpoints and logs.

## References

- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Right for the Right Reasons](https://arxiv.org/abs/1703.03717)
- Sparse Autoencoders for mechanistic interpretability
