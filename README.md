# The Lazy Artist — Shortcut Learning in CNNs

A study on how image classification models exploit spurious features (like color) instead of learning meaningful ones (like shape), and methods to detect, visualize, and mitigate this behavior.

Built as part of the **Precog (IIIT-H) research assignment**.

## Update
 
A third retraining method has been added to Task 4, inspired by [Learning Not to Learn: Training Deep Neural Networks with Biased Data (Kim et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Learning_Not_to_Learn_Training_Deep_Neural_Networks_With_Biased_CVPR_2019_paper.pdf). Unlike Methods 1 and 2 which operate on input gradients, this approach trains a separate bias prediction network alongside the main model and uses adversarial training to force the feature extractor to unlearn color information entirely.
 
The training alternates between two phases:
- **Phase 1 (Classification + Entropy Minimization):** The main model is trained to classify digits while simultaneously minimizing the negative conditional entropy of the bias predictor's output. This pushes the bias predictor's predictions toward a uniform distribution, meaning the features contain no useful color information.
- **Phase 2 (Gradient Reversal):** The bias predictor is trained to predict color from the feature representation, but the gradients flowing back into the feature extractor are *reversed*. This forces the feature extractor to actively remove color information from its representations.
At the end of training, the bias predictor fails to predict color — not because it is poorly trained, but because the feature extractor has successfully unlearned the color bias.
 
> **Note:** This method is still under development. Training and hyperparameter tuning are in progress, and results are awaited.

## Overview

A 3-layer CNN was trained on **ColoredMNIST** — a modified MNIST dataset where each digit is assigned a specific color. The model achieves $\sim 100\%$ train accuracy but drops to $\sim 3\%$ on test data where colors are shuffled. The rest of the project investigates why this happens and what can be done about it.

## Tasks

### Task 0 — Biased Canvas (Dataset Creation)

Created a custom `ColoredMNIST` dataset from standard MNIST ($60{,}000$ samples):

- **Train split (the 'Easy' set):** Each digit gets a fixed color (e.g., $0 \rightarrow$ Orange, $1 \rightarrow$ Red, $2 \rightarrow$ Blue, ...) with a clean black background.
- **Test split (the 'Hard' set):** Digits get a *different* color than their assigned one, with a noisy textured background.

Binary masks were generated from the grayscale images to separate digit pixels from background, used later in Task 4.

### Task 1 — The Cheater (Training the Biased Model)

Architecture: 3 conv layers (encoder $\rightarrow$ middle $\rightarrow$ decoder) with BatchNorm + ReLU, followed by a 2-layer FC classifier. Input: 3-channel RGB $28 \times 28$, Output: 10 classes. Flattened decoder features ($3136$-dim) are mapped to a $64$-dim space and then to $10$ output classes.

- **Loss:** Cross-Entropy
- **Optimizer:** Adam with lr $= 0.001$
- **Epochs:** 20

| Split | Loss | Accuracy |
|-------|------|----------|
| Train | $0.00$ | $100\%$ |
| Val | $0.0002$ | $99.99\%$ |
| Test | $47.35$ | $3.27\%$ |

**Observation:** The model learned to classify digits entirely by color. When colors were changed at test time, it failed completely.

### Task 2 — The Prober (Neuron Analysis via Activation Maximization)

Used activation maximization to visualize what the encoder's 16 feature maps respond to. A random input image is iteratively updated (gradient ascent on the target channel's mean activation) over $50{,}000$ iterations to reveal the patterns that channel is most responsive to.

Two setups were tried:
- **Without regularization:** Noisy results, but several channels clearly responded to specific colors (uniform yellow, green, blue, cyan). Some showed stripe/edge-like patterns.
- **With regularization** ($L_2$ decay + Gaussian blur): Cleaner outputs. Multiple channels (0, 1, 6, 11, 14) showed solid uniform colors, confirming the model's neurons are primarily encoding color rather than shape.

### Task 3 — The Interrogation (Grad-CAM)

Implemented Grad-CAM from scratch to highlight which spatial regions influenced the model's prediction. The method computes the gradient of the target class score with respect to the encoder's feature maps, uses these gradients as importance weights, and produces a weighted combination passed through ReLU to generate the heatmap.

**Observation:** On biased (train) images, the heatmap focuses on the digit area — but since color and shape overlap spatially, Grad-CAM alone can't distinguish which feature is being used. Combined with Task 2's results, it supports the conclusion that color is the dominant signal. On conflicting (test) images, the heatmap appears smeared due to the noisy background and mismatched colors.

### Task 4 — The Intervention (Retraining to Remove Bias)

Two gradient-based retraining methods were applied without modifying the dataset or converting inputs to grayscale.

**Method 1 — Right for Right Reasons:**

Penalizes the total gradient magnitude on background pixels using the binary mask. The total loss is: `classification_loss + λ × gradient_penalty`, with $\lambda = 100$.

| Split | Loss | Accuracy |
|-------|------|----------|
| Test | $40.58$ | $5.37\%$ |

Did not help much — since color and shape share the same spatial region (the digit), penalizing background gradients doesn't address the actual shortcut. This method was designed for spatially separable confounds (like Decoy-MNIST with corner patches), not for cases where the spurious feature overlaps with the real one.

**Method 2 — Color Deviation Penalty:**

Instead of penalizing all gradients, this penalizes the *deviation* between R, G, B channel gradients on digit pixels. The model is allowed to look at the digit but forced to treat all color channels equally. The total loss is: `cross_entropy_loss + λ × color_deviation_penalty`, with $\lambda = 0.5$.

| Split | Loss | Accuracy |
|-------|------|----------|
| Test | $9.32$ | $17.67\%$ |

Better than Method 1 but still not satisfactory. The fundamental difficulty is that color and shape are spatially entangled in this dataset.

### Task 5 — The Invisible Cloak (Targeted Adversarial Attack)

Used PGD (Projected Gradient Descent) to craft an adversarial example: take an image of digit $7$ and try to make the retrained model classify it as $3$. At each step, the image is perturbed in the direction of the gradient's sign by a step size $\alpha = 0.001$, then clipped back to within $\varepsilon = 0.05$ of the original image. This is repeated for $\sim 200$–$300$ iterations.

**Observation:** The model's probability of choosing $3$ increased slightly but remained low-confidence, showing that the retrained model was reluctant to be fooled.

### Task 6 — Decomposition using Sparse Autoencoder

Trained a Sparse Autoencoder (SAE) on the classifier layer's activations (input dim $= 64$, hidden dim $= 32768$) to decompose dense polysemantic representations into sparse monosemantic features. The SAE is a simple 2-layer encoder-decoder trained with MSE reconstruction loss + $L_1$ sparsity penalty ($\lambda = 0.001$), using Adam (lr $= 10^{-3}$, weight decay $= 0.001$).

**Feature identification:**
- **Color features:** Found by checking which SAE latent neurons activate consistently across samples of the same color but different digits.
- **Shape features:** Found by checking which neurons activate across samples of the same digit but different colors.

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
