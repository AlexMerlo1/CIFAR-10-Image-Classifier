# Project Outline 

## Goal  
Predict the class of an image in the CIFAR-10 with at least 75% test accuracy.

## Dataset: Cifar-10
60000 32x32 RGB images in 10 classes

## Core Layers (From Scratch)
- Fully Connect Layer
- Convolutional layer
- ReLU activation
- Softmax

## Baseline evaluation
- Simple CNN  
- SGD (No momentum)  
- Fixed learning rate  
- No regularization  
- No augmentation  



## Optimization Techniques
- Gradient Descent with momentum
- Adam
- Cosine learning rate decay
- Data Augmentation (Flipping & Cropping)

## Evaluations
- Training Accuracy per epoch
- Test accuracy per epoch
- Cost per iteration

## Analysis (Ablation Study)
### Goal:
Evaluate the timpact of different optimization & regularization techniques
### For Each Experiment
- Baseline architecture is fixed
- One factor changed at a time
- Trained over 5-10 epochs for observation
- Training & Test accuracy recorded per epoch

### Learning Rate Comparisons

**Baseline:** Constant learning rate  
$$
\eta_t = \eta_0
$$
**Step Decay:** Reduce learning rate by fixed factor at predefined intervals  
$$
\eta_t = \eta_0 \cdot \gamma^{\left\lfloor \frac{t}{s} \right\rfloor}
$$
**Cosine Learning Rate Decay:** Follows cosine curve from initial value down to near 0  
$$
\eta_t = \frac{\eta_0}{2}\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$
### Regularization
**L2:** Add additional term to loss function 
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}\big(\hat{y}^{(i)},y^{(i)} \big)
+ \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$
**Dropout:** Randomly stop a select percent of nodes from each layer from updating paramaters.  

Evaluate different dropout rates:
- p = 0.0 (baseline)
- p = 0.2
- p = 0.5
- p = 0.7

### Effect of L2 Regularization Strength (λ)

Evaluate different values of λ to measure its impact on overfitting.

- λ = 0 (baseline)
- λ = 1e-4
- λ = 1e-3
- λ = 1e-2



Plots:
- Accuracy vs Epoch (for each λ)
- Train vs test accuracy gap comparison

### Effect of Optimization Algorithm

- SGD (baseline)
- SGD + Momentum
- Adam

Plots:
- Training loss vs iteration
- Test accuracy vs epoch

### Effect of Adam Hyperparameters (β₁, β₂)

Values Tested:
- β₁ ∈ {0.8, 0.9, 0.95}
- β₂ ∈ {0.99, 0.999}
