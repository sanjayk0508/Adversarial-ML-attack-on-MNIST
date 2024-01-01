# **Adversarial ML attack on MNIST Dataset**

This project explores various adversarial attacks on machine learning models, utilizing the Adversarial Robustness Toolbox (ART).

## **Introduction**
Machine learning (ML) models are vulnerable to adversarial attacks, where adversaries aim to manipulate model behavior to produce incorrect outputs. ART is a Python framework offering tools to evaluate and defend ML models against such attacks. This project demonstrates different attack types and their implementation in ART, as well as examples of poisoning data to corrupt model training.

### **Types of Attacks**

**1. Evasion Attacks**

* **Definition:** Attacker perturbs input data at inference time to cause misclassification.

* **Fast Gradient Method - FGM:**

```python
import art
import tensorflow as tf

# Load model and classifier
model = tf.keras.models.load_model(...)
classifier = art.estimators.classification.TensorFlowV2Classifier(model=model, ...)

# Create FGM attack
attack = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.3)

# Generate adversarial examples
x_test_adv = attack.generate(x_test)
```

**2. Extraction Attacks**

* **Definition:** Attacker aims to copy or steal a victim model.

* **CopycatCNN:**

```python
import art
import torch

# Load victim model (PyTorch)
victim_model = torch.load(...)

# Create CopycatCNN attack
attack = art.attacks.extraction.CopycatCNN(victim_model)

# Extract model
extracted_model = attack.extract()
```

**3. Inference Attacks**

* **Definition:** Attacker attempts to obtain knowledge about the victim model's training data.

* **Model Inversion Attack - MIFace:**

```python
import art
import numpy as np

# Load model and classifier
model = tf.keras.models.load_model(...)
classifier = art.estimators.classification.TensorFlowV2Classifier(model=model, ...)

# Create MIFace attack
attack = art.attacks.inference.MIFace(estimator=classifier)

# Infer training data properties
reconstructed_images = attack.infer(x_test, y_test)
```

**4. Poisoning Attacks**

* **Definition:** Attacker perturbs training data to corrupt model behavior during training.

* **Backdoor Attack:**

```python
import art
import tensorflow as tf

# ... (Code for loading data and defining attack, as shown previously)

# Poison training data
x_train_poisoned, y_train_poisoned = backdoor.poison(x_train[:5], target)

# Retrain model with poisoned data
model = tf.keras.models.Sequential(...)  # Define model architecture
model.compile(...)
model.fit(x_train_poisoned, y_train_poisoned, ...)
```
