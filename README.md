# **Adversarial ML attack on MNIST Dataset**

This notebook explores various adversarial attacks on machine learning models, utilizing the Adversarial Robustness Toolbox (ART). **I would highly recommend going through the Google Colab notebook as everything has been written clearly and demonstrated properly 📝.**

## **Introduction**
Machine learning (ML) models are vulnerable to adversarial attacks, where adversaries aim to manipulate model behavior to produce incorrect outputs. ART is a Python framework offering tools to evaluate and defend ML models against such attacks. This project demonstrates different attack types and their implementation in ART, as well as examples of poisoning data to corrupt model training. We have used MNIST dataset to work on this project.

### **Types of Attacks**

**1. Evasion Attacks**

* **Definition:** Evasion attacks typically work by perturbing input data to cause a trained model to misclassify it. Evasion is done after training and during inference, i.e. when models are already deployed in production. Adversaries perform evasion attacks to avoid detection by AI systems. As an example, adversaries might run an evasion attack to cause the victim model to miss phishing emails. Evasion attacks might require access to the victim model.

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

* **Definition:** Extraction is an attack where an adversary attempts to build a model that is similar or identical to a victim model. In simple words, extraction is the attempt of copying or stealing a machine learning model. Extraction attacks typically require access to the original model, as well as to data that is similar or identical to the data originally used to train the victim model.

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

* **Definition:** Inference attacks generally aim at reconstructing a part or the entirety of the dataset that was used to train the victim model. Adversaries can use inference attacks to reconstruct entire training samples, separate features, or determine if a sample has been used to train the victim model. Inference attacks typically require access to the victim model. In some cases, attackers might also need to have access to some portion of the data used to train the model.

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

* **Definition:** Poisoning attacks aim to perturb training data to corrupt the victim model during training. Poisoned data contains features (called a backdoor) that trigger the desired output in a trained model. Essentially, the perturbed features cause the model to overfit to them. As a very simple example (which we’ll have a look at in code below), an attacker could poison the digits in the MNIST dataset so that the victim model classifies all digits as 9s. Poisoning attacks require access to the training data of a model before the actual training occurs.

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
