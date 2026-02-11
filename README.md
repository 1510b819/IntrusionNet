# IntrusionNet

**Neural Network–Based Intrusion Detection using NSL-KDD**

IntrusionNet is a PyTorch-based intrusion detection system that performs **multi-class classification** on the NSL-KDD dataset and evaluates results using **five high-level attack categories**.

---

## Overview

* Fully connected neural network (MLP)
* Trained on **NSL-KDD**
* One-hot encoding + feature normalization
* Multi-class prediction of network intrusions
* Evaluation with confusion matrix & classification report

---

## Repository Structure

```
IntrusionNet/
├── data/
│   └── nslkdd/
│       ├── KDDTrain+.txt
│       └── KDDTest+.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── model.pth
│   ├── confusion_matrix.png
│   └── classification_report.csv
└── README.md
```

---

## Dataset (NSL-KDD)

NSL-KDD is an improved version of the KDD’99 dataset designed for network intrusion detection research.

**Preprocessing**

* Removes `difficulty` column
* One-hot encodes categorical features:

  * `protocol_type`
  * `service`
  * `flag`
* Standardizes numerical features
* Encodes labels using `LabelEncoder`

---

## Model Architecture

Implemented in `model.py`:

```
Input → Linear(128) → ReLU
      → Linear(64)  → ReLU
      → Linear(num_classes)
```

---

## Training

Run:

```bash
python train.py
```

**Training Details**

* Optimizer: Adam
* Loss: CrossEntropyLoss
* Epochs: 25
* Batch size: 64
* Device: CPU

Model is saved to:

```
results/model.pth
```

---

## Evaluation

Run:

```bash
python evaluate.py
```

### Attack Category Mapping

Fine-grained attack labels are mapped to:

| Category | Description        |
| -------- | ------------------ |
| Normal   | Benign traffic     |
| DoS      | Denial of Service  |
| Probe    | Scanning & probing |
| R2L      | Remote to Local    |
| U2R      | User to Root       |

### Outputs

* Confusion matrix (`confusion_matrix.png`)
* Classification report (`classification_report.csv`)

---

## Requirements

```txt
python >= 3.8
torch
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

---

## Future Work

* GPU support
* Regularization & early stopping
* Deep architectures (CNN / LSTM)
* Binary vs multi-class comparison
* Explainability (SHAP, feature importance)

---

## References

* Tavallaee et al., *A Detailed Analysis of the KDD CUP 99 Data Set*, 2009
* NSL-KDD Dataset: [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)

---
