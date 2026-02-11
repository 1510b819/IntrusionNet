# **IntrusionNet**

**IntrusionNet** is a PyTorch-based neural network for detecting and classifying network intrusions using the **NSL-KDD dataset**. This project preprocesses network traffic data, trains a neural network, and evaluates its performance on multi-class attack detection.

---

## **Features**

* Preprocessing of NSL-KDD dataset (encoding categorical features, normalization)
* Custom PyTorch Dataset and DataLoader for efficient training
* Multi-class attack classification
* Evaluation metrics for model performance
* Fully customizable neural network architecture

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/1510b819/IntrusionNet.git
cd IntrusionNet
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## **Usage**

1. Prepare the **NSL-KDD dataset** files (`KDDTrain+.txt`, `KDDTest+.txt`).
2. Update file paths in your script or notebook.
3. Load data and create dataloaders:

```python
from dataset import get_dataloaders

train_loader, test_loader, input_size, num_classes = get_dataloaders(
    "KDDTrain+.txt", "KDDTest+.txt", batch_size=64
)
```

4. Define your neural network, train, and evaluate:

```python
from model import MyNeuralNet  # Example: replace with your model
model = MyNeuralNet(input_size, num_classes)
```

5. Train the model and monitor metrics such as accuracy and loss.

---

## **Dataset**

**NSL-KDD**: A benchmark dataset for network intrusion detection. It improves on the original KDD’99 dataset by removing redundant records and providing a more balanced distribution of attacks.

Columns include network connection features such as:

* `duration`, `protocol_type`, `service`, `src_bytes`, `dst_bytes`, etc.
* `label` for attack type or normal traffic

---

## **Contributing**

Contributions are welcome! Please:

1. Fork the repo
2. Create a branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## **License**

This project is licensed under the MIT License.

