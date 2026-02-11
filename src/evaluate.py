import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import IntrusionNet

# Attack categories mapping
CATEGORY_MAP = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    # Probe attacks
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R"
}

def map_labels(label_encoder, labels):
    """Map original attack labels to categories."""
    label_names = label_encoder.inverse_transform(labels)
    mapped = [CATEGORY_MAP.get(name, "Unknown") for name in label_names]
    return mapped

def evaluate():
    device = torch.device("cpu")

    # Paths
    train_path = "../data/nslkdd/KDDTrain+.txt"
    test_path = "../data/nslkdd/KDDTest+.txt"
    model_path = Path("../results/model.pth")
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Load test data
    _, test_loader, input_dim, num_classes = get_dataloaders(
        train_path, test_path, batch_size=64
    )

    # Load model
    model = IntrusionNet(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    # Convert to attack categories
    from dataset import load_nslkdd  # import function to get label_encoder
    _, _, _, _, label_count = load_nslkdd(train_path, test_path)
    label_encoder = get_dataloaders(train_path, test_path)[0].dataset  # placeholder

    # For simplicity, reuse the encoder from get_dataloaders
    # But we just need inverse_transform
    # Actually, easiest is: re-load encoder from dataset.py
    import dataset
    _, y_train, _, y_test, le_count = dataset.load_nslkdd(train_path, test_path)
    # Create label encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(pd.read_csv(train_path, names=dataset.NSL_KDD_COLUMNS)["label"].tolist() +
           pd.read_csv(test_path, names=dataset.NSL_KDD_COLUMNS)["label"].tolist())

    all_labels_mapped = map_labels(le, all_labels)
    all_preds_mapped = map_labels(le, all_preds)

    # Classification report
    print("Classification Report (categories):")
    print(classification_report(all_labels_mapped, all_preds_mapped, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels_mapped, all_preds_mapped, labels=["Normal","DoS","Probe","R2L","U2R"])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Normal","DoS","Probe","R2L","U2R"],
                yticklabels=["Normal","DoS","Probe","R2L","U2R"], cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (5 Categories)")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.show()

    # Save metrics to CSV
    report_dict = classification_report(all_labels_mapped, all_preds_mapped, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(results_dir / "classification_report.csv")
    print(f"Saved confusion matrix and metrics to {results_dir}/")

if __name__ == "__main__":
    evaluate()