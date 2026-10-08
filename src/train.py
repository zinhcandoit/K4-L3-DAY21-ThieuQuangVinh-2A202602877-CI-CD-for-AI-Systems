import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
F1_THRESHOLD = 0.65

# Bonus 1: Ho tro tracking server tu xa (nhu DagsHub) neu co bien moi truong
if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    Ho tro cac tinh nang Bonus:
      - Bonus 1: Remote tracking DagsHub qua bien moi truong
      - Bonus 2: Dieu chinh nguong quyet dinh toi uu F1
      - Bonus 3: Bao cao Precision/Recall va Confusion Matrix chi tiet
      - Bonus 5: Kiem tra do lech du lieu (Data Drift)
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Kiem tra Data Drift tren ty le lop duong (moc chuan Adult: 24.8% ~ 0.248)
    positive_rate = float(y_train.mean())
    if abs(positive_rate - 0.248) > 0.05:
        print(f"⚠️ CANH BAO DATA DRIFT: Ty le lop duong hien tai la {positive_rate:.4f}, lech > 5% so voi moc 0.248!")
    else:
        print(f"✅ Data distribution on dinh: Ty le lop duong = {positive_rate:.4f} (moc: 0.248)")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_metric("positive_rate", positive_rate)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X=X_train, y=y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so (nguong mac dinh 0.5)
        preds = model.predict(X_eval)
        f1    = float(f1_score(y_eval, preds, zero_division=0))
        acc   = float(accuracy_score(y_eval, preds))

        # Bonus 2: Quet nguong xac suat tu 0.1 den 0.9 (buoc 0.05) de tim F1 toi uu
        probs = model.predict_proba(X_eval)[:, 1]
        best_thresh = 0.5
        best_f1 = f1
        for t in np.arange(0.1, 0.95, 0.05):
            t_val = round(float(t), 2)
            t_preds = (probs >= t_val).astype(int)
            t_f1 = float(f1_score(y_eval, t_preds, zero_division=0))
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_thresh = t_val

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_thresh)
        mlflow.log_metric("best_f1_score", best_f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        # TODO 7: In ket qua ra man hinh
        print(f"F1 (threshold 0.5): {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Bonus 2 - Optimal Threshold: {best_thresh:.2f} -> Best F1: {best_f1:.4f}")

        # Bonus 3: Tao bao cao chi tiet Precision / Recall va Confusion Matrix
        os.makedirs("outputs", exist_ok=True)
        cm = confusion_matrix(y_eval, preds)
        cr = classification_report(y_eval, preds, digits=4, zero_division=0)
        detail_text = (
            "=== CONFUSION MATRIX ===\n"
            f"{cm}\n\n"
            "=== CLASSIFICATION REPORT (PRECISION & RECALL) ===\n"
            f"{cr}\n"
            f"F1 (nguong mac dinh 0.5): {f1:.4f}\n"
            f"Accuracy: {acc:.4f}\n"
            f"Bonus 2 - Nguong toi uu: {best_thresh:.2f} (F1 toi uu: {best_f1:.4f})\n"
            f"Bonus 5 - Ty le lop duong: {positive_rate:.4f} (Moc chuan: 0.248)\n"
        )
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_text)

        # TODO 8: Luu metrics ra file outputs/report.json
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump({
                "f1_score": f1,
                "accuracy": acc,
                "best_threshold": best_thresh,
                "best_f1_score": best_f1,
                "positive_rate": positive_rate,
            }, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)