"""Compare automated title/source coding against the manual "publisher type".

Manual labels: "publisher type" column of data/ri_snap_food_access_documents.csv.
Automated labels: "category" column of data/ri_snap_food_access_title_source_coded.csv.
"Community Organization" (manual) is treated as "Community/Nonprofit Organization".
"Tie" and "Uncoded" predictions count as errors (they match no manual class).

Per-class metrics are one-vs-rest:
    Accuracy  = (TP + TN) / (TP + FN + FP + TN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)
Overall accuracy is correct documents / all documents.

Usage:
    python scripts/evaluate_publisher_type.py

Output, in data/:
    ri_snap_food_access_title_source_evaluation.csv   (per-class metrics)
    ri_snap_food_access_title_source_confusion.csv    (confusion matrix)
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
MANUAL_MAP = {"Community Organization": "Community/Nonprofit Organization"}


def main():
    docs = pd.read_csv(DATA / "ri_snap_food_access_documents.csv", encoding="utf-8-sig")
    auto = pd.read_csv(DATA / "ri_snap_food_access_title_source_coded.csv", encoding="utf-8-sig")
    df = docs.merge(auto, left_on="id", right_on="document_number")
    df["manual"] = df["publisher type"].str.strip().replace(MANUAL_MAP)
    df["auto"] = df["category"]

    classes = sorted(df["manual"].unique())
    pred_labels = classes + sorted(set(df["auto"]) - set(classes))
    n = len(df)

    cm = pd.crosstab(df["manual"], df["auto"]).reindex(index=classes, columns=pred_labels, fill_value=0)
    cm.index.name, cm.columns.name = "manual \\ automated", None

    rows = []
    for c in classes:
        tp = int(((df.manual == c) & (df.auto == c)).sum())
        fn = int(((df.manual == c) & (df.auto != c)).sum())
        fp = int(((df.manual != c) & (df.auto == c)).sum())
        tn = n - tp - fn - fp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"class": c, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "accuracy": (tp + tn) / n, "precision": p, "recall": r, "f1": f1,
                     "support": tp + fn})
    m = pd.DataFrame(rows)
    macro = {"class": "macro avg", **{k: m[k].mean() for k in ["accuracy", "precision", "recall", "f1"]},
             "support": n}
    w = m["support"] / n
    weighted = {"class": "weighted avg", **{k: (m[k] * w).sum() for k in ["accuracy", "precision", "recall", "f1"]},
                "support": n}
    m = pd.concat([m, pd.DataFrame([macro, weighted])], ignore_index=True)

    correct = int((df.manual == df.auto).sum())
    print(f"Overall accuracy: {correct}/{n} = {correct / n:.3f}\n")
    print("Confusion matrix (rows = manual, columns = automated):")
    print(cm.to_string(), "\n")
    pd.set_option("display.width", 200)
    print(m.round(3).to_string(index=False))
    print("\nMisclassified documents:")
    bad = df[df.manual != df.auto][["id", "publisher_author", "manual", "auto"]]
    print(bad.to_string(index=False))

    m.to_csv(DATA / "ri_snap_food_access_title_source_evaluation.csv", index=False)
    cm.to_csv(DATA / "ri_snap_food_access_title_source_confusion.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
