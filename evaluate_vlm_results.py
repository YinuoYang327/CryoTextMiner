# evaluate_vlm_results.py
import pandas as pd
import ast
from pathlib import Path

def evaluate_results(results_path="results/all_models_summary.csv"):
    """
    Evaluate VLM predictions for mitochondria detection accuracy.
    Supports both combined (all_models_summary.csv) and single-model result files.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {results_path}")

    print(f" Loading results from: {path}")
    df = pd.read_csv(path)

    if "model" not in df.columns:
        model_name = path.stem.replace("_results", "")
        df["model"] = model_name

    def safe_parse(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]

    df["ground_truth"] = df["ground_truth"].apply(safe_parse)
    df["predictions"] = df["predictions"].apply(safe_parse)


    df["hit_mito"] = df.apply(
        lambda r: any("mitochond" in p.lower() for p in r["predictions"]),
        axis=1,
    )

    summary = (
        df.groupby("model")["hit_mito"]
        .mean()
        .reset_index()
        .rename(columns={"hit_mito": "accuracy_mito"})
    )
    summary["n_samples"] = df.groupby("model")["image_id"].count().values

    print("Evaluation Summary:")
    print(summary.to_string(index=False, justify="center"))

    # save summary
    out_path = Path("results") / "evaluation_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved evaluation summary to {out_path}")

    wrong = df[~df["hit_mito"]][["model", "image_id", "predictions"]]
    if not wrong.empty:
        print("Misclassified examples (did NOT mention mitochondria):")
        print(wrong.head(10).to_string(index=False))
        wrong.to_csv("results/misclassified_examples.csv", index=False)
        print("Saved detailed misclassifications to results/misclassified_examples.csv")
    else:
        print("All images correctly identified!")

    return summary
