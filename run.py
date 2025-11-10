import pandas as pd
from pathlib import Path
from llm.openai_client import analyze_image_openai
from llm.gemini_client import analyze_image_gemini
from llm.claude_client import analyze_image_claude

def run_all_models(openai_client, gemini_model, claude_client, prompt_path):
    df = pd.read_csv("demo_dataset/annotations.csv")
    prompt = open(prompt_path).read()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    models = {
        "openai": lambda path: analyze_image_openai(openai_client, path, prompt),
        "gemini": lambda path: analyze_image_gemini(gemini_model, path, prompt),
        "claude": lambda path: analyze_image_claude(claude_client, path, prompt),
    }

    all_outputs = []

    for model_name, infer_fn in models.items():
        model_results = []
        print(f"Running {model_name.upper()} on {len(df)} images...")

        for i, row in df.iterrows():
            image_path = row["image_path"]
            print(f"[{i+1}/{len(df)}] {image_path}")

            try:
                preds = infer_fn(image_path)
            except Exception as e:
                preds = [f"ERROR: {e}"]

            model_results.append({
                "model": model_name,
                "image_id": row["image_id"],
                "ground_truth": row["structures"],
                "predictions": preds
            })

        out_file = results_dir / f"{model_name}_results.csv"
        pd.DataFrame(model_results).to_csv(out_file, index=False)
        print(f"Saved {out_file}")
        all_outputs.extend(model_results)

    # 汇总结果
    df_all = pd.DataFrame(all_outputs)
    df_all.to_csv(results_dir / "all_models_summary.csv", index=False)
    print("All models finished. Summary saved to results/all_models_summary.csv")
