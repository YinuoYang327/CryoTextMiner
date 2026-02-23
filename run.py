# run.py
import pandas as pd
from pathlib import Path
from llm import analyze_image_openai, analyze_image_gemini, analyze_image_claude

def run_all_models(openai_client, gemini_model, claude_client, prompt_text, experiment_name, dataset_csv="demo_dataset/annotations_segmenetation.csv"):
    """
    Executes model inference and saves results into a wide-format CSV.
    Supports both Coordinate Detection and Segmentation modes.
    """
    # 1. Check if dataset exists
    if not Path(dataset_csv).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_csv}. Make sure to use the 1011 Master version.")

    df = pd.read_csv(dataset_csv)
    
    # 2. Create the output directory
    results_base = Path("results")
    experiment_dir = results_base / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory initialized: {experiment_dir}")
    print(f"Using Prompt: {experiment_name}")

    # Map model identifiers to their corresponding inference functions
    models = {
        "openai": lambda path: analyze_image_openai(openai_client, path, prompt_text),
        "gemini": lambda path: analyze_image_gemini(gemini_model, path, prompt_text),
        "claude": lambda path: analyze_image_claude(claude_client, path, prompt_text),
    }

    # Initialize summary dataframe by copying the original dataset
    summary_df = df.copy()

    for model_name, infer_fn in models.items():
        print(f"\n--- Running Inference: {model_name.upper()} ---")
        model_predictions = []

        for i, row in df.iterrows():
            image_path = row["image_path"]
            print(f"  [{i+1}/{len(df)}] Processing: {image_path}")

            try:
                # The prompt_text passed here will be the BBox prompt from collection.txt
                preds = infer_fn(image_path)
            except Exception as e:
                print(f"Error for {image_path} with {model_name}: {e}")
                preds = f"ERROR: {e}"

            model_predictions.append(preds)

        summary_df[f"{model_name}_predictions"] = model_predictions
        
        # Backup individual results
        individual_out = experiment_dir / f"{model_name}_results.csv"
        # Include gt_bboxes in the backup if it exists
        cols_to_save = ["image_id", f"{model_name}_predictions"]
        if "gt_bboxes" in summary_df.columns:
            cols_to_save.insert(1, "gt_bboxes")
            
        summary_df[cols_to_save].to_csv(individual_out, index=False)

    # --- Final Step: Save the master summary ---
    summary_file = experiment_dir / "all_models_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n Wide-format summary saved: {summary_file}")
    return summary_file