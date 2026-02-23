# run_multiple.py
import pandas as pd
from pathlib import Path
# Import the new multiple-image clients
from llm import analyze_image_openai, analyze_image_gemini, analyze_image_claude
from llm import analyze_sequence_claude
from llm import analyze_sequence_gemini
from llm import analyze_sequence_openai

def run_multiple_inference(openai_client, gemini_model, claude_client, prompt_text, experiment_name, dataset_csv="demo_dataset/annotations.csv"):
    """
    Groups all images in the CSV as a single sequence and runs inference.
    """
    df = pd.read_csv(dataset_csv)
    image_paths = df["image_path"].tolist()
    
    # Setup results directory
    results_base = Path("results")
    experiment_dir = results_base / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sequence Mode: Results will be saved to: {experiment_dir}")
    print(f"Grouping {len(image_paths)} images into one sequence context.")

    # Define the models and their multi-image functions
    models = {
        "openai": lambda paths: analyze_sequence_openai(openai_client, paths, prompt_text),
        "gemini": lambda paths: analyze_sequence_gemini(gemini_model, paths, prompt_text),
        "claude": lambda paths: analyze_sequence_claude(claude_client, paths, prompt_text),
    }

    all_outputs = []

    for model_name, infer_fn in models.items():
        print(f"\n Running {model_name.upper()} sequence inference...")
        
        try:
            # Perform inference on the ENTIRE list of paths
            prediction_content = infer_fn(image_paths)
        except Exception as e:
            print(f"Critical Error with {model_name}: {e}")
            prediction_content = [f"SEQUENCE ERROR: {e}"]

        # Distribute the single sequence prediction back to each image row for the summary
        model_results = []
        for i, row in df.iterrows():
            model_results.append({
                "model": model_name,
                "image_id": row.get("image_id"),
                "ground_truth": row.get("structures"),
                "predictions": prediction_content # Every image in the sequence shares the same context
            })

        # Save individual model sequence results
        out_file = experiment_dir / f"{model_name}_results.csv"
        pd.DataFrame(model_results).to_csv(out_file, index=False)
        all_outputs.extend(model_results)

    # Save final combined summary
    summary_file = experiment_dir / "all_models_summary.csv"
    pd.DataFrame(all_outputs).to_csv(summary_file, index=False)
    
    return summary_file