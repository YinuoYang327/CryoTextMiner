# evaluate_vlm_results.py
import pandas as pd
import ast
from pathlib import Path

# Define synonyms to bridge the gap between AI descriptions and Expert labels
SYNONYMS = {
    "lysosome": ["lysosome", "vesicle", "mvb", "multivesicular", "endosome", "vacuole", "circular membrane-bound"],
    "mitochondrion": ["mitochondrion", "mitochondria", "cristae", "double-membrane"],
    "membrane": ["membrane", "envelope", "bilayer", "er", "tubule", "nuclear envelope"], 
    "microtubule": ["microtubule", "microtubules", "filament", "cytoskeleton", "tubular structure", "linear density"], 
    "ribosome": ["ribosome", "puncta", "particle", "granule", "dense dots"]
}

def evaluate_results(results_path="results/all_models_summary.csv"):
    """
    Evaluates VLM performance using fuzzy matching (synonyms) to account for 
    different descriptive terms used by AI models.
    """
    path = Path(results_path)
    if not path.exists():
        print(f"⚠️ Warning: Results file not found at {results_path}. Run inference first.")
        return

    print(f"\n📊 Loading results for fuzzy-match evaluation: {path}")
    df = pd.read_csv(path)

    # Safely parse string lists into Python lists
    def safe_parse(x):
        if pd.isna(x): return []
        if isinstance(x, list): return x
        try:
            # Handle standard list strings: ['a', 'b']
            return ast.literal_eval(x)
        except:
            # Handle raw text or improperly formatted lists
            return [str(x)]

    df["ground_truth"] = df["ground_truth"].apply(safe_parse)
    df["predictions"] = df["predictions"].apply(safe_parse)

    # Identify all unique organelle types present in the Ground Truth
    all_gt_labels = set([label for sublist in df["ground_truth"] for label in sublist])
    
    results_summary = []

    # Calculate Recall for each model
    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        metrics = {"model": model_name, "total_samples": len(model_df)}
        
        for label in sorted(all_gt_labels):
            label_lower = label.lower()
            
            def check_hit(row):
                # Only evaluate if this label exists in the expert Ground Truth for this image
                gt_list = [str(g).lower() for g in row["ground_truth"]]
                if label_lower not in gt_list:
                    return None
                
                # Get the list of accepted synonyms for this organelle
                valid_keywords = SYNONYMS.get(label_lower, [label_lower])
                
                # Join all AI predictions into one big string for searching
                pred_text = " ".join([str(p).lower() for p in row["predictions"]])
                
                # Check if ANY synonym or the label itself is mentioned
                if any(word in pred_text for word in valid_keywords):
                    return 1
                return 0

            hits = model_df.apply(check_hit, axis=1).dropna()
            
            if len(hits) > 0:
                recall = hits.mean()
                metrics[f"{label}_recall"] = f"{recall:.1%}"
            else:
                metrics[f"{label}_recall"] = "N/A"
        
        results_summary.append(metrics)

    # Create and display the summary table
    summary_df = pd.DataFrame(results_summary)
    
    # Reorder columns to put model and total first
    cols = ["model", "total_samples"] + [c for c in summary_df.columns if "recall" in c]
    summary_df = summary_df[cols]

    print("\n" + "="*80)
    print("CRYOTEXTMINER: IDENTIFICATION PERFORMANCE (FUZZY MATCHING)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Save final report
    out_path = Path(results_path).parent / "evaluation_report_fuzzy.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n Fuzzy-match evaluation report saved to {out_path}")

if __name__ == "__main__":
    evaluate_results()