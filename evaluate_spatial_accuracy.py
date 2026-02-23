# evaluate_spatial_accuracy.py
import pandas as pd
import json
import math
import re
from pathlib import Path
import ast

# Synonym library to bridge nomenclature gaps
SYNONYMS = {
    "lysosome": ["lysosome", "vesicle", "mvb", "multivesicular", "endosome", "vacuole", "circular membrane-bound"],
    "mitochondrion": ["mitochondrion", "mitochondria", "cristae", "double-membrane"],
    "membrane": ["membrane", "envelope", "bilayer", "er", "tubule", "nuclear envelope"], 
    "microtubule": ["microtubule", "microtubules", "filament", "cytoskeleton", "tubular structure", "linear density"], 
    "ribosome": ["ribosome", "puncta", "particle", "granule", "dense dots"]
}

def robust_json_parser(raw_text):
    if pd.isna(raw_text) or "ERROR" in str(raw_text) or "unable to identify" in str(raw_text).lower():
        return None
    
    text = str(raw_text).strip()
    
    # Strategy 1: Try ast.literal_eval (Handles single quotes perfectly)
    try:
        # Remove markdown code blocks if present
        clean_text = re.sub(r'```[a-z]*\n?|```', '', text).strip()
        # Find the dictionary part { ... }
        dict_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if dict_match:
            return ast.literal_eval(dict_match.group(1))
    except Exception:
        pass

    # Strategy 2: Standard JSON parsing (Backup)
    try:
        json_text = text.replace("'", '"')
        json_match = re.search(r'(\{.*\}|\[.*\])', json_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                for key in data.keys():
                    if isinstance(data[key], list): return data[key]
                return data
    except Exception:
        return None
    return None

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two [y, x] coordinates."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def evaluate_coordinate_errors(summary_path):
    """
    Performs full-range spatial evaluation. 
    Categorizes results into HIT (<150px) or OUTLIER (>=150px) rather than ignoring them.
    Handles both list-of-dicts and flat-dict response formats.
    """
    if not Path(summary_path).exists():
        print(f"Results file not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    PIXEL_TO_NM = 1.4985  # Physical scale per pixel (Voxel size)
    MODELS = ['openai', 'gemini', 'claude']
    THRESHOLD_PX = 150 # Radius for a 'High Precision Hit'
    
    overall_results = []

    print(f"\n{'Model':<10} | {'Image':<10} | {'Organelle':<15} | {'Status':<10} | {'Error (nm)'}")
    print("-" * 80)

    for model in MODELS:
        pred_col = f"{model}_predictions"
        if pred_col not in df.columns: continue

        for _, row in df.iterrows():
            img_id = row['image_id']
            # Ground Truth from expert annotation
            gt_dict = json.loads(row['gt_coords'])
            # Cleaned predictions from LLM
            predictions = robust_json_parser(row[pred_col])
            
            if not predictions:
                continue

            for gt_label, gt_pt in gt_dict.items():
                best_match_dist = float('inf')
                valid_words = SYNONYMS.get(gt_label, [gt_label])
                
                # Identify potential candidate coordinates from model output
                candidate_points = []
                
                # Format A: List of dictionaries e.g., [{"label": "lysosome", "center": [y, x]}]
                if isinstance(predictions, list):
                    for p in predictions:
                        if isinstance(p, dict):
                            p_label = str(p.get('label', '')).lower()
                            if any(word in p_label for word in valid_words):
                                candidate_points.append(p.get('center'))
                
                # Format B: Flat dictionary e.g., {"lysosome": [y, x]}
                elif isinstance(predictions, dict):
                    for p_label, p_pt in predictions.items():
                        if any(word in p_label.lower() for word in valid_words):
                            # Ensure the value is a coordinate list [y, x] and not a scalar
                            if isinstance(p_pt, list) and len(p_pt) == 2:
                                candidate_points.append(p_pt)

                # Find the distance to the nearest semantically matching candidate
                for p_pt in candidate_points:
                    if p_pt and isinstance(p_pt, list) and len(p_pt) == 2:
                        dist = calculate_distance(gt_pt, p_pt)
                        if dist < best_match_dist:
                            best_match_dist = dist

                # Record the distance and categorize as HIT or OUTLIER
                if best_match_dist != float('inf'):
                    dist_nm = best_match_dist * PIXEL_TO_NM
                    status = "HIT" if best_match_dist < THRESHOLD_PX else "OUTLIER"
                    
                    print(f"{model:<10} | {img_id:<10} | {gt_label:<15} | {status:<10} | {dist_nm:.2f} nm")
                    
                    overall_results.append({
                        "model": model,
                        "image": img_id,
                        "organelle": gt_label,
                        "status": status,
                        "error_nm": dist_nm
                    })
                else:
                    # Logic for organelles mentioned but without matching coordinates
                    overall_results.append({
                        "model": model, "image": img_id, "organelle": gt_label,
                        "status": "NOT_FOUND", "error_nm": None
                    })

    # Final Summary Report generation
    report_df = pd.DataFrame(overall_results)
    if not report_df.empty:
        print("\n" + "="*60)
        print("FINAL SPATIAL PERFORMANCE SUMMARY (1011 SCALE)")
        print("="*60)
        
        for model in report_df['model'].unique():
            model_data = report_df[report_df['model'] == model]
            hits = model_data[model_data['status'] == 'HIT']
            
            avg_hit_err = hits['error_nm'].mean() if not hits.empty else 0
            # reliability calculates average including outliers (if coordinates exist)
            valid_coords_df = model_data.dropna(subset=['error_nm'])
            avg_total_err = valid_coords_df['error_nm'].mean() if not valid_coords_df.empty else 0
            success_rate = (len(hits) / len(model_data)) * 100
            
            print(f"Model: {model.upper()}")
            print(f" - Success Rate (<150px): {success_rate:.1f}%")
            print(f" - Precision (Mean HIT Error): {avg_hit_err:.2f} nm")
            print(f" - Reliability (Mean Total Error): {avg_total_err:.2f} nm")
            print("-" * 30)
    else:
        print("\n Evaluation failed: No parseable spatial data found.")

if __name__ == "__main__":
    pass