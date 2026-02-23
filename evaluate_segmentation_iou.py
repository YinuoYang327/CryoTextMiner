import pandas as pd
import json
import re
import ast
from pathlib import Path

# Synonym library to map LLM labels to expert ground truth labels
SYNONYMS = {
    "lysosome": ["lysosome", "vesicle", "mvb", "multivesicular", "endosome", "vacuole", "lysosome-type"],
    "mitochondrion": ["mitochondrion", "mitochondria", "cristae", "double-membrane", "mitochondrion-type"],
    "membrane": ["membrane", "envelope", "bilayer", "er", "tubule", "nuclear envelope", "membrane-type"]
}

def robust_json_parser(raw_text):
    """
    Parses LLM output strings into Python dictionaries.
    Supports markdown code blocks, single/double quotes, and malformed JSON.
    """
    if pd.isna(raw_text) or "ERROR" in str(raw_text) or "sorry" in str(raw_text).lower():
        return None
    
    text = str(raw_text).strip()
    try:
        # Remove markdown formatting if present
        clean_text = re.sub(r'```[a-z]*\n?|```', '', text).strip()
        # Extract content between curly braces
        dict_match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
        if dict_match:
            # Safely evaluate string as a Python dict
            return ast.literal_eval(dict_match.group(1))
    except Exception:
        pass
    return None

def get_enclosing_box(box_data):
    """
    Handles both single [y,x,y,x] and multiple [[y,x,y,x], [...]] boxes.
    If multiple boxes are provided, returns the minimal enclosing rectangle (Union).
    """
    if not isinstance(box_data, list) or len(box_data) == 0:
        return None
    
    # Case 1: Single box [ymin, xmin, ymax, xmax]
    if isinstance(box_data[0], (int, float)):
        return box_data if len(box_data) == 4 else None
    
    # Case 2: List of boxes [[y,x,y,x], [y,x,y,x]]
    try:
        # Filter valid sub-boxes with length 4
        valid_boxes = [b for b in box_data if isinstance(b, list) and len(b) == 4]
        if not valid_boxes: return None
        
        y_min = min(b[0] for b in valid_boxes)
        x_min = min(b[1] for b in valid_boxes)
        y_max = max(b[2] for b in valid_boxes)
        x_max = max(b[3] for b in valid_boxes)
        return [y_min, x_min, y_max, x_max]
    except Exception:
        return None

def calculate_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) for two boxes in [ymin, xmin, ymax, xmax] format.
    """
    # Coordinates of intersection
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0: return 0.0

    # Areas of individual boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

def evaluate_segmentation_performance(summary_path, example_ids=['z187']):
    """
    Main evaluation pipeline. Separates results into Generalization and Memorization.
    """
    if not Path(summary_path).exists():
        print(f"File not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    MODELS = ['openai', 'gemini', 'claude']
    results = []

    print(f"\n{'Model':<10} | {'Image':<10} | {'Organelle':<15} | {'IoU (%)':<10}")
    print("-" * 65)

    for model in MODELS:
        pred_col = f"{model}_predictions"
        if pred_col not in df.columns: continue

        for _, row in df.iterrows():
            img_id = row['image_id']
            gt_dict = json.loads(row['gt_bboxes'])
            predictions = robust_json_parser(row[pred_col])
            
            if not predictions: continue

            for gt_label, gt_box in gt_dict.items():
                best_iou = 0.0
                valid_keywords = SYNONYMS.get(gt_label, [gt_label])
                
                # Check for label matches in AI output
                for p_label, p_data in predictions.items():
                    if any(word in p_label.lower() for word in valid_keywords):
                        # Use get_enclosing_box to handle nested/multiple boxes
                        final_p_box = get_enclosing_box(p_data)
                        if final_p_box:
                            iou = calculate_iou(gt_box, final_p_box)
                            best_iou = max(best_iou, iou)

                print(f"{model:<10} | {img_id:<10} | {gt_label:<15} | {best_iou*100:>7.2f}%")
                results.append({"model": model, "image": img_id, "label": gt_label, "iou": best_iou})

    # Final reporting logic
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print("\n" + "="*65)
        print("FINAL SEGMENTATION SUMMARY (IoU)")
        print("="*65)
        for model in res_df['model'].unique():
            m_data = res_df[res_df['model'] == model]
            # Split Test vs Example
            test_set = m_data[~m_data['image'].isin(example_ids)]
            ex_set = m_data[m_data['image'].isin(example_ids)]
            
            test_avg = test_set['iou'].mean() if not test_set.empty else 0
            ex_avg = ex_set['iou'].mean() if not ex_set.empty else 0
            
            print(f"Model: {model.upper():<8}")
            print(f" - [Generalization] New Images Mean: {test_avg*100:.2f}%")
            print(f" - [Memorization]   Example IoU:     {ex_avg*100:.2f}%")
            print("-" * 45)