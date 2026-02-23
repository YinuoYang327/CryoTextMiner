# main.py
import os
from utils import load_api_keys, get_prompt_by_id
from llm import init_openai_client, init_gemini_client, init_claude_client
from run import run_all_models
from evaluate_vlm_results import evaluate_results
from evaluate_spatial_accuracy import evaluate_coordinate_errors
from evaluate_segmentation_iou import evaluate_segmentation_performance

def main(mode="identification"):
    """
    Central orchestration script for VLM Cryo-ET identification and localization experiments.
    """
    # --- 1. Path Configuration ---
    # Define file paths before using them in function calls
    KEY_FILE = "keys/api_keys.txt"
    PROMPT_FILE = "prompts/collection.txt"

    # --- 2. Initialization ---
    print(f"--- System Initialization ---")
    keys = load_api_keys(KEY_FILE)
    
    # Initialize clients for OpenAI, Gemini, and Claude
    o_client = init_openai_client(keys.get("OPENAI_API_KEY"))
    g_model  = init_gemini_client(keys.get("GEMINI_API_KEY"))
    c_client = init_claude_client(keys.get("ANTHROPIC_API_KEY"))

    # --- 3. Experiment Mode Selection ---
    if mode == "identification":
        # Traditional identification task using standard annotation labels
        DATASET_CSV = "demo_dataset/annotations.csv"
        SELECTED_PROMPT_ID = "SIMPLE_IDENTIFICATION_V2" 
        eval_func = evaluate_results 
    
    elif mode == "Coordinate Detection":
        # Visual grounding task using pixel-level expert centroids
        DATASET_CSV = "demo_dataset/annotations_with_coords_final.csv"
        SELECTED_PROMPT_ID = "COORDINATE_DETECTION_V2" 
        eval_func = evaluate_coordinate_errors 
    
    elif mode == "Segmentation":
        DATASET_CSV = "demo_dataset/annotations_segmenetation.csv"
        SELECTED_PROMPT_ID = "SEGMENTATION_3D_FEW_SHOT" 
        eval_func = evaluate_segmentation_performance 
    
    else:
        print(f"Critical Error: Unsupported mode '{mode}'")
        return

    # --- 4. Prompt Retrieval ---
    prompt_content = get_prompt_by_id(PROMPT_FILE, SELECTED_PROMPT_ID)
    if not prompt_content:
        print(f" Error: Could not retrieve prompt template for '{SELECTED_PROMPT_ID}'")
        return

    print(f" Active Mode: {mode}")
    print(f" Target Prompt ID: {SELECTED_PROMPT_ID}")

    # --- 5. Batch Inference Execution ---
    # run_all_models processes each image in the CSV independently (Single-slice Baseline)
    summary_path = run_all_models(
        openai_client=o_client,
        gemini_model=g_model,
        claude_client=c_client,
        prompt_text=prompt_content,
        dataset_csv=DATASET_CSV,
        experiment_name=SELECTED_PROMPT_ID
    )

    # --- 6. Post-Inference Evaluation ---
    # Trigger the appropriate scoring function based on the experiment mode
    print(f"\n--- Launching Post-Processing Evaluation: {mode} ---")
    eval_func(summary_path)

if __name__ == "__main__":
    main(mode="Segmentation")