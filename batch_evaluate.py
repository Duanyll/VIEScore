# batch_evaluate.py

import argparse
import pandas as pd
import os
import math
import csv
import json # <-- Import json for handling detailed scores
from pathlib import Path

# Rich for beautiful outputs
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

# Import VIEScore
from viescore import VIEScore

# --- InsightFace related functions (no changes here) ---
g_insightface_model = None
def get_insightface_model():
    global g_insightface_model
    if g_insightface_model is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            console.print("Initializing InsightFace model for Deepfake SC score...", style="yellow")
            providers = ['CPUExecutionProvider']
            g_insightface_model = FaceAnalysis(name='buffalo_l', providers=providers)
            g_insightface_model.prepare(ctx_id=0, det_size=(640, 640))
            console.print("InsightFace model initialized.", style="bold green")
        except ImportError:
            raise ImportError("Please install 'insightface' and its dependencies to run deepfake evaluation.")
    return g_insightface_model

def calculate_face_similarity_score(img_path1, img_path2):
    try:
        import cv2
        import numpy as np
        model = get_insightface_model()
        img1, img2 = cv2.imread(img_path1), cv2.imread(img_path2)
        if img1 is None or img2 is None: return 0.0
        faces1, faces2 = model.get(img1), model.get(img2)
        if not faces1 or not faces2: return 0.0
        emb1, emb2 = faces1[0].embedding, faces2[0].embedding
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(10 * max(0, similarity))
    except Exception:
        return 0.0

def evaluate_deepfake(ori_image_path, target_image_path, gen_image_path, viescore_model):
    """
    [MODIFIED] Custom evaluation logic for 'deepfake' using optimized calls.
    """
    # 1. Calculate SC score using face similarity (no MLLM call)
    sc_score = calculate_face_similarity_score(target_image_path, gen_image_path)
    # Create a corresponding details dictionary for consistency
    sc_details = {'score': [sc_score], 'reason': 'Calculated via face cosine similarity.', 'category': ['face_similarity']}
    
    # 2. Calculate PQ score using VIEScore's 'evaluate_pq' (ONLY ONE MLLM call)
    try:
        pq_details = viescore_model.evaluate_pq([gen_image_path])
        pq_score = min(pq_details.get('score', [0.0]))
    except Exception:
        pq_details = {'score': [0.0], 'reason': 'VIEScore PQ evaluation failed.'}
        pq_score = 0.0
        
    # 3. Calculate final Overall score
    o_score = math.sqrt(sc_score * pq_score) if sc_score > 0 and pq_score > 0 else 0.0
    
    return sc_score, pq_score, o_score, sc_details, pq_details


def main(args):
    global console
    console = Console()
    console.print(Panel(f"[bold cyan]VIEScore Batch Evaluator (Optimized)[/bold cyan]\nTask: [yellow]{args.task}[/yellow]\nBackbone: [yellow]{args.backbone}[/yellow]", title="Configuration", expand=False))

    # --- Load and filter index file ---
    df = pd.read_csv(args.index_file)
    task_filters = ['tie', 'vttie'] if args.task == 'tie' else [args.task]
    df_task = df[df['task_type'].isin(task_filters)].copy()
    if df_task.empty:
        console.print(f"[bold red]Error:[/bold red] No entries for task '{args.task}'.")
        return
    viescore_task_name = 't2i' if args.task == 't2i' else 'tie'

    # --- Initialize Models ---
    console.print("Initializing VIEScore model...")
    viescore_model = VIEScore(backbone=args.backbone, task=viescore_task_name, key_path=args.key_path)
    console.print(f"VIEScore for task '{viescore_task_name}' initialized.", style="bold green")
    if args.task == 'deepfake':
        get_insightface_model()

    # --- Setup output file and progress bar ---
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            # [MODIFIED] New header with detailed sub-scores
            writer.writerow(['index', 'SC_score', 'PQ_score', 'O_score', 'SC_details', 'PQ_details'])

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), transient=True) as progress:
            task_progress = progress.add_task(f"[cyan]Evaluating {args.task}", total=len(df_task))

            for _, row in df_task.iterrows():
                index = row['index']
                gen_image_path = os.path.join(args.input_folder, f"{index}.png")

                if not os.path.exists(gen_image_path):
                    progress.console.print(f"[yellow]Warning:[/yellow] Skipping index {index}: Generated image not found.")
                    progress.update(task_progress, advance=1)
                    continue

                # --- [MODIFIED] Main evaluation logic ---
                sc_score, pq_score, o_score, sc_details, pq_details = 0.0, 0.0, 0.0, {}, {}
                try:
                    if args.task == 'deepfake':
                        ori_path = os.path.join(args.image_folder, row['ori_image'])
                        target_path = os.path.join(args.image_folder, row['target_image'])
                        sc_score, pq_score, o_score, sc_details, pq_details = evaluate_deepfake(ori_path, target_path, gen_image_path, viescore_model)
                    else: # t2i, tie, vttie
                        prompt = row['prompt']
                        image_prompts = gen_image_path
                        if args.task in ['tie', 'vttie']:
                            ori_path = os.path.join(args.image_folder, row['ori_image'])
                            image_prompts = [ori_path, gen_image_path]

                        results_dict = viescore_model.evaluate(image_prompts, prompt)
                        sc_details = results_dict.get('SC', {})
                        pq_details = results_dict.get('PQ', {})
                        sc_score = min(sc_details.get('score', [0.0]))
                        pq_score = min(pq_details.get('score', [0.0]))
                        o_score = math.sqrt(sc_score * pq_score) if sc_score > 0 and pq_score > 0 else 0.0

                except Exception as e:
                    progress.console.print(f"[bold red]Error on index {index}:[/bold red] {e}")
                
                # [MODIFIED] Write detailed results to CSV
                writer.writerow([
                    index, f"{sc_score:.4f}", f"{pq_score:.4f}", f"{o_score:.4f}",
                    json.dumps(sc_details), json.dumps(pq_details)
                ])
                f.flush()
                progress.update(task_progress, advance=1)

    console.print(f"\n[bold green]Evaluation complete![/bold green] Results saved to [underline]{output_path}[/underline].")
    
    # --- Calculate and display average scores (no changes needed here) ---
    results_df = pd.read_csv(output_path)
    current_indices = df_task['index'].tolist()
    final_results = results_df[results_df['index'].isin(current_indices)]
    if not final_results.empty:
        avg_sc = pd.to_numeric(final_results['SC_score']).mean()
        avg_pq = pd.to_numeric(final_results['PQ_score']).mean()
        avg_o = pd.to_numeric(final_results['O_score']).mean()
        
        table = Table(title="Average Scores")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_row("Avg. SC Score", f"{avg_sc:.4f}")
        table.add_row("Avg. PQ Score", f"{avg_pq:.4f}")
        table.add_row("Avg. Overall Score", f"{avg_o:.4f}")
        console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized batch evaluation script for VIEScore.")
    # --- Arguments (no changes) ---
    parser.add_argument("-i", "--input_folder", required=True, help="Folder with generated images named as ${index}.png")
    parser.add_argument("-t", "--task", required=True, choices=['t2i', 'tie', 'vttie', 'deepfake'], help="Task type to evaluate.")
    parser.add_argument("--index_file", default="index.csv", help="Path to the master CSV index file.")
    parser.add_argument("--image_folder", default="images/", help="Path to the folder containing original/target images.")
    parser.add_argument("-o", "--output_csv", default="results_detailed.csv", help="Path to save the output scores with details.")
    parser.add_argument("--backbone", default="qwenvl", help="Backbone MLLM for VIEScore (e.g., 'gemini', 'gpt4o').")
    parser.add_argument("--key_path", default=None, help="Path to API key file if required by the backbone.")
    
    args = parser.parse_args()
    main(args)