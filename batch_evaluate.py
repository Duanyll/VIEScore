import argparse
import pandas as pd
import os
import math
import csv
from pathlib import Path

# Rich for beautiful outputs
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

# Import VIEScore and the face similarity tool
from viescore import VIEScore

# --- Global variable for insightface model to avoid re-initialization ---
g_insightface_model = None

def get_insightface_model():
    """Initializes and returns a singleton insightface model."""
    global g_insightface_model
    if g_insightface_model is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            print("Initializing InsightFace model for Deepfake SC score...")
            # Using a list for providers is recommended
            providers = ['CPUExecutionProvider']
            g_insightface_model = FaceAnalysis(name='buffalo_l', providers=providers)
            g_insightface_model.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace model initialized.")
        except ImportError:
            raise ImportError("Please install 'insightface' and its dependencies (`pip install insightface onnxruntime opencv-python`) to run deepfake evaluation.")
    return g_insightface_model

def calculate_face_similarity_score(img_path1, img_path2):
    """
    Calculates the scaled face cosine similarity score for the deepfake task.
    SC_score = 10 * max(0, cos(f_target, f_gen))
    """
    try:
        import cv2
        import numpy as np
        model = get_insightface_model()

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            return 0.0 # Cannot read image, score is 0

        faces1 = model.get(img1)
        faces2 = model.get(img2)

        if not faces1 or not faces2:
            return 0.0 # No face detected, score is 0

        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding
        
        # Calculate cosine similarity from normalized embeddings
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # As per competition rules: scale and clamp
        sc_score = 10 * max(0, similarity)
        return sc_score

    except Exception:
        # If any error occurs during face analysis, return 0
        return 0.0


def evaluate_deepfake(ori_image_path, target_image_path, gen_image_path, viescore_model):
    """
    Custom evaluation logic for the 'deepfake' task.
    """
    # 1. Calculate SC score using face similarity
    sc_score = calculate_face_similarity_score(target_image_path, gen_image_path)
    
    # 2. Calculate PQ score using VIEScore's 'tie' mode
    # Fabricate a prompt as requested
    pq_prompt = "Swap the person's face with a different one while keeping the background, pose, and all other elements unchanged."
    
    try:
        # VIEScore returns [SC_score, PQ_score, O_score]
        # We only need the PQ score from this call.
        _, pq_score, _ = viescore_model.evaluate(
            image_prompts=[ori_image_path, gen_image_path],
            text_prompt=pq_prompt,
            extract_all_score=True
        )
    except Exception:
        # If VIEScore fails, PQ is 0
        pq_score = 0.0
        
    # 3. Calculate final Overall score
    # Use math.sqrt only if scores are valid to avoid math domain error
    o_score = math.sqrt(sc_score * pq_score) if sc_score > 0 and pq_score > 0 else 0.0
    
    return [sc_score, pq_score, o_score]


def main(args):
    console = Console()
    console.print(Panel(f"[bold cyan]VIEScore Batch Evaluator[/bold cyan]\nTask: [yellow]{args.task}[/yellow]\nBackbone: [yellow]{args.backbone}[/yellow]", title="Configuration", expand=False))

    # --- 1. Load and filter index file ---
    try:
        df = pd.read_csv(args.index_file)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Index file not found at [green]{args.index_file}[/green]")
        return

    if args.task == 'tie': # combine tie and vttie
        task_filters = ['tie', 'vttie']
        df_task = df[df['task_type'].isin(task_filters)].copy()
        viescore_task_name = 'tie'
    else:
        df_task = df[df['task_type'] == args.task].copy()
        viescore_task_name = 't2i' if args.task == 't2i' else 'tie'

    if df_task.empty:
        console.print(f"[bold red]Error:[/bold red] No entries found for task '{args.task}' in the index file.")
        return

    # --- 2. Initialize Models ---
    console.print("Initializing VIEScore model...")
    viescore_model = VIEScore(backbone=args.backbone, task=viescore_task_name, key_path=args.key_path)
    console.print(f"VIEScore for task '{viescore_task_name}' initialized.")
    
    # For deepfake, PQ score uses 'tie' model, which is already set. SC score uses insightface.
    if args.task == 'deepfake':
        get_insightface_model() # This will initialize the model on first call

    # --- 3. Setup output file and progress bar ---
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open file in append mode and create a csv writer
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header only if the file is new/empty
        if f.tell() == 0:
            writer.writerow(['index', 'SC_score', 'PQ_score', 'O_score'])

        # --- 4. Main evaluation loop ---
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_progress = progress.add_task(f"[cyan]Evaluating {args.task}", total=len(df_task))

            for _, row in df_task.iterrows():
                index = row['index']
                gen_image_path = os.path.join(args.input_folder, f"{index}.png")
                if not os.path.exists(gen_image_path):
                    gen_image_path = os.path.join(args.input_folder, f"{index}.jpg")

                if not os.path.exists(gen_image_path):
                    console.print(f"[yellow]Warning:[/yellow] Generated image not found for index {index}, skipping.")
                    progress.update(task_progress, advance=1)
                    continue

                scores = []
                try:
                    if args.task == 't2i':
                        prompt = row['prompt']
                        scores = viescore_model.evaluate(gen_image_path, prompt, extract_all_score=True)
                    
                    elif args.task in ['tie', 'vttie']:
                        prompt = row['prompt']
                        ori_image_path = os.path.join(args.image_folder, row['ori_image'])
                        if not os.path.exists(ori_image_path):
                            console.print(f"[yellow]Warning:[/yellow] Original image not found for index {index}, skipping.")
                            progress.update(task_progress, advance=1)
                            continue
                        scores = viescore_model.evaluate([ori_image_path, gen_image_path], prompt, extract_all_score=True)

                    elif args.task == 'deepfake':
                        ori_image_path = os.path.join(args.image_folder, row['ori_image'])
                        target_image_path = os.path.join(args.image_folder, row['target_image'])
                        if not os.path.exists(ori_image_path) or not os.path.exists(target_image_path):
                            console.print(f"[yellow]Warning:[/yellow] Original or Target image not found for index {index}, skipping.")
                            progress.update(task_progress, advance=1)
                            continue
                        scores = evaluate_deepfake(ori_image_path, target_image_path, gen_image_path, viescore_model)

                except Exception as e:
                    console.print(f"[bold red]Error on index {index}:[/bold red] {e}")
                    scores = [0.0, 0.0, 0.0]

                # --- 5. Write results immediately ---
                if scores:
                    writer.writerow([index] + scores)
                    f.flush() # Ensure it's written to disk

                progress.update(task_progress, advance=1)

    console.print(f"\n[bold green]Evaluation complete![/bold green] Results saved to [underline]{output_path}[/underline].")

    # --- 6. Calculate and display average scores ---
    try:
        results_df = pd.read_csv(output_path)
        # Filter for the current run's task indices to avoid including past results
        current_indices = df_task['index'].tolist()
        final_results = results_df[results_df['index'].isin(current_indices)]

        if not final_results.empty:
            avg_sc = final_results['SC_score'].mean()
            avg_pq = final_results['PQ_score'].mean()
            avg_o = final_results['O_score'].mean()

            table = Table(title="Average Scores")
            table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
            table.add_column("Score", style="magenta")
            table.add_row("Avg. SC Score", f"{avg_sc:.4f}")
            table.add_row("Avg. PQ Score", f"{avg_pq:.4f}")
            table.add_row("Avg. Overall Score", f"{avg_o:.4f}")
            console.print(table)
        else:
            console.print("[yellow]No results were generated in this run to calculate averages.[/yellow]")
    except Exception as e:
        console.print(f"[red]Could not calculate final averages: {e}[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation script using VIEScore.")
    parser.add_argument("-i", "--input_folder", required=True, help="Folder with generated images named as ${index}.png")
    parser.add_argument("-t", "--task", required=True, choices=['t2i', 'tie', 'vttie', 'deepfake'], help="Task type to evaluate.")
    parser.add_argument("--index_file", default="index.csv", help="Path to the master CSV index file.")
    parser.add_argument("--image_folder", default="images/", help="Path to the folder containing original/target images from the index file.")
    parser.add_argument("-o", "--output_csv", default="results.csv", help="Path to save the output scores.")
    parser.add_argument("--backbone", default="qwenvl", help="Backbone MLLM for VIEScore (e.g., 'gemini', 'gpt4o').")
    parser.add_argument("--key_path", default=None, help="Path to API key file if required by the backbone.")
    
    args = parser.parse_args()
    main(args)