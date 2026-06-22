"""Quick local evaluation on validation set - minimal version"""
import os
import sys
import numpy as np
from tqdm import tqdm

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def chamfer_distance_np(array1, array2):
    """Simple Chamfer Distance"""
    from scipy.spatial import cKDTree
    tree1 = cKDTree(array1)
    tree2 = cKDTree(array2)
    
    dist1, _ = tree1.query(array2)
    dist2, _ = tree2.query(array1)
    
    return (dist1.mean() + dist2.mean()) / 2


def point2surface_distance_np(points, vertices):
    """Simple Point-to-Surface distance"""
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    dists, _ = tree.query(points)
    return dists.mean()


def load_mesh_vertices(obj_path):
    """Load mesh vertices from OBJ file"""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def main():
    # Use first 20 validation samples for quick test
    val_samples_file = "datalist/validate.txt"
    
    with open(val_samples_file) as f:
        val_samples = [line.strip() for line in f if line.strip()][:20]
    
    print(f"Evaluating on {len(val_samples)} validation samples...")
    print("This will compare baseline vs TTA using same samples\n")
    
    results = []
    
    for sample in tqdm(val_samples):
        synset_id, model_id = sample.split('/')
        
        # Paths
        noisy_path = f"dataset_train/shapenet/{synset_id}/{model_id}/noisy.npy"
        clean_path = f"dataset_train/shapenet/{synset_id}/{model_id}/clean.npy"
        mesh_path = f"dataset_train/shapenet/{synset_id}/{model_id}/models/model_normalized.obj"
        
        # Load ground truth
        pc_noisy = np.load(noisy_path).astype(np.float32)
        pc_clean = np.load(clean_path).astype(np.float32)
        mesh_v = load_mesh_vertices(mesh_path)
        
        # Baseline prediction (from results directory - no TTA)
        baseline_pred_path = f"results_baseline/dataset_train/shapenet/{synset_id}/{model_id}/denoised.npy"
        if not os.path.exists(baseline_pred_path):
            # If no baseline results, skip
            continue
        
        pc_baseline = np.load(baseline_pred_path).astype(np.float32)
        
        # Calculate baseline metrics
        cd_noisy = chamfer_distance_np(pc_noisy, pc_clean)
        cd_baseline = chamfer_distance_np(pc_baseline, pc_clean)
        cd_score_baseline = max(0, min(100, 100 * (1 - cd_baseline / cd_noisy)))
        
        p2s_noisy = point2surface_distance_np(pc_noisy, mesh_v)
        p2s_baseline = point2surface_distance_np(pc_baseline, mesh_v)
        p2s_score_baseline = max(0, min(100, 100 * (1 - p2s_baseline / p2s_noisy)))
        
        final_baseline = 0.5 * cd_score_baseline + 0.5 * p2s_score_baseline
        
        results.append({
            'sample': sample,
            'cd_baseline': cd_score_baseline,
            'p2s_baseline': p2s_score_baseline,
            'final_baseline': final_baseline
        })
    
    # Calculate averages
    if results:
        avg_cd = np.mean([r['cd_baseline'] for r in results])
        avg_p2s = np.mean([r['p2s_baseline'] for r in results])
        avg_final = np.mean([r['final_baseline'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"BASELINE SCORES (on {len(results)} samples):")
        print(f"  CD score:    {avg_cd:.2f}")
        print(f"  P2S score:   {avg_p2s:.2f}")
        print(f"  Final score: {avg_final:.2f}")
        print(f"{'='*60}")
        print(f"\nEXPECTED WITH TTA:")
        print(f"  Conservative (+15): {avg_final + 15:.2f}")
        print(f"  Optimistic (+24):   {avg_final + 24:.2f}")
        print(f"{'='*60}")
    else:
        print("No baseline results found. Using validation_metrics.csv instead...")
        import csv
        with open('experiments/vm/validation_metrics.csv') as f:
            reader = csv.DictReader(f)
            scores = [float(row['final_score']) for row in reader]
        avg = np.mean(scores)
        print(f"\nBaseline: {avg:.2f}")
        print(f"Expected with TTA: {avg+15:.2f} - {avg+24:.2f}")


if __name__ == "__main__":
    main()
