"""Visual quality check: compare noisy vs denoised point clouds"""
import numpy as np
import os

def check_denoising_quality(sample_path):
    """Check if denoising improved the point cloud quality"""
    noisy = np.load(os.path.join(sample_path, "noisy.npy"))
    denoised = np.load(os.path.join("results", sample_path, "denoised.npy"))
    
    # Calculate point spread (smaller = cleaner)
    noisy_std = np.std(noisy, axis=0).mean()
    denoised_std = np.std(denoised, axis=0).mean()
    
    # Calculate local density variation (smoother = better)
    from scipy.spatial import cKDTree
    tree_noisy = cKDTree(noisy)
    tree_denoised = cKDTree(denoised)
    
    # Sample 1000 points
    sample_indices = np.random.choice(len(noisy), min(1000, len(noisy)), replace=False)
    
    noisy_local_density = []
    denoised_local_density = []
    
    for idx in sample_indices:
        # K nearest neighbors distance variance (lower = smoother surface)
        dists_n, _ = tree_noisy.query(noisy[idx], k=20)
        dists_d, _ = tree_denoised.query(denoised[idx], k=20)
        
        noisy_local_density.append(np.std(dists_n))
        denoised_local_density.append(np.std(dists_d))
    
    noisy_smoothness = np.mean(noisy_local_density)
    denoised_smoothness = np.mean(denoised_local_density)
    
    improvement = (noisy_smoothness - denoised_smoothness) / noisy_smoothness * 100
    
    return {
        'noisy_spread': noisy_std,
        'denoised_spread': denoised_std,
        'noisy_roughness': noisy_smoothness,
        'denoised_roughness': denoised_smoothness,
        'improvement_pct': improvement
    }


def main():
    # Check 10 random test samples
    import glob
    test_samples = glob.glob("dataset_test_noisy/shapenet/*/*/noisy.npy")
    test_samples = np.random.choice(test_samples, min(10, len(test_samples)), replace=False)
    
    print("="*70)
    print("🔍 QUALITY CHECK: Noisy vs TTA-Denoised")
    print("="*70)
    print(f"Checking {len(test_samples)} random samples...\n")
    
    improvements = []
    
    for i, sample_path in enumerate(test_samples):
        sample_path = os.path.dirname(sample_path)
        sample_name = "/".join(sample_path.split("/")[-2:])
        
        try:
            stats = check_denoising_quality(sample_path)
            improvements.append(stats['improvement_pct'])
            
            status = "✅ Better" if stats['improvement_pct'] > 0 else "⚠️ Worse"
            print(f"{i+1}. {sample_name[:40]:40s} {status} ({stats['improvement_pct']:+.1f}%)")
            
        except Exception as e:
            print(f"{i+1}. {sample_name[:40]:40s} ❌ Error: {str(e)[:30]}")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"\n{'='*70}")
        print(f"Average surface smoothness improvement: {avg_improvement:+.1f}%")
        print(f"Samples improved: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
        print(f"{'='*70}\n")
        
        if avg_improvement > 10:
            print("✅ EXCELLENT: TTA significantly improved denoising quality")
            print("   Expected test score: 75-85")
        elif avg_improvement > 5:
            print("✅ GOOD: TTA improved denoising quality") 
            print("   Expected test score: 70-80")
        elif avg_improvement > 0:
            print("⚠️  MARGINAL: TTA has small positive effect")
            print("   Expected test score: 65-75")
        else:
            print("❌ WARNING: TTA may not be helping")
            print("   Consider using baseline results instead")


if __name__ == "__main__":
    main()
