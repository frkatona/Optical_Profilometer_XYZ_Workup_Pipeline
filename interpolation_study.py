import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
import time
from pathlib import Path

def generate_synthetic_data(shape=(128, 128), step_size=16, noise_type='gaussian', noise_level=0.5):
    """
    Generate synthetic height data with steps and varying noise types.
    
    Args:
        shape (tuple): Grid dimensions (height, width).
        step_size (int): Width of the plateau steps.
        noise_type (str): 'gaussian', 'poisson', 'speckle'.
        noise_level (float): Scale parameter for noise.
    
    Returns:
        np.ndarray: Generated height data.
    """
    height, width = shape
    data = np.zeros(shape)
    
    # Create stepped pattern (strips for "plateau every other 16 pixels")
    for y in range(height):
        for x in range(width):
            if (x // step_size) % 2 == 1:
                data[y, x] = 10.0  # Plateau height
            
    # Add noise
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, shape)
        data += noise
    elif noise_type == 'poisson':
        # Simulated shot noise: variance = mean
        # We need positive data for Poisson. Let's offset, add noise, then remove offset
        # Scale determines the "intensity" (higher intensity = relatively lower noise)
        offset = 20.0 
        temp_data = (data + offset) * noise_level # scale up
        # Poisson sample
        noisy = np.random.poisson(temp_data).astype(float)
        # Scale back
        data = (noisy / noise_level) - offset
    elif noise_type == 'speckle':
        # Multiplicative noise: data + data * noise
        noise = np.random.normal(0, noise_level, shape)
        data = data + data * noise
        
    return data

def apply_mask(data, ratio=0.3, mask_type='random', seed=42):
    """
    Simulate missing data by setting pixels to NaN.
    """
    np.random.seed(seed)
    masked_data = data.copy()
    height, width = data.shape
    
    if mask_type == 'random':
        mask = np.random.random(data.shape) < ratio
        masked_data[mask] = np.nan
        
    elif mask_type == 'scratch':
        n_scratches = int(ratio * 100)
        for _ in range(n_scratches):
            sy, sx = np.random.randint(0, height), np.random.randint(0, width)
            h, w = np.random.randint(1, height//10), np.random.randint(1, width//4)
            masked_data[sy:sy+h, sx:sx+w] = np.nan
            
    return masked_data

def interpolate_data(masked_data, method='linear'):
    """
    Interpolate missing values including Laplacian and Kriging.
    """
    y, x = np.indices(masked_data.shape)
    mask = ~np.isnan(masked_data)
    
    # Common setup for griddata-based methods
    points = np.column_stack((y[mask], x[mask]))
    values = masked_data[mask]
    grid_coords = np.column_stack((y.ravel(), x.ravel()))
    
    if method in ['nearest', 'linear', 'cubic']:
        filled = griddata(points, values, grid_coords, method=method)
        filled = filled.reshape(masked_data.shape)
        return filled
        
    elif method == 'laplacian':
        # Iterative diffusion
        filled = masked_data.copy()
        # Initialize NaNs with mean
        nan_mask = np.isnan(filled)
        filled[nan_mask] = np.nanmean(filled)
        
        # Simple convolution loop (faster than manual per-pixel)
        # We simulate diffusion for N steps
        iterations = 100
        for _ in range(iterations):
            # Shifted arrays
            up = np.roll(filled, -1, axis=0)
            down = np.roll(filled, 1, axis=0)
            left = np.roll(filled, -1, axis=1)
            right = np.roll(filled, 1, axis=1)
            
            # Update only originally missing pixels
            average = (up + down + left + right) / 4.0
            filled[nan_mask] = average[nan_mask]
            
        return filled
        
    elif method == 'kriging':
        # RBF Interpolation (Thin Plate Spline)
        # Optimize: Subsample if too many points
        max_points = 2000
        if len(values) > max_points:
            idx = np.random.choice(len(values), max_points, replace=False)
            fit_points = points[idx]
            fit_values = values[idx]
        else:
            fit_points = points
            fit_values = values
            
        # Create interpolator
        rbf = RBFInterpolator(fit_points, fit_values, kernel='thin_plate_spline')
        
        # Predict on grid
        filled_flat = rbf(grid_coords)
        filled = filled_flat.reshape(masked_data.shape)
        return filled
        
    else:
        raise ValueError(f"Unknown method {method}")

def compare_methods(original, masked, methods=['nearest', 'linear', 'cubic', 'laplacian', 'kriging']):
    """
    Run interpolation methods and compare results.
    """
    def get_metrics(truth, reconstruction, mask):
        missing_mask = ~mask
        eval_mask = missing_mask & ~np.isnan(reconstruction)
        
        if np.sum(eval_mask) == 0:
            return float('nan'), float('nan')
            
        diff = truth[eval_mask] - reconstruction[eval_mask]
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        return mae, rmse

    print(f"{'Method':<15} | {'MAE':<10} | {'RMSE':<10} | {'Time (s)':<10}")
    print("-" * 55)
    
    results = {}
    
    for method in methods:
        start_time = time.time()
        try:
            reconstructed = interpolate_data(masked, method=method)
            dt = time.time() - start_time
            
            mae, rmse = get_metrics(original, reconstructed, ~np.isnan(masked))
            
            results[method] = {
                'data': reconstructed,
                'mae': mae,
                'rmse': rmse,
                'time': dt
            }
            print(f"{method:<15} | {mae:<10.4f} | {rmse:<10.4f} | {dt:<10.4f}")
            
        except Exception as e:
            print(f"{method:<15} | FAILED: {e}")
            
    return results

def visualize_comparison(original, masked, results, title="Interpolation Comparison"):
    """
    Visualize original, masked, reconstructed results AND error maps.
    """
    method_names = list(results.keys())
    n_methods = len(method_names)
    
    rows = 3
    # Use max(2, n_methods) to ensure we have at least 2 columns
    cols = max(2, n_methods)
    
    fig = plt.figure(figsize=(4 * cols, 12))
    
    # Row 1: Truth & Masked (Centered or left-aligned)
    ax0 = plt.subplot(rows, cols, 1)
    im0 = ax0.imshow(original, cmap='viridis')
    ax0.set_title("Ground Truth")
    plt.colorbar(im0, ax=ax0)
    
    ax1 = plt.subplot(rows, cols, 2)
    cmap_masked = plt.cm.viridis.copy()
    cmap_masked.set_bad('white', 1.0)
    im1 = ax1.imshow(masked, cmap=cmap_masked)
    ax1.set_title("Masked Input")
    plt.colorbar(im1, ax=ax1)
    
    # Global error bounds for consistent colormap
    all_diffs = []
    for m in method_names:
        res = results[m]['data']
        diff = np.abs(original - res)
        all_diffs.append(diff)
    max_err = np.nanmax(all_diffs) if all_diffs else 1.0

    for i, method in enumerate(method_names):
        # Row 2: Reconstruction (cols+1..cols+N)
        # Note: if N < cols, we might have empty slots, which is fine
        if i >= cols: break 
        
        ax_rec = plt.subplot(rows, cols, cols + i + 1)
        res = results[method]
        im_rec = ax_rec.imshow(res['data'], cmap='viridis')
        ax_rec.set_title(f"{method.capitalize()}\nRMSE: {res['rmse']:.3f} | T: {res['time']:.2f}s")
        plt.colorbar(im_rec, ax=ax_rec)
        
        # Row 3: Difference Map (2*cols+1..2*cols+N)
        ax_err = plt.subplot(rows, cols, 2 * cols + i + 1)
        diff = np.abs(original - res['data'])
        
        im_err = ax_err.imshow(diff, cmap='inferno', vmin=0, vmax=max_err)
        ax_err.set_title(f"Diff: {method.capitalize()}")
        plt.colorbar(im_err, ax=ax_err)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

if __name__ == "__main__":
    def run_study():
        # Define simulation scenarios
        noise_settings = [
            {'type': 'gaussian', 'level': 0.5, 'name': 'Gaussian (Sigma=0.5)'},
            {'type': 'poisson', 'level': 5.0, 'name': 'Poisson (Scale=5)'}, # Higher scale = cleaner
            {'type': 'speckle', 'level': 0.1, 'name': 'Speckle (Var=0.1)'} 
        ]
        
        mask_settings = [
            {'ratio': 0.3, 'type': 'random', 'name': 'Random 30%'},
            {'ratio': 0.1, 'type': 'scratch', 'name': 'Scratch 10%'}
        ]
        
        # Run all combinations
        for noise in noise_settings:
            print(f"\nGeneratin Data: {noise['name']}...")
            ground_truth = generate_synthetic_data(
                shape=(128, 128), 
                step_size=16, 
                noise_type=noise['type'], 
                noise_level=noise['level']
            )
            
            for mask in mask_settings:
                # Limit scratch mask for speckle/poisson to save time if needed, but 128x128 is fast
                print(f"\n--- Running: {noise['name']} + {mask['name']} ---")
                
                masked_data = apply_mask(ground_truth, ratio=mask['ratio'], mask_type=mask['type'])
                
                # Compare all methods
                methods_to_test = ['nearest', 'linear', 'cubic', 'laplacian', 'kriging']
                res = compare_methods(ground_truth, masked_data, methods=methods_to_test)
                
                # Save plot
                noise_clean = noise['type']
                mask_clean = mask['type']
                fname = f"comparison_{noise_clean}_{mask_clean}.png"
                visualize_comparison(ground_truth, masked_data, res, 
                                   title=f"{noise['name']} | {mask['name']}")
                plt.savefig(fname)
                print(f"Saved plot to {fname}")
                plt.close()

    run_study()
