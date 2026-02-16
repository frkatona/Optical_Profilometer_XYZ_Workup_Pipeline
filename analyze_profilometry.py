"""
Optical Profilometry Data Analysis and Visualization Script

This script processes and visualizes XYZ optical profilometry data files
containing 1024x1024 height measurements with a 14-line header.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import argparse
import time
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter


def load_xyz_file(filepath, resolution_factor=1):
    """
    Load XYZ profilometry data from file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the XYZ file
    resolution_factor : int
        Downsampling factor (1, 2, 4, 8, etc.). Higher values = faster processing.
        
    Returns:
    --------
    data : np.ndarray
        2D array of height values (NaN for missing data)
    metadata : dict
        Dictionary containing file metadata
    """
    print(f"Loading file: {filepath}")
    print(f"Resolution factor: {resolution_factor}x")
    
    start_time = time.time()
    
    # Read the file
    header_lines = []
    with open(filepath, 'r') as f:
        # Read header (14 lines)
        for i in range(14):
            header_lines.append(f.readline().strip())
    
    # Parse array dimensions from header line 4 (index 3)
    # Format: reserved reserved width height
    try:
        dim_parts = header_lines[3].split()
        width = int(dim_parts[2])
        height = int(dim_parts[3])
    except (IndexError, ValueError):
        print("Warning: Could not parse dimensions from header line 4. Defaulting to 1024x1024.")
        width, height = 1024, 1024

    # Initialize array with dynamic dimensions
    full_data = np.full((height, width), np.nan)
    
    # Parse pixel spacing from header line 8 (0-indexed line 7)
    # 7th element is m/pixel
    try:
        parts = header_lines[7].split()
        if len(parts) >= 8:
            # Value 7: True lateral sampling (um per pixel)
            pixel_spacing_um = float(parts[6])*(10**6)
            
            # Value 4: Wavelength/modulation parameter
            wavelength = float(parts[3])
            
            # Value 5: Coherence flag
            coherence_flag = int(parts[4])
            
            # Value 8: Unix timestamp
            timestamp = int(parts[7])
        else:
            raise ValueError(f"Insufficient values in header line 8: {len(parts)} < 8")
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse header line 8: {e}")
        print(f"Using default values")
        pixel_spacing_um = 0.5  # Default estimate from line 8 index 1
        wavelength = None
        coherence_flag = None
        timestamp = None
    
    # Read the data (re-opening file to read after header)
    with open(filepath, 'r') as f:
        # Skip header
        for _ in range(14):
            f.readline()
            
        # Parse data lines
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 3:
                continue
            
            x = int(parts[0])
            y = int(parts[1])
            
            # Check if data is valid
            if parts[2] == "No" and len(parts) > 3 and parts[3] == "Data":
                # No data point
                continue
            else:
                try:
                    # User specified: z column values are already in microns
                    z_microns = float(parts[2])
                    if 0 <= y < height and 0 <= x < width:
                        full_data[y, x] = z_microns
                except (ValueError, IndexError):
                    continue
    
    # Downsample if requested
    if resolution_factor > 1:
        # Use mean pooling for downsampling (ignoring NaN values)
        new_height = height // resolution_factor
        new_width = width // resolution_factor
        data = np.full((new_height, new_width), np.nan)
        
        for i in range(new_height):
            for j in range(new_width):
                block = full_data[
                    i*resolution_factor:(i+1)*resolution_factor,
                    j*resolution_factor:(j+1)*resolution_factor
                ]
                # Calculate mean ignoring NaN
                if not np.all(np.isnan(block)):
                    data[i, j] = np.nanmean(block)
        
        # Adjust pixel spacing for downsampled data
        pixel_spacing_um *= resolution_factor
    else:
        data = full_data
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Pixel spacing: {pixel_spacing_um:.6f} µm/pixel")
    print(f"Array size: {width}x{height}")
    if wavelength is not None:
        print(f"Wavelength parameter: {wavelength}")
    if coherence_flag is not None:
        print(f"Coherence flag: {coherence_flag}")
    
    # Extract metadata
    metadata = {
        'header': header_lines,
        'filepath': str(filepath),
        'resolution_factor': resolution_factor,
        'original_size': (width, height),
        'processed_size': data.shape,
        'load_time': load_time,
        'pixel_spacing_um': pixel_spacing_um,
        'wavelength': wavelength,
        'coherence_flag': coherence_flag,
        'timestamp': timestamp
    }
    
    return data, metadata


def export_roughness_to_obj(roughness, pixel_spacing_um, output_path):
    """
    Export roughness data to Wavefront OBJ format for Blender import.
    
    Args:
        roughness: 2D numpy array of roughness values (in microns)
        pixel_spacing_um: Pixel spacing in microns
        output_path: Path to output OBJ file
    """
    print(f"\nExporting roughness to OBJ format...")
    start_time = time.time()
    
    height, width = roughness.shape
    
    # Scale factor for better visualization in Blender (convert µm to mm)
    scale_xy = pixel_spacing_um / 1000.0  # Convert to mm
    scale_z = 1.0 / 1000.0  # Convert µm to mm
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Wavefront OBJ file - Roughness Surface\n")
        f.write(f"# Generated from optical profilometry data\n")
        f.write(f"# Dimensions: {width} x {height} pixels\n")
        f.write(f"# Pixel spacing: {pixel_spacing_um:.3f} µm\n")
        f.write(f"# Units: millimeters (X, Y, Z)\n")
        f.write(f"# Z values represent roughness component\n\n")
        
        # Write vertices
        f.write("# Vertices\n")
        for y in range(height):
            for x in range(width):
                z_val = roughness[y, x]
                if np.isnan(z_val):
                    z_val = 0.0  # Replace NaN with 0
                
                # OBJ coordinates: X, Y, Z in mm
                x_mm = x * scale_xy
                y_mm = y * scale_xy
                z_mm = z_val * scale_z
                
                f.write(f"v {x_mm:.6f} {y_mm:.6f} {z_mm:.6f}\n")
        
        # Write faces (triangles)
        f.write("\n# Faces\n")
        for y in range(height - 1):
            for x in range(width - 1):
                # Vertex indices (OBJ uses 1-based indexing)
                v1 = y * width + x + 1
                v2 = y * width + (x + 1) + 1
                v3 = (y + 1) * width + x + 1
                v4 = (y + 1) * width + (x + 1) + 1
                
                # Create two triangles for each quad
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v2} {v4} {v3}\n")
    
    export_time = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  OBJ export completed in {export_time:.2f} seconds")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Vertices: {height * width:,}")
    print(f"  Faces: {2 * (height - 1) * (width - 1):,}")
    print(f"  Saved to: {output_path}")


def interpolate_nans(data, method='bilinear'):
    """
    Interpolate NaN values in the data array.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array with NaN values to interpolate
    method : str
        Interpolation method: 'bilinear', 'laplacian', or 'kriging'
        
    Returns:
    --------
    interpolated : np.ndarray
        Data with NaN values filled
    """
    if not np.any(np.isnan(data)):
        print("No NaN values to interpolate")
        return data.copy()
    
    print(f"Interpolating NaN values using {method} method...")
    start_time = time.time()
    
    result = data.copy()
    nan_mask = np.isnan(data)
    
    if method == 'bilinear':
        # Use scipy's griddata with linear interpolation
        # Get coordinates of valid points
        valid_mask = ~nan_mask
        y_valid, x_valid = np.where(valid_mask)
        z_valid = data[valid_mask]
        
        # Get coordinates of NaN points
        y_nan, x_nan = np.where(nan_mask)
        
        if len(y_nan) > 0 and len(y_valid) > 0:
            # Interpolate
            points = np.column_stack([y_valid, x_valid])
            values = z_valid
            xi = np.column_stack([y_nan, x_nan])
            
            # Use linear interpolation with nearest for extrapolation (edges/corners)
            z_interp = griddata(points, values, xi, method='linear', fill_value=np.nan)
            
            # Fill remaining NaNs (at edges) with nearest neighbor
            still_nan = np.isnan(z_interp)
            if np.any(still_nan):
                z_nearest = griddata(points, values, xi[still_nan], method='nearest')
                z_interp[still_nan] = z_nearest
            
            result[nan_mask] = z_interp
    
    elif method == 'laplacian':
        # Iterative Laplacian interpolation (solves Laplace equation)
        # This naturally handles edges well
        max_iterations = 1000
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            old_result = result.copy()
            
            # For each NaN point, replace with average of neighbors
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if nan_mask[i, j]:
                        # Get valid neighbors
                        neighbors = []
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1]:
                                if not np.isnan(result[ni, nj]):
                                    neighbors.append(result[ni, nj])
                        
                        if neighbors:
                            result[i, j] = np.mean(neighbors)
            
            # Check convergence
            change = np.nanmax(np.abs(result - old_result))
            if change < tolerance:
                break
        
        print(f"  Laplacian converged in {iteration + 1} iterations")
    
    elif method == 'kriging':
        # Use RBF (Radial Basis Function) interpolation as a kriging approximation
        # This is computationally expensive but handles edges well
        valid_mask = ~nan_mask
        y_valid, x_valid = np.where(valid_mask)
        z_valid = data[valid_mask]
        
        # Limit number of points for performance
        max_points = 5000
        if len(y_valid) > max_points:
            # Randomly sample points
            indices = np.random.choice(len(y_valid), max_points, replace=False)
            y_valid = y_valid[indices]
            x_valid = x_valid[indices]
            z_valid = z_valid[indices]
        
        y_nan, x_nan = np.where(nan_mask)
        
        if len(y_nan) > 0 and len(y_valid) > 0:
            # Use RBF interpolation
            points = np.column_stack([y_valid, x_valid])
            xi = np.column_stack([y_nan, x_nan])
            
            # Use thin-plate spline kernel (good for smooth surfaces)
            rbf = RBFInterpolator(points, z_valid, kernel='thin_plate_spline', smoothing=0.1)
            z_interp = rbf(xi)
            
            result[nan_mask] = z_interp
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    interp_time = time.time() - start_time
    print(f"  Interpolation completed in {interp_time:.2f} seconds")
    
    return result


def decompose_surface(data, pixel_spacing_um):
    """
    Decompose surface into form, waviness, and roughness components.
    
    Parameters:
    -----------
    data : np.ndarray
        2D height data
    pixel_spacing_um : float
        Pixel spacing in microns
        
    Returns:
    --------
    form : np.ndarray
        Form (large-scale shape) - polynomial fit
    waviness : np.ndarray
        Waviness (medium-scale) - Gaussian filtered
    roughness : np.ndarray
        Roughness (fine-scale) - residual
    """
    print("Decomposing surface into form, waviness, and roughness...")
    
    # Handle NaN values by using only valid data
    valid_mask = ~np.isnan(data)
    
    # 1. Form: Fit a 2nd order polynomial surface (large-scale shape)
    y_coords, x_coords = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    
    # Flatten and get valid points
    x_flat = x_coords[valid_mask]
    y_flat = y_coords[valid_mask]
    z_flat = data[valid_mask]
    
    # Fit polynomial: z = a + bx + cy + dxx + eyy + fxy
    if len(z_flat) > 6:
        A = np.column_stack([
            np.ones_like(x_flat),
            x_flat, y_flat,
            x_flat**2, y_flat**2,
            x_flat * y_flat
        ])
        
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
        
        # Evaluate polynomial on full grid
        A_full = np.column_stack([
            np.ones(x_coords.size),
            x_coords.ravel(), y_coords.ravel(),
            x_coords.ravel()**2, y_coords.ravel()**2,
            x_coords.ravel() * y_coords.ravel()
        ])
        
        form = (A_full @ coeffs).reshape(data.shape)
    else:
        form = np.full_like(data, np.nanmean(data))
    
    # 2. Remove form to get residual
    residual = data - form
    
    # 3. Waviness: Gaussian filter of residual
    # Cutoff wavelength for waviness: typically 0.8mm for surface analysis
    # Convert to pixels
    cutoff_wavelength_um = 800  # 0.8mm
    sigma_pixels = cutoff_wavelength_um / pixel_spacing_um / (2 * np.pi)
    
    # Apply Gaussian filter (handles NaN by replacing with mean temporarily)
    residual_filled = residual.copy()
    residual_filled[np.isnan(residual_filled)] = np.nanmean(residual_filled)
    waviness = gaussian_filter(residual_filled, sigma=sigma_pixels)
    
    # 4. Roughness: Residual after removing waviness
    roughness = residual - waviness
    
    return form, waviness, roughness


def compute_statistics(data):
    """
    Compute comprehensive statistics for the height data.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of height values
        
    Returns:
    --------
    stats : dict
        Dictionary containing statistical measures
    """
    print("\nComputing statistics...")
    
    # Mask for valid (non-NaN) data
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    
    total_points = data.size
    valid_points = valid_data.size
    missing_points = total_points - valid_points
    
    stats = {
        'total_points': total_points,
        'valid_points': valid_points,
        'missing_points': missing_points,
        'coverage_percent': (valid_points / total_points) * 100,
        'min': np.min(valid_data) if valid_points > 0 else np.nan,
        'max': np.max(valid_data) if valid_points > 0 else np.nan,
        'mean': np.mean(valid_data) if valid_points > 0 else np.nan,
        'median': np.median(valid_data) if valid_points > 0 else np.nan,
        'std': np.std(valid_data) if valid_points > 0 else np.nan,
        'range': np.ptp(valid_data) if valid_points > 0 else np.nan,
        'percentile_25': np.percentile(valid_data, 25) if valid_points > 0 else np.nan,
        'percentile_75': np.percentile(valid_data, 75) if valid_points > 0 else np.nan,
        'rms': np.sqrt(np.mean(valid_data**2)) if valid_points > 0 else np.nan,
    }
    
    # Surface roughness parameters (common in profilometry)
    if valid_points > 0:
        # Ra: Average roughness
        stats['Ra'] = np.mean(np.abs(valid_data - stats['mean']))
        
        # Rq: RMS roughness
        stats['Rq'] = np.sqrt(np.mean((valid_data - stats['mean'])**2))
        
        # Rz: Ten-point height (simplified as max-min)
        stats['Rz'] = stats['range']
    
    return stats


def print_statistics(stats):
    """Print statistics in a formatted way."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    print(f"\nData Coverage:")
    print(f"  Total points:   {stats['total_points']:,}")
    print(f"  Valid points:   {stats['valid_points']:,}")
    print(f"  Missing points: {stats['missing_points']:,}")
    print(f"  Coverage:       {stats['coverage_percent']:.2f}%")
    
    print(f"\nHeight Statistics (units from file):")
    print(f"  Min:            {stats['min']:.6f}")
    print(f"  Max:            {stats['max']:.6f}")
    print(f"  Range:          {stats['range']:.6f}")
    print(f"  Mean:           {stats['mean']:.6f}")
    print(f"  Median:         {stats['median']:.6f}")
    print(f"  Std Dev:        {stats['std']:.6f}")
    print(f"  25th percentile:{stats['percentile_25']:.6f}")
    print(f"  75th percentile:{stats['percentile_75']:.6f}")
    
    print(f"\nSurface Roughness Parameters:")
    print(f"  Ra (avg roughness):     {stats['Ra']:.6f}")
    print(f"  Rq (RMS roughness):     {stats['Rq']:.6f}")
    print(f"  Rz (max height):        {stats['Rz']:.6f}")
    
    print("="*60 + "\n")


def create_visualizations(data, metadata, stats, output_dir=None, original_data=None):
    """
    Create comprehensive visualizations of the profilometry data.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of height values
    metadata : dict
        File metadata
    stats : dict
        Statistical measures
    output_dir : str or Path, optional
        Directory to save figures. If None, displays interactively.
    original_data : np.ndarray, optional
        Original data before interpolation (for coverage map)
    """
    print("Creating visualizations...")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get pixel spacing
    pixel_spacing_um = metadata['pixel_spacing_um']
    
    # Create extent for imshow (in microns)
    extent_um = [0, data.shape[1] * pixel_spacing_um, 
                 0, data.shape[0] * pixel_spacing_um]
    
    # Decompose surface into form, waviness, and roughness
    form, waviness, roughness = decompose_surface(data, pixel_spacing_um)
    
    # Create figure with 3x3 subplots
    fig = plt.figure(figsize=(20, 18))
    
    # Row 1: Original data analysis
    # 1. 2D Height Map
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(data, cmap='viridis', origin='lower', interpolation='nearest',
                     extent=extent_um)
    ax1.set_title('2D Height Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position (µm)')
    ax1.set_ylabel('Y Position (µm)')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Height (µm)')
    
    # 2. Roughness Distribution (changed from height distribution)
    ax2 = plt.subplot(3, 3, 2)
    valid_roughness = roughness[~np.isnan(roughness)]
    ax2.hist(valid_roughness, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.nanmean(roughness), color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {np.nanmean(roughness):.2f} µm")
    ax2.axvline(np.nanmedian(roughness), color='green', linestyle='--', linewidth=2, 
                label=f"Median: {np.nanmedian(roughness):.2f} µm")
    ax2.set_title('Roughness Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Roughness (µm)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Data Coverage Map (before interpolation)
    ax3 = plt.subplot(3, 3, 3)
    # Use original data before interpolation for coverage map
    data_for_coverage = original_data if original_data is not None else data
    coverage_map = ~np.isnan(data_for_coverage)
    im3 = ax3.imshow(coverage_map, cmap='RdYlGn', origin='lower', interpolation='nearest',
                     extent=extent_um)
    ax3.set_title(f'Data Coverage (Before Interpolation) ({stats["coverage_percent"]:.1f}%)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Position (µm)')
    ax3.set_ylabel('Y Position (µm)')
    plt.colorbar(im3, ax=ax3, label='Valid Data', ticks=[0, 1])
    
    # Row 2: 3D and profiles
    # 4. 3D Surface Plot - Original Height Map
    ax4 = plt.subplot(3, 3, 4, projection='3d')
    
    # Downsample for 3D plot if data is large
    plot_factor = max(1, data.shape[0] // 200)
    if plot_factor > 1:
        plot_data = data[::plot_factor, ::plot_factor]
        plot_spacing = pixel_spacing_um * plot_factor
    else:
        plot_data = data
        plot_spacing = pixel_spacing_um
    
    x_um = np.arange(plot_data.shape[1]) * plot_spacing
    y_um = np.arange(plot_data.shape[0]) * plot_spacing
    X, Y = np.meshgrid(x_um, y_um)
    
    # Create surface plot of original height data
    surf = ax4.plot_surface(X, Y, plot_data, cmap='viridis', 
                           linewidth=0, antialiased=True, alpha=0.9)
    ax4.set_title('3D Height Map', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X Position (µm)')
    ax4.set_ylabel('Y Position (µm)')
    ax4.set_zlabel('Height (µm)')
    ax4.view_init(elev=30, azim=45)  # 45 degree azimuth angle
    
    # 5. Cross-section profiles - ROUGHNESS
    ax5 = plt.subplot(3, 3, 5)
    
    # Horizontal cross-section at middle
    mid_y = roughness.shape[0] // 2
    h_profile = roughness[mid_y, :]
    x_positions_um = np.arange(len(h_profile)) * pixel_spacing_um
    ax5.plot(x_positions_um, h_profile, label=f'Horizontal (Y={mid_y*pixel_spacing_um:.0f} µm)', 
             linewidth=2, alpha=0.7)
    
    # Vertical cross-section at middle
    mid_x = roughness.shape[1] // 2
    v_profile = roughness[:, mid_x]
    y_positions_um = np.arange(len(v_profile)) * pixel_spacing_um
    ax5.plot(y_positions_um, v_profile, label=f'Vertical (X={mid_x*pixel_spacing_um:.0f} µm)', 
             linewidth=2, alpha=0.7)
    
    ax5.set_title('Roughness Cross-Section Profiles', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Position (µm)')
    ax5.set_ylabel('Roughness (µm)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Gradient/Slope Map
    ax6 = plt.subplot(3, 3, 6)
    
    # Compute gradient magnitude (accounting for pixel spacing)
    gy, gx = np.gradient(np.nan_to_num(data, nan=0))
    # Convert to slope (dimensionless)
    gradient_mag = np.sqrt(gx**2 + gy**2) / pixel_spacing_um
    gradient_mag[np.isnan(data)] = np.nan
    
    im6 = ax6.imshow(gradient_mag, cmap='hot', origin='lower', interpolation='nearest',
                     extent=extent_um)
    ax6.set_title('Surface Gradient Magnitude', fontsize=14, fontweight='bold')
    ax6.set_xlabel('X Position (µm)')
    ax6.set_ylabel('Y Position (µm)')
    plt.colorbar(im6, ax=ax6, label='Gradient (µm/µm)')
    
    # Row 3: Form, Waviness, Roughness decomposition
    # 7. Form (large-scale shape)
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.imshow(form, cmap='coolwarm', origin='lower', interpolation='nearest',
                     extent=extent_um)
    ax7.set_title('Form (Large-Scale Shape)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('X Position (µm)')
    ax7.set_ylabel('Y Position (µm)')
    plt.colorbar(im7, ax=ax7, label='Height (µm)')
    
    # 8. Waviness (medium-scale)
    ax8 = plt.subplot(3, 3, 8)
    im8 = ax8.imshow(waviness, cmap='seismic', origin='lower', interpolation='nearest',
                     extent=extent_um, vmin=-np.nanstd(waviness)*3, vmax=np.nanstd(waviness)*3)
    ax8.set_title('Waviness (Medium-Scale)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('X Position (µm)')
    ax8.set_ylabel('Y Position (µm)')
    plt.colorbar(im8, ax=ax8, label='Height (µm)')
    
    # 9. Roughness (fine-scale)
    ax9 = plt.subplot(3, 3, 9)
    im9 = ax9.imshow(roughness, cmap='RdBu_r', origin='lower', interpolation='nearest',
                     extent=extent_um, vmin=-np.nanstd(roughness)*3, vmax=np.nanstd(roughness)*3)
    ax9.set_title('Roughness (Fine-Scale)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('X Position (µm)')
    ax9.set_ylabel('Y Position (µm)')
    plt.colorbar(im9, ax=ax9, label='Height (µm)')
    
    # Add overall title with metadata
    filename = Path(metadata['filepath']).name
    fig.suptitle(f'Optical Profilometry Analysis: {filename}\n' + 
                 f'Resolution: {data.shape[0]}x{data.shape[1]} pixels ' +
                 f'({data.shape[0]*pixel_spacing_um:.0f}×{data.shape[1]*pixel_spacing_um:.0f} µm, ' +
                 f'factor: {metadata["resolution_factor"]}x)',
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save or show
    if output_dir:
        output_file = output_dir / f"{Path(metadata['filepath']).stem}_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize optical profilometry XYZ data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze at full resolution
  python analyze_profilometry.py data.xyz
  
  # Analyze with 4x downsampling for faster processing
  python analyze_profilometry.py data.xyz -r 4
  
  # Interpolate missing data with bilinear method
  python analyze_profilometry.py data.xyz -r 4 --interpolate bilinear
  
  # Use Laplacian interpolation (good for edges)
  python analyze_profilometry.py data.xyz -r 4 --interpolate laplacian
  
  # Use kriging interpolation (slower but smooth)
  python analyze_profilometry.py data.xyz -r 4 --interpolate kriging
  
  # Save outputs to a directory
  python analyze_profilometry.py data.xyz -r 2 -o results/
  
  # Process without showing plots (save only)
  python analyze_profilometry.py data.xyz -o results/ --no-display
        """
    )
    
    parser.add_argument('input_file', type=str,
                       help='Path to XYZ profilometry data file')
    parser.add_argument('-r', '--resolution-factor', type=int, default=1,
                       choices=[1, 2, 4, 8, 16, 32],
                       help='Resolution reduction factor for faster processing (default: 1)')
    parser.add_argument('-i', '--interpolate', type=str, default=None,
                       choices=['bilinear', 'laplacian', 'kriging'],
                       help='Interpolate NaN values using specified method (default: None)')
    parser.add_argument('--export-obj', action='store_true',
                       help='Export roughness map to OBJ file for Blender import')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                       help='Directory to save output figures and statistics')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots interactively (only save)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only compute and print statistics, skip visualization')
    parser.add_argument('--bounds', type=float, nargs=4, metavar=('X1', 'X2', 'Y1', 'Y2'),
                       help='Crop image bounds in microns (x1 x2 y1 y2)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    # Load data
    data, metadata = load_xyz_file(input_path, args.resolution_factor)
    
    # Apply bounds cropping if requested
    if args.bounds:
        x1_um, x2_um, y1_um, y2_um = args.bounds
        pixel_spacing_um = metadata['pixel_spacing_um']
        
        # Convert microns to pixel indices
        x1_px = int(x1_um / pixel_spacing_um)
        x2_px = int(x2_um / pixel_spacing_um)
        y1_px = int(y1_um / pixel_spacing_um)
        y2_px = int(y2_um / pixel_spacing_um)
        
        # Validate bounds
        if x1_px < 0 or x2_px > data.shape[1] or y1_px < 0 or y2_px > data.shape[0]:
            print(f"Warning: Bounds exceed data dimensions. Data size: {data.shape[1]*pixel_spacing_um:.0f}x{data.shape[0]*pixel_spacing_um:.0f} µm")
            print(f"Requested bounds: x=[{x1_um:.0f}, {x2_um:.0f}] µm, y=[{y1_um:.0f}, {y2_um:.0f}] µm")
        if x1_px >= x2_px or y1_px >= y2_px:
            print(f"Error: Invalid bounds. x1 must be < x2 and y1 must be < y2")
            return 1
        
        # Crop the data
        data = data[y1_px:y2_px, x1_px:x2_px]
        print(f"Cropped to bounds: x=[{x1_um:.0f}, {x2_um:.0f}] µm, y=[{y1_um:.0f}, {y2_um:.0f}] µm")
        print(f"Cropped size: {data.shape[1]}x{data.shape[0]} pixels")
        metadata['bounds'] = (x1_um, x2_um, y1_um, y2_um)
        metadata['bounds_px'] = (x1_px, x2_px, y1_px, y2_px)
    
    # Store original data before interpolation (for coverage map)
    original_data = data.copy()
    
    # Apply interpolation if requested
    if args.interpolate:
        data = interpolate_nans(data, method=args.interpolate)
        metadata['interpolation_method'] = args.interpolate
    else:
        metadata['interpolation_method'] = 'None'
    
    # Compute statistics
    stats = compute_statistics(data)
    print_statistics(stats)
    
    # Save statistics to file if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = output_dir / f"{input_path.stem}_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Optical Profilometry Analysis\n")
            f.write(f"File: {metadata['filepath']}\n")
            f.write(f"Resolution Factor: {metadata['resolution_factor']}x\n")
            f.write(f"Interpolation Method: {metadata['interpolation_method']}\n")
            f.write(f"Processed Size: {metadata['processed_size']}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"HEADER METADATA\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Lateral Sampling: {metadata['pixel_spacing_um']:.3f} µm/pixel\n")
            f.write(f"Lateral Sampling: {metadata['pixel_spacing_um']:.6f} µm/pixel\n")
            f.write(f"Original Dimensions: {metadata['original_size'][0]}x{metadata['original_size'][1]}\n")
            if metadata['wavelength'] is not None:
                f.write(f"Wavelength Parameter: {metadata['wavelength']}\n")
            if metadata['coherence_flag'] is not None:
                f.write(f"Coherence Flag: {metadata['coherence_flag']}\n")
            if metadata['timestamp'] is not None:
                from datetime import datetime
                dt = datetime.fromtimestamp(metadata['timestamp'])
                f.write(f"Acquisition Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"STATISTICAL ANALYSIS\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Processed Size: {metadata['processed_size']}\n")
            f.write(f"Data Coverage:\n")
            f.write(f"  Total points:   {stats['total_points']:,}\n")
            f.write(f"  Valid points:   {stats['valid_points']:,}\n")
            f.write(f"  Missing points: {stats['missing_points']:,}\n")
            f.write(f"  Coverage:       {stats['coverage_percent']:.2f}%\n\n")
            
            f.write(f"Height Statistics (µm):\n")
            f.write(f"  Min:            {stats['min']:.6f}\n")
            f.write(f"  Max:            {stats['max']:.6f}\n")
            f.write(f"  Range:          {stats['range']:.6f}\n")
            f.write(f"  Mean:           {stats['mean']:.6f}\n")
            f.write(f"  Median:         {stats['median']:.6f}\n")
            f.write(f"  Std Dev:        {stats['std']:.6f}\n\n")
            
            f.write(f"Surface Roughness Parameters (µm):\n")
            f.write(f"  Ra (avg roughness):     {stats['Ra']:.6f}\n")
            f.write(f"  Rq (RMS roughness):     {stats['Rq']:.6f}\n")
            f.write(f"  Rz (max height):        {stats['Rz']:.6f}\n")
        
        print(f"Statistics saved to: {stats_file}")
    
    # Create visualizations unless stats-only mode
    if not args.stats_only:
        output_dir = args.output_dir if args.output_dir or args.no_display else None
        create_visualizations(data, metadata, stats, output_dir, original_data=original_data)
        
        if not args.no_display and not args.output_dir:
            print("\nDisplaying interactive plots...")
    
    # Export roughness to OBJ if requested
    if args.export_obj:
        # Compute roughness if not already done (stats_only mode)
        if args.stats_only:
            from scipy.ndimage import gaussian_filter
            # Quick decomposition for export only
            form, waviness, roughness = decompose_surface(data, metadata['pixel_spacing_um'])
        else:
            # Reuse decomposition from visualization
            form, waviness, roughness = decompose_surface(data, metadata['pixel_spacing_um'])
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path.parent
        
        obj_file = output_dir / f"{input_path.stem}_roughness.obj"
        export_roughness_to_obj(roughness, metadata['pixel_spacing_um'], obj_file)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == '__main__':
    exit(main())
