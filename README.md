# Optical Profilometry Analysis Pipeline

This directory contains a python pipeline for analyzing and visualizing optical profilometry .xyz data.

---

## Example exports: 

### Cross-hatch surface
![crosshatch](exports/profilometery_workup_v1/crosshatch_1.png)

---

### 3D rendering
![fresnel render](exports/fresnel-render.png)

---

## Files

- **`analyze_profilometry.py`** - Main analysis script
- **`ceramics/`** - Directory containing XYZ data files (1024×1024 height measurements)

## Requirements

```bash
pip install numpy matplotlib scipy
```

## How it works

### 1. **NaN Interpolation Methods**
- **Bilinear**: Fast 2D linear interpolation (fastest)
- **Laplacian**: Laplace equation smooth interpolation (slow)
- **Kriging**: RBF-based smooth interpolation (slowest)

### 2. **Surface Decomposition**

The surface decomposition separates the measured height map into three physically meaningful components that correspond to different spatial frequency regimes:

- **Form**: Large-scale shape (polynomial fit)
- **Waviness**: Medium-scale features (Gaussian filtered, ~0.8mm cutoff)
- **Roughness**: Fine-scale texture (residual after form & waviness removal)

#### Implementation Details and Design Choices

The decomposition is performed by the `decompose_surface()` function in three sequential steps:

##### Step 1: Form Extraction (Large-Scale Shape)

**Code:**
```python
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
```

**Design Choice:** 2nd-order polynomial fit  
**Alternative Approaches:**
- **1st-order (planar) fit**: Simpler, removes only tilt and piston. Use for nearly flat samples.
  - *When to use*: If you know the sample is intentionally flat (e.g., polished wafer) and only mounting tilt needs removal
- **3rd or higher-order fit**: Captures more complex curvature. Use for intentionally curved samples.
  - *When to use*: If the sample has known bowl-shaped deformation or complex macroscopic curvature
- **Spline surface fit**: Very flexible, can model complex long-wavelength distortions
  - *When to use*: For samples with systematic but non-polynomial distortion (e.g., thermal warping)

**Sample Knowledge Influence:**  
If you know the sample was mounted on a curved stage or has intentional macro-scale curvature from processing, you may want a higher polynomial order. For ceramic samples with only mounting tilt, 2nd-order is appropriate and prevents overfitting.

##### Step 2: Waviness Extraction (Medium-Scale Features)

**Code:**
```python
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
```

**Design Choice:** 0.8mm Gaussian filter cutoff  
**Alternative Approaches:**
- **ISO 4287/4288 standard filters**: Use standardized cutoffs (0.08mm, 0.25mm, 0.8mm, 2.5mm, 8mm)
  - *When to use*: For comparing to published metrology standards or industry specs
- **Smaller cutoff (e.g., 0.25mm)**: Moves more features into the "roughness" category
  - *When to use*: If you know the relevant surface features are smaller (e.g., fine machining marks)
- **Larger cutoff (e.g., 2.5mm)**: Moves more features into "waviness"
  - *When to use*: If you're interested in larger-scale periodic patterns (e.g., wide laser scan lines)
- **Band-pass filtering**: Extract a specific wavelength range rather than low-pass
  - *When to use*: Isolating periodic structures with known spatial frequency
- **Median filter instead of Gaussian**: More robust to outliers
  - *When to use*: If your data has spikes or contamination that shouldn't influence waviness

**Sample Knowledge Influence:**  
For laser-etched ceramics with ~100µm line spacing, the 0.8mm cutoff is well above the feature size, so the periodic pattern appears in "waviness". If you're studying finer features (sub-100µm grain structure), consider 0.25mm. If the relevant manufacturing defects are at larger scales, use 2.5mm.

##### Step 3: Roughness Calculation (Fine-Scale Texture)

**Code:**
```python
# 4. Roughness: Residual after removing waviness
roughness = residual - waviness
```

**Design Choice:** Simple subtraction (high-pass filtering)  
**Alternative Approaches:**
- **RMS-based normalization**: Scale roughness by local waviness amplitude
  - *When to use*: If roughness magnitude varies systematically across the sample
- **Detrended Fluctuation Analysis (DFA)**: Remove local trends at multiple scales
  - *When to use*: For fractal or self-affine surfaces where scale-dependent analysis is needed
- **Wavelet decomposition**: Multi-resolution analysis that separates scales more flexibly
  - *When to use*: When you need to analyze roughness at multiple independent length scales simultaneously

**Sample Knowledge Influence:**  
If you know the ceramic processing creates roughness that scales with local geometry (e.g., rougher in valleys), consider normalized roughness. For most optical profilometry QC work, the simple residual is interpretable and standard.

#### Summary of Key Decision Points

1. **Polynomial order for form** → Depends on sample flatness and mounting
2. **Waviness cutoff wavelength** → Should be larger than features of interest, smaller than sample size
3. **Filter type (Gaussian vs. median vs. ISO)** → Gaussian is standard; median for noisy data; ISO for regulatory compliance
4. **Roughness normalization** → Raw residual is typical; normalize if roughness varies spatially with known causes

### 3. **Downsampling**
Optionally resolution for faster processing (2x, 4x, 8x, 16x, 32x).  Also optionally skip the visualizer rendering

### 4. **Export to Blender**
Optionally export roughness map as OBJ file for 3D visualization in Blender

## Command-Line Options

```
usage: analyze_profilometry.py [-h] [-r {1,2,4,8,16,32}] 
                               [-i {bilinear,laplacian,kriging}]
                               [--export-obj] [-o OUTPUT_DIR]
                               [--no-display] [--stats-only]
                               [--bounds X1 X2 Y1 Y2]
                               input_file

positional arguments:
  input_file            Path to XYZ profilometry data file

optional arguments:
  -h, --help            Show help message
  -r, --resolution-factor {1,2,4,8,16,32}
                        Resolution reduction factor for faster processing (default: 1)
  -i, --interpolate {bilinear,laplacian,kriging}
                        Interpolate NaN values using specified method (default: None)
  --export-obj          Export roughness map to OBJ file for Blender import
  -o, --output-dir OUTPUT_DIR
                        Directory to save output figures and statistics
  --no-display          Do not display plots interactively (only save)
  --stats-only          Only compute and print statistics, skip visualization
  --bounds X1 X2 Y1 Y2  Crop image bounds in microns (x1 x2 y1 y2)
```

## Example Usage

```bash
# Full resolution analysis, default to bilinear inerpolation
py analyze_profilometry.py heightmaps/PCD_01mm_2.75x_05x_001.xyz

# Kriging interpolation, 4x downsampling
py analyze_profilometry.py heightmaps/PCD_01mm_2.75x_05x_001.xyz -r 4 -i kriging

# Save visualizations and statistics to 'results' folder
py analyze_profilometry.py heightmaps/PCD_01mm_2.75x_05x_001.xyz -r 4 -i bilinear -o results/

# Just compute and print statistics
py analyze_profilometry.py heightmaps/PCD_01mm_2.75x_05x_001.xyz -r 4 --stats-only

# Export roughness map as OBJ file for 3D visualization in Blender
py analyze_profilometry.py heightmaps/PCD_01mm_2.75x_05x_001.xyz -r 4 -i bilinear --export-obj -o results/

# Crop to a specific region (from 100 to 400 microns in x, 200 to 500 microns in y)
py analyze_profilometry.py heightmaps/PCD_01mm_2.75x_05x_001.xyz --bounds 100 400 200 500

# Windows PowerShell Batch Processing
Get-ChildItem heightmaps\*.xyz | ForEach-Object { py analyze_profilometry.py $_.FullName -r 4 -i bilinear -o results/ --no-display }
```
## Data Format

The XYZ files have the following structure:
- **Header**: 14 lines of metadata
- **Data**: Lines with format `X Y Z`, where some Z are NaN
  - X, Y: Integer coordinates (0-1023)
    - it is not clear as of this commit (2/10/26) how to translate this into physical distance.  The values that result from the header scalar do not match intuition (the line spacing should be ~100 microns)
  - Z: Height value
    - it is not yet totally clear as of this commit (2/10/26) if these values are in um or if the header contains the scalar from meters which would differe by a factor of ~5

## Interpolation Methods for NaN Points

- Bilinear
- Laplacian
- Kriging

## Downsampling and other performance flags

- Use `-r 4` or `-r 8` for initial exploration (much faster)
- Use `-r 1` (full resolution) only when you need detailed analysis
- Use `--stats-only` if you only need numerical statistics
- Use `-o` with `--no-display` for batch processing without GUI
- Bilinear interpolation is fastest; kriging is slowest but smoothest
- Pixel spacing is automatically extracted from file header

---

# interpolation methods exploration

## gaussian random

![alt text](exports/interpolation-methods/comparison_gaussian_random.png)

## gaussian scratch

![alt text](exports/interpolation-methods/comparison_gaussian_scratch.png)

---

## poisson random

![alt text](exports/interpolation-methods/comparison_poisson_random.png)

## poisson scratch

![alt text](exports/interpolation-methods/comparison_poisson_scratch.png)

---

## speckle random

![alt text](exports/interpolation-methods/comparison_speckle_random.png)

## speckle scratch

![alt text](exports/interpolation-methods/comparison_speckle_scratch.png)

---

## Notes

- Interpolation fills NaN values *before* surface decomposition and visualization
- Downsampling uses mean pooling (averaging valid points in each block)
- Statistics should be computed only on valid (non-NaN) data points (**double check this is accurate**)
- Pixel spacing and vertical scale factor extracted from header line 8 (**still figuring out if this is correct**)
- A pixel size of ~0.5 um puts the Nyquist resolution at ~1 um.  
- Surface decomposition:
  - **form:** 2nd order polynomial
  - **waviness:** Gaussian filter with 0.8mm cutoff (after subtracting form)
  - **roughness:** Residual (after removing form and waviness)

# Current Data

## Standard - 100 um scan interval (1)

![alt text](exports/profilometery_workup_v1/standard_1.png)

## Standard - 100 um scan interval (2)

![alt text](exports/profilometery_workup_v1/standard_2.png)

## Standard - 100 um scan interval (3)

![alt text](exports/profilometery_workup_v1/standard_3.png)

---

## ? (1)

![alt text](exports/profilometery_workup_v1/PCD_01mm_2.75x_05x_001.png)

## ? (2)

![alt text](exports/profilometery_workup_v1/PCD_01mm_2.75x_05x_002.png)

## ? (3)

![alt text](exports/profilometery_workup_v1/PCD_01mm_2.75x_05x_003.png)

---

## Cross-hatch (1)

![alt text](exports/profilometery_workup_v1/crosshatch_1.png)

## Cross-hatch (2)

![alt text](exports/profilometery_workup_v1/crosshatch_2.png)

## Cross-hatch (3)

![alt text](exports/profilometery_workup_v1/crosshatch_3.png)

---

## 2x line density (1)

![2x line density (1)](exports/profilometery_workup_v1/2x-line-density_1.png)

## 2x line density (2)

![2x line density (2)](exports/profilometery_workup_v1/2x-line-density_2.png)

## 2x line density (3)

![2x line density (3)](exports/profilometery_workup_v1/2x-line-density_3.png)

---

# Other notes

## Noise floor

If I'm interpreting the header info correctly (it's unlabeled, so maybe not), the instrument has a noise floor of a few microns, which sets the limit on measurable surface features.  Ra/Rq values should be compared to noise floor for if this is true.

## To-do
- show the data coverage before (maybe also after like this) to better tell whether the interpolation mode was appropriate (Kriging sometimes useful for big gaps in an effort to preserve PSD)
- re-evaluate the xy pixel intervals — the images seem way too big for the units (~50 um lines are like 5 um)
- subtract DC before form?  Should I consider the first big jump in the heights histogram to be the DC offset?
- additional figures
  - 3D zoomed in on a single line trough pair
  - roughness cross-section taken along the center of a scan line and the center of a trough
    - perhaps also averaging a few orthogonal peak-to-valley lines
  - export roughness to something blender compatible (obj? glb?)
- find literature and think on how substantial these surface gradients are (probably most reflective of the "sharp cutoff" of interest)
- find more reliable source for the metadata interpretation and let Jackie and Ben know if their z-scaling has been way off
- accounting for and reporting high frequency regime pitfalls
  - the noise floor
  - sampling/Nyquist — aliasing near $f_{max} = 1/(2\Delta$)
  - instrument's transfer function (objective NA, coherence mode, lateral resolution chaning at high $f$)

### spatial wavelength regimes considerations
- currently just playing with upper and lower bounds for waviness
- consider migrating paradigms
- power spectral density (PSD)
    - detrend/level (remove piston and tilt)
    - window (Hann/Tukey) before FFT to reduce edge leakage
    - compute 2D PSD then radially average to get 1D PSD vs f
    - report band-limited RMS
- band-limited metrics with band-pass filtering
    - FFT + mask or spatial domain equivalents (Gaussian / spline / ISO filters)
- Autocorrelation / structure function (correlation lengths and isotropy)
- Wavelets / multiresolution decomposition
- Directional / oriented analysis

## Resources

instrument
- https://en.wikipedia.org/wiki/White_light_interferometry

interpolation
- https://en.wikipedia.org/wiki/Radial_basis_function_interpolation
- https://en.wikipedia.org/wiki/Kriging

### MSC's Profilometer Description

> #### Zygo Nexview 3D Optical Surface Profiler:
> White Light Interferometry.
> 2.5x, 10x, 20x, 50x objectives with 0.5x, 1x, 2x internal magnification.
> Automated image stitching.
> 200mm XY stage and 100mm Z clearance with capacity for up to 10lbs.
> A white light interferometer is a type of profilometer in which light from a lamp is split into two paths by a beam splitter. One path directs the light onto the surface under test, the other path directs the light to a reference mirror. Reflections from the two surfaces are recombined and projected onto an array detector. When the path difference between the recombined beams is on the order of a few wavelengths of light or less, interference can occur. This interference contains information about the surface contours of the test surface. Vertical resolution can be on the order of several angstroms while lateral resolution depends upon the system and objective and is typically in the range of 0.26um – 4.4um.

---

# Synthetic Interpolation Study

The script `interpolation_study.py` provides a framework for validating interpolation strategies on synthetic 128x128 surfaces. It simulates realistic profilometry artifacts including stepped geometries and various noise distributions.

## Interpolation Methods & Formulae

### 1. Nearest Neighbor
Assigns the value of the visually closest data point.
$$f(x, y) = f(x_i, y_i) \quad \text{where} \quad \sqrt{(x-x_i)^2 + (y-y_i)^2} \text{ is minimized}$$

### 2. Bilinear Interpolation
A linear extension of 1D interpolation, estimating the value based on the four nearest neighbors.
$$f(x, y) \approx a_0 + a_1 x + a_2 y + a_3 xy$$

### 3. Bicubic Interpolation
Uses a third-degree polynomial (cubic spline) for smoother transitions, utilizing a $4 \times 4$ neighborhood.
$$f(x, y) = \sum_{i=0}^3 \sum_{j=0}^3 a_{ij} x^i y^j$$

### 4. Laplacian Diffusion
Solves the steady-state diffusion (Laplace) equation to fill holes, ensuring a "smooth" harmonic transition.
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = 0$$

### 5. Kriging / RBF (Radial Basis Function)
Computes a weighted sum of basis functions centered at valid data points. The script defaults to the **Thin Plate Spline** kernel.
$$f(\mathbf{x}) = \sum_{i=1}^n w_i \phi(\|\mathbf{x} - \mathbf{x}_i\|) + P(\mathbf{x})$$
Where the Thin Plate Spline kernel is defined as:
$$\phi(r) = r^2 \ln(r)$$

## Features
- **Noise Models**: Gaussian (Additive), Poisson (Shot), and Speckle (Multiplicative).
- **Visualization**: Generates `comparison_<noise>_<mask_type>.png` plots showing Ground Truth, Masked Input, Reconstructions, and Difference Maps ($|Original - Reconstructed|$).

## Usage
```bash
py interpolation_study.py
```
