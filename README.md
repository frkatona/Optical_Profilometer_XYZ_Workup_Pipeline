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
Automatically separates the surface into:
- **Form**: Large-scale shape (polynomial fit)
- **Waviness**: Medium-scale features (Gaussian filtered, ~0.8mm cutoff)
- **Roughness**: Fine-scale texture (residual after form & waviness removal)

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