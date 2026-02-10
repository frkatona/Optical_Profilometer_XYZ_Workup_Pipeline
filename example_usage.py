"""
Quick Example: How to use the profilometry analysis script

This demonstrates common usage patterns for analyzing ceramic surface data.
"""

import subprocess
import os
from pathlib import Path

# Get the ceramics directory
ceramics_dir = Path("ceramics")
output_dir = Path("example_results")

print("="*70)
print("OPTICAL PROFILOMETRY ANALYSIS - QUICK START EXAMPLES")
print("="*70)

# Example 1: Quick statistics for one file
print("\n1. Quick Statistics (8x downsampling, stats only)")
print("-" * 70)
cmd = [
    "py", "analyze_profilometry.py",
    str(ceramics_dir / "PCD_01mm_2.75x_05x_001.xyz"),
    "-r", "8",
    "--stats-only"
]
print(f"Command: {' '.join(cmd)}\n")
subprocess.run(cmd)

# Example 2: Full analysis with visualization
print("\n\n2. Full Analysis with Visualization (4x downsampling)")
print("-" * 70)
cmd = [
    "py", "analyze_profilometry.py",
    str(ceramics_dir / "PCD_01mm_2.75x_05x_001.xyz"),
    "-r", "4",
    "-o", str(output_dir)
]
print(f"Command: {' '.join(cmd)}\n")
subprocess.run(cmd)

# Example 3: Compare multiple samples
print("\n\n3. Compare Multiple Samples (statistics only)")
print("-" * 70)
samples = [
    "PCD_01mm_2.75x_05x_001.xyz",
    "PCD_01mm_2.75x_05x_002.xyz",
    "PCD_01mm_2.75x_05x_003.xyz"
]

for sample in samples:
    print(f"\n--- {sample} ---")
    cmd = [
        "py", "analyze_profilometry.py",
        str(ceramics_dir / sample),
        "-r", "8",
        "--stats-only"
    ]
    subprocess.run(cmd)

print("\n" + "="*70)
print("Examples complete! Check the 'example_results' folder for outputs.")
print("="*70)
