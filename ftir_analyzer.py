"""
FTIR Spectroscopy Analyzer
==========================
Converts Absorbance to Transmittance, applies smoothing and baseline correction,
identifies major peaks, and generates publication-quality plots.

Author: AI Spectroscopist
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d
import os
import argparse
import glob
import warnings
warnings.filterwarnings('ignore')

from ftir_database import FUNCTIONAL_GROUPS, COMPOUND_RULES

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
WAVENUMBER_MIN = 400
WAVENUMBER_MAX = 4000
OUTPUT_DPI = 300
SAVGOL_WINDOW = 31       # Must be odd
SAVGOL_POLY_ORDER = 3
PEAK_PROMINENCE = 3.0    # %T prominence threshold for "main" peaks
PEAK_MIN_DISTANCE = 30   # Minimum distance between peaks (in data points)

def load_csv(filepath):
    """Load FTIR CSV with auto-detection of delimiter and decimal format."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip()

    # Detect if European format (semicolon delimiter + comma decimal)
    is_european = ';' in first_line and ',' in first_line.split(';')[0]

    wavenumbers = []
    absorbances = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ';' in line:
            parts = line.split(';')
        elif '\t' in line:
            parts = line.split('\t')
        else:
            # Fallback to comma, but also handle cases where it might be space-separated
            parts = line.split(',')
            if len(parts) < 2:
                parts = line.split()

        if len(parts) < 2:
            continue

        try:
            wn_str = parts[0].strip()
            abs_str = parts[1].strip()

            if is_european:
                wn_str = wn_str.replace(',', '.')
                abs_str = abs_str.replace(',', '.')

            wavenumbers.append(float(wn_str))
            absorbances.append(float(abs_str))
        except ValueError:
            continue

    df = pd.DataFrame({'wavenumber': wavenumbers, 'y_value': absorbances})
    df = df.sort_values('wavenumber').reset_index(drop=True)
    return df


def filter_range(df, wn_min=WAVENUMBER_MIN, wn_max=WAVENUMBER_MAX):
    """Keep only data within the standard IR range."""
    mask = (df['wavenumber'] >= wn_min) & (df['wavenumber'] <= wn_max)
    return df[mask].reset_index(drop=True)


def clean_absorbance(df):
    """
    Remove noise / saturated regions.
    Absorbance values >= 4.0 are considered saturated (detector limit).
    Negative absorbance values are clipped to 0.
    """
    df = df.copy()
    df['absorbance'] = df['absorbance'].clip(lower=0, upper=3.5)
    return df


def convert_to_transmittance(absorbance):
    """Convert Absorbance to %Transmittance: %T = 10^(2 - A)."""
    return np.power(10, (2.0 - absorbance))

def calculate_quality_score(transmittance, smoothed):
    """Smart feature: Calculate a basic Signal-to-Noise / Quality score."""
    noise = np.abs(transmittance - smoothed)
    mean_noise = np.mean(noise)
    signal_range = np.max(smoothed) - np.min(smoothed)
    
    if mean_noise == 0:
        snr = 999
    else:
        snr = signal_range / mean_noise
        
    if snr > 50:
        quality = "Excellent"
    elif snr > 20:
        quality = "Good"
    elif snr > 10:
        quality = "Fair"
    else:
        quality = "Poor (High Noise)"
        
    return snr, quality


def smooth_spectrum(y, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLY_ORDER):
    """Apply Savitzky-Golay filter for noise reduction while preserving peaks."""
    if len(y) < window:
        window = len(y) if len(y) % 2 == 1 else len(y) - 1
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def baseline_correction(wavenumber, transmittance):
    """
    Simple polynomial baseline correction.
    Fits a polynomial to the upper envelope and subtracts offset.
    """
    # Use a low-order polynomial fit to the data
    coeffs = np.polyfit(wavenumber, transmittance, deg=2)
    baseline = np.polyval(coeffs, wavenumber)

    # Shift so maximum transmittance ≈ 100%
    corrected = transmittance - baseline + 100
    corrected = np.clip(corrected, 0, 120)
    return corrected


def find_main_peaks(wavenumber, transmittance, prominence=PEAK_PROMINENCE,
                    distance=PEAK_MIN_DISTANCE):
    """
    Find absorption peaks (troughs in %T spectrum).
    We invert the spectrum to find minima as maxima.
    """
    inverted = -transmittance
    peaks, properties = find_peaks(inverted, prominence=prominence,
                                   distance=distance)

    # Sort by prominence (strongest peaks first)
    sorted_idx = np.argsort(-properties['prominences'])
    peaks = peaks[sorted_idx]

    peak_wavenumbers = wavenumber[peaks]
    peak_transmittance = transmittance[peaks]

    return peaks, peak_wavenumbers, peak_transmittance


def identify_functional_groups(peak_wavenumbers):
    """Match peak wavenumbers to functional groups."""
    results = []
    for wn in peak_wavenumbers:
        matched = False
        for wn_high, wn_low, bond, group in FUNCTIONAL_GROUPS:
            if wn_low <= wn <= wn_high:
                results.append({
                    'Wavenumber (cm⁻¹)': f'{wn:.1f}',
                    'Bond': bond,
                    'Functional Group': group,
                })
                matched = True
                break
        if not matched:
            results.append({
                'Wavenumber (cm⁻¹)': round(wn, 1),
                'Bond': '—',
                'Functional Group': 'Fingerprint region / Unassigned'
            })
            
    if not results:
        return pd.DataFrame(columns=['Wavenumber (cm⁻¹)', 'Bond', 'Functional Group'])
        
    return pd.DataFrame(results)

def predict_compound(found_bonds):
    """Predict compound class based on the combination of identified bonds."""
    predictions = []
    
    for rule in COMPOUND_RULES:
        # Check if all required bonds are in the found_bonds list
        has_all_reqs = all(any(req in b for b in found_bonds) for req in rule["requires"])
        
        # Check if any excluded bonds are in the found_bonds list
        has_no_excludes = not any(any(exc in b for b in found_bonds) for exc in rule["excludes"])
        
        if has_all_reqs and has_no_excludes:
            predictions.append(rule["compound"])
            
    if not predictions:
        # Fallbacks for complex mixtures or partial matches
        if any("O–H stretch" in b for b in found_bonds):
            predictions.append("Unknown Oxygenated Compound (contains O-H)")
        elif any("C=O stretch" in b for b in found_bonds):
            predictions.append("Unknown Carbonyl Compound")
        elif any("C–O stretch" in b for b in found_bonds):
            predictions.append("Unknown Oxygenated Compound (contains C-O)")
        else:
            predictions.append("Unidentified / Complex Mixture")
            
    return predictions


def plot_spectrum(wavenumber, transmittance, peaks, peak_wn, peak_t,
                  title, output_path):
    """Generate a publication-quality FTIR spectrum plot."""

    # ── Style Setup ──────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
    })

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=OUTPUT_DPI)

    # ── Background ───────────────────────────────────────────────────
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')

    # ── Main Spectrum Line ───────────────────────────────────────────
    ax.plot(wavenumber, transmittance, color='#1B2838', linewidth=1.3,
            zorder=3, label='%T Spectrum')

    # ── Shade the absorption bands lightly ───────────────────────────
    ax.fill_between(wavenumber, transmittance, 100,
                    alpha=0.06, color='#2196F3', zorder=1)

    # ── Peak Markers ─────────────────────────────────────────────────
    ax.scatter(peak_wn, peak_t, color='#D32F2F', s=45, zorder=5,
              edgecolors='#B71C1C', linewidths=0.8, marker='v',
              label=f'Identified Peaks ({len(peak_wn)})')

    # ── Peak Annotations ─────────────────────────────────────────────
    annotated_positions = []
    for i, (wn, t) in enumerate(zip(peak_wn, peak_t)):
        # Avoid overlapping labels
        too_close = False
        for prev_wn, prev_y in annotated_positions:
            if abs(wn - prev_wn) < 80 and abs(t - prev_y) < 8:
                too_close = True
                break

        if not too_close and len(annotated_positions) < 20:
            y_offset = -12 if t > 30 else 12
            ax.annotate(
                f'{wn:.0f}',
                xy=(wn, t),
                xytext=(0, y_offset),
                textcoords='offset points',
                fontsize=7.5,
                fontweight='bold',
                color='#D32F2F',
                ha='center', va='top' if y_offset < 0 else 'bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#DDDDDD', alpha=0.85),
                arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.5),
                zorder=6,
            )
            annotated_positions.append((wn, t))

    # ── Axis Configuration ───────────────────────────────────────────
    ax.set_xlim(WAVENUMBER_MAX, WAVENUMBER_MIN)  # Inverted X-axis
    ax.set_ylim(bottom=max(0, transmittance.min() - 5),
                top=min(110, transmittance.max() + 8))

    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=13, fontweight='bold',
                  labelpad=8)
    ax.set_ylabel('Transmittance (%T)', fontsize=13, fontweight='bold',
                  labelpad=8)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15,
                 color='#1B2838')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='#CCCCCC', fontsize=9)

    # ── Border ───────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()
    plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Plot saved -> {output_path}")

def interactive_wizard():
    print("\n" + "="*70)
    print(" 🌟 FTIR Spectroscopy Analyzer Wizard 🌟")
    print("="*70)
    
    path_input = input("\n1. Enter the path to your .CSV file or folder (or drag and drop it here):\n> ").strip().strip('"').strip("'")
    if not path_input:
        print("[ERROR] No path provided. Exiting.")
        return
        
    y_type_choice = input("\n2. Data Format: [1] Auto-Detect (Smart)  [2] Force Transmittance (%T)  [3] Force Absorbance (A)\n> [1]: ").strip()
    force_y_type = None
    if y_type_choice == '2':
        force_y_type = 'transmittance'
    elif y_type_choice == '3':
        force_y_type = 'absorbance'
        
    custom_name = input("\n3. Custom Plot Title (leave blank to use filename):\n> ").strip()
    if not custom_name:
        custom_name = None
        
    sens_choice = input("\n4. Peak Sensitivity: [1] Normal  [2] High (finds small peaks)  [3] Low (major peaks only)\n> [1]: ").strip()
    sensitivity = 'normal'
    if sens_choice == '2':
        sensitivity = 'high'
    elif sens_choice == '3':
        sensitivity = 'low'
        
    print("\nStarting analysis...\n")
    
    if os.path.isfile(path_input):
        analyze_file(path_input, custom_name=custom_name, force_y_type=force_y_type, sensitivity=sensitivity)
    elif os.path.isdir(path_input):
        search_pattern = os.path.join(path_input, "*.CSV")
        csv_files = glob.glob(search_pattern) + glob.glob(os.path.join(path_input, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files found in {path_input}")
        else:
            for f in set(csv_files):
                analyze_file(f, force_y_type=force_y_type, sensitivity=sensitivity)
    else:
        print(f"[ERROR] Path not found: {path_input}")


def analyze_file(filepath, custom_name=None, output_dir=None, force_y_type=None, sensitivity='normal'):
    """Full FTIR analysis pipeline for a single file."""
    basename = os.path.basename(filepath)
    if custom_name:
        name_no_ext = custom_name
        print_name = custom_name
    else:
        name_no_ext = os.path.splitext(basename)[0]
        print_name = basename
        
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    else:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FTIR Analysis: {print_name}")
    print(f"{'='*70}")

    # 1. Load Data
    print("\n[1/6] Loading data...")
    df = load_csv(filepath)
    print(f"  - Raw data points: {len(df)}")
    print(f"  - Wavenumber range: {df['wavenumber'].min():.1f} - {df['wavenumber'].max():.1f} cm-1")

    # [SMART FEATURE] Detect data type
    max_y = df['y_value'].max()
    if force_y_type == 'transmittance':
        data_type = 'Transmittance'
        print(f"  - [FORCED] Processing as Transmittance (%T)")
    elif force_y_type == 'absorbance':
        data_type = 'Absorbance'
        print(f"  - [FORCED] Processing as Absorbance (A)")
    else:
        if max_y > 10:
            data_type = 'Transmittance'
            print(f"  - [SMART DETECT] Max Y = {max_y:.1f}. Auto-detected format: Transmittance (%T)")
        else:
            data_type = 'Absorbance'
            print(f"  - [SMART DETECT] Max Y = {max_y:.2f}. Auto-detected format: Absorbance (A)")

    # 2. Filter to standard IR range
    print("\n[2/6] Filtering to standard range (400-4000 cm-1)...")
    df = filter_range(df)
    print(f"  - Data points after filtering: {len(df)}")

    # 3 & 4. Process based on detected type
    if data_type == 'Absorbance':
        print("\n[3/6] Cleaning saturated absorbance values...")
        df['y_value'] = df['y_value'].clip(lower=0, upper=3.5)
        print("\n[4/6] Converting Absorbance -> Transmittance & smoothing...")
        raw_transmittance = convert_to_transmittance(df['y_value'].values)
    else:
        print("\n[3/6] Cleaning Transmittance values...")
        df['y_value'] = df['y_value'].clip(lower=0, upper=120)
        print("\n[4/6] Smoothing Transmittance spectrum...")
        raw_transmittance = df['y_value'].values

    transmittance = smooth_spectrum(raw_transmittance)
    wavenumber = df['wavenumber'].values
    
    # [SMART FEATURE] Quality Score
    snr, quality = calculate_quality_score(raw_transmittance, transmittance)
    print(f"  - [SMART] Spectrum Quality: {quality} (Estimated SNR: {snr:.1f})")

    # 5. Find peaks
    print("\n[5/6] Identifying major absorption peaks...")
    
    # Adjust sensitivity
    prominence = PEAK_PROMINENCE
    if sensitivity == 'high':
        prominence = max(0.5, PEAK_PROMINENCE * 0.4)
        print(f"  - Sensitivity: HIGH (Prominence threshold: {prominence:.1f})")
    elif sensitivity == 'low':
        prominence = PEAK_PROMINENCE * 2.0
        print(f"  - Sensitivity: LOW (Prominence threshold: {prominence:.1f})")
    else:
        print(f"  - Sensitivity: NORMAL (Prominence threshold: {prominence:.1f})")
        
    peaks, peak_wn, peak_t = find_main_peaks(wavenumber, transmittance, prominence=prominence)
    print(f"  - Main peaks found: {len(peaks)}")
    for wn, t in zip(peak_wn, peak_t):
        print(f"    -> {wn:.1f} cm-1  ({t:.1f} %T)")

    # 6. Functional Group Identification
    print("\n[6/6] Functional group analysis...")
    fg_table = identify_functional_groups(peak_wn)
    
    if fg_table.empty:
        print("  - No significant peaks or functional groups found.")
        predictions = ["Unidentified (No peaks found)"]
    else:
        table_str = fg_table.to_string(index=False)
        # Safely print, replacing any problematic chars
        print(table_str.encode('ascii', 'replace').decode('ascii'))
        
        # [SMART FEATURE] Predict Compound
        found_bonds = fg_table['Bond'].unique().tolist()
        predictions = predict_compound(found_bonds)
        
        print(f"\n  [SMART PREDICTION] Based on spectral analysis, the compound is likely:")
        for pred in predictions:
            print(f"   -> {pred}")

    # Save table
    table_path = os.path.join(output_dir, f"{name_no_ext}_functional_groups.csv")
    
    # Append prediction to the bottom of the CSV
    with open(table_path, 'w', encoding='utf-8') as f:
        fg_table.to_csv(f, index=False)
        f.write(f"\nAI PREDICTION:,{ ' OR '.join(predictions) }\n")
        
    print(f"\n  [OK] Table saved -> {table_path}")

    # Generate plot
    title = name_no_ext.replace('_', ' ')
    png_path = os.path.join(output_dir, f"{name_no_ext}_FTIR_spectrum.png")
    plot_spectrum(wavenumber, transmittance, peaks, peak_wn, peak_t,
                  title, png_path)

    return fg_table


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # Run interactive wizard if no arguments are passed
        interactive_wizard()
        print(f"\n{'='*70}")
        print("  All analyses complete.")
        print(f"{'='*70}\n")
    else:
        parser = argparse.ArgumentParser(description="FTIR Spectroscopy Analyzer")
        parser.add_argument('--file', type=str, help="Path to a single .CSV file to analyze")
        parser.add_argument('--name', type=str, help="Custom name for the single file analysis plot title")
        parser.add_argument('--dir', type=str, help="Directory containing multiple .CSV files to analyze")
        parser.add_argument('--out', type=str, help="Output directory for plots and tables")
        
        args = parser.parse_args()
        
        if args.file:
            if os.path.exists(args.file):
                analyze_file(args.file, custom_name=args.name, output_dir=args.out)
            else:
                print(f"[ERROR] File not found: {args.file}")
                
        elif args.dir:
            if os.path.isdir(args.dir):
                search_pattern = os.path.join(args.dir, "*.CSV")
                # Support both .CSV and .csv
                csv_files = glob.glob(search_pattern) + glob.glob(os.path.join(args.dir, "*.csv"))
                
                if not csv_files:
                    print(f"[WARN] No CSV files found in {args.dir}")
                else:
                    for f in set(csv_files):
                        analyze_file(f, output_dir=args.out)
            else:
                print(f"[ERROR] Directory not found: {args.dir}")
        else:
            parser.print_help()
            
        print(f"\n{'='*70}")
        print("  All analyses complete.")
        print(f"{'='*70}\n")
