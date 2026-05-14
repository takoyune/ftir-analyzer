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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter1d
import os
import argparse
import glob
import warnings
import urllib.request
import urllib.parse
import io
import matplotlib.image as mpimg
warnings.filterwarnings('ignore')

from ftir_ai import FTIRExpertSystem

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

            wn_val = float(wn_str)
            abs_val = float(abs_str)
            
            wavenumbers.append(wn_val)
            absorbances.append(abs_val)
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

def convert_to_absorbance(transmittance):
    """Convert %Transmittance to Absorbance: A = 2 - log10(%T)."""
    # Clip near zero to avoid log10(0)
    t_safe = np.clip(transmittance, 1e-5, None)
    return 2.0 - np.log10(t_safe)

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


def baseline_correction_asls(y, lam=1e10, p=0.01, max_iter=15, data_type='Transmittance'):
    """
    Advanced Asymmetric Least Squares (AsLS) Baseline Correction.
    Adapts intelligently to Transmittance (baseline on top) or Absorbance (baseline on bottom).
    """
    n = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    DTD = D.T @ D
    w = np.ones(n)
    baseline = y.copy()
    
    # In Transmittance, baseline is at ~100%, so we heavily penalize baseline dipping BELOW data
    # In Absorbance, baseline is at ~0, so we heavily penalize baseline spiking ABOVE data
    actual_p = 0.99 if data_type == 'Transmittance' else p
    
    for _ in range(max_iter):
        W = diags(w)
        # Convert to CSC format for efficient solving and to suppress SparseEfficiencyWarning
        matrix_to_solve = (W + lam * DTD).tocsc()
        baseline = spsolve(matrix_to_solve, w * y)
        w = np.where(y > baseline, actual_p, 1 - actual_p)
        
    if data_type == 'Transmittance':
        # Flatten and align to 100% T
        corrected = y - baseline + 100
        return np.clip(corrected, 0, 120), baseline
    else:
        # Flatten and align to 0 Absorbance
        corrected = y - baseline
        return np.clip(corrected, 0, None), baseline


def find_main_peaks(wavenumber, y_data, data_type='Transmittance', prominence=PEAK_PROMINENCE, distance=PEAK_MIN_DISTANCE):
    """Identify major absorption peaks using adaptive noise thresholding."""
    if data_type == 'Transmittance':
        signal_for_peaks = -y_data
        base_prominence = prominence
    else:
        signal_for_peaks = y_data
        base_prominence = max(0.01, prominence * 0.015)

    # 1. Adaptive Prominence: Must be at least 2% of the total signal dynamic range
    signal_range = np.max(signal_for_peaks) - np.min(signal_for_peaks)
    adaptive_prominence = max(base_prominence, signal_range * 0.02)

    # 2. Find peaks with width constraint to ignore 1-point noise spikes
    # Broad peaks (like O-H) have very large width but low prominence
    peaks, properties = find_peaks(
        signal_for_peaks, 
        prominence=adaptive_prominence, 
        distance=distance,
        width=2  # Prevents infinitesimally sharp noise spikes
    )

    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    # 3. Sort by prominence (most prominent first)
    sorted_idx = np.argsort(-properties['prominences'])
    peaks = peaks[sorted_idx]
    prominences = properties['prominences'][sorted_idx]

    # 4. Smart Filtering: Drop peaks that are extremely tiny compared to the absolute largest peak
    max_prom = prominences[0]
    # Lowered threshold to 3% to catch broad, less prominent O-H stretches
    significant_peaks = [p for p, prom in zip(peaks, prominences) if prom >= max_prom * 0.03]
    
    # 5. Cap at top 35 most important peaks to avoid cluttering but preserve important broad bands
    peaks = np.array(significant_peaks[:35])

    # Re-sort the final filtered peaks by wavenumber (left to right) for consistency
    final_peaks_sorted = np.sort(peaks)

    peak_wavenumbers = wavenumber[final_peaks_sorted]
    peak_y = y_data[final_peaks_sorted]

    return final_peaks_sorted, peak_wavenumbers, peak_y



def plot_spectrum(wavenumber, y_data, peaks, peak_wn, peak_y,
                  title, output_path, data_type='Transmittance',
                  fg_table=None, top_compound=None):
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

    fig = plt.figure(figsize=(14, 9) if (fg_table is not None and not fg_table.empty) else (14, 5.5), dpi=OUTPUT_DPI)
    fig.patch.set_facecolor('#FAFAFA')
    
    # ── GridSpec Layout ──────────────────────────────────────────────
    if fg_table is not None and not fg_table.empty:
        if top_compound:
            gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], width_ratios=[3.5, 1], hspace=0.25, wspace=0.05)
            ax = fig.add_subplot(gs[0, :])
            ax_table = fig.add_subplot(gs[1, 0])
            ax_img_box = fig.add_subplot(gs[1, 1])
            ax_img_box.axis('off')
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.25)
            ax = fig.add_subplot(gs[0])
            ax_table = fig.add_subplot(gs[1])
            ax_img_box = None
            
        ax_table.axis('off')
        
        # Draw table
        table_data = fg_table.copy()
        if 'Confidence' in table_data.columns:
            table_data['Confidence'] = table_data['Confidence'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x)
        
        display_data = table_data.head(10) # Show top 10 rows
        
        cell_text = []
        for row in range(len(display_data)):
            cell_text.append(display_data.iloc[row].values)
            
        table = ax_table.table(cellText=cell_text, colLabels=display_data.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#1B2838')
            else:
                cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')
                
        # Draw compound image
        if ax_img_box is not None and top_compound:
            try:
                safe_name = urllib.parse.quote(top_compound)
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/PNG"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    img_data = response.read()
                    
                img = mpimg.imread(io.BytesIO(img_data))
                ax_img_box.imshow(img)
                ax_img_box.set_title(f"AI Prediction:\n{top_compound}", fontsize=11, fontweight='bold', color='#1B2838')
            except Exception as e:
                print(f"  [WARN] Could not fetch compound image for {top_compound}: {e}")
    else:
        ax = fig.add_subplot(111)

    # ── Background ───────────────────────────────────────────────────
    ax.set_facecolor('#FFFFFF')

    # ── Main Spectrum Line ───────────────────────────────────────────
    label_txt = '%T Spectrum' if data_type == 'Transmittance' else 'Absorbance Spectrum'
    ax.plot(wavenumber, y_data, color='#1B2838', linewidth=1.3,
            zorder=3, label=label_txt)

    # ── Shade the absorption bands lightly ───────────────────────────
    if data_type == 'Transmittance':
        ax.fill_between(wavenumber, y_data, 100, alpha=0.06, color='#2196F3', zorder=1)
    else:
        ax.fill_between(wavenumber, y_data, 0, alpha=0.06, color='#2196F3', zorder=1)

    # ── Peak Markers ─────────────────────────────────────────────────
    marker_type = 'v' if data_type == 'Transmittance' else '^'
    ax.scatter(peak_wn, peak_y, color='#D32F2F', s=45, zorder=5,
              edgecolors='#B71C1C', linewidths=0.8, marker=marker_type,
              label=f'Identified Peaks ({len(peak_wn)})')

    # ── Peak Annotations ─────────────────────────────────────────────
    annotated_positions = []
    for i, (wn, y) in enumerate(zip(peak_wn, peak_y)):
        too_close = False
        for prev_wn, prev_y in annotated_positions:
            if abs(wn - prev_wn) < 80 and abs(y - prev_y) < (8 if data_type=='Transmittance' else 0.1):
                too_close = True
                break

        if not too_close and len(annotated_positions) < 20:
            if data_type == 'Transmittance':
                y_offset = -12 if y > 30 else 12
            else:
                y_offset = 12
                
            ax.annotate(
                f'{wn:.0f}',
                xy=(wn, y),
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
            annotated_positions.append((wn, y))

    # ── Axis Configuration ───────────────────────────────────────────
    ax.set_xlim(WAVENUMBER_MAX, WAVENUMBER_MIN)  # Inverted X-axis
    
    if data_type == 'Transmittance':
        ax.set_ylim(bottom=max(0, y_data.min() - 5), top=min(110, y_data.max() + 8))
        ax.set_ylabel('Transmittance (%T)', fontsize=13, fontweight='bold', labelpad=8)
    else:
        ax.set_ylim(bottom=0, top=y_data.max() + 0.1)
        ax.set_ylabel('Absorbance (A)', fontsize=13, fontweight='bold', labelpad=8)

    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=13, fontweight='bold',
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

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def interactive_wizard():
    # Enable ANSI colors on Windows
    os.system('color')
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*70)
    print("  🚀 FTIR Spectroscopy Analyzer Wizard - Premium Edition 🚀  ")
    print("="*70 + f"{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}1. Select your .CSV file or folder{Colors.RESET}")
    print(f"   {Colors.YELLOW}(Press ENTER to open File Explorer, or paste path here){Colors.RESET}")
    path_input = input("> ").strip().strip('"').strip("'")
    
    if not path_input:
        print(f"   {Colors.CYAN}Opening File Explorer...{Colors.RESET}")
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path_input = filedialog.askopenfilename(
            title="Select FTIR CSV File",
            filetypes=[("CSV Files", "*.csv *.CSV"), ("All Files", "*.*")]
        )
        if not path_input:
            print(f"{Colors.RED}[ERROR] No file selected. Exiting.{Colors.RESET}")
            return
        print(f"   {Colors.GREEN}Selected: {path_input}{Colors.RESET}")
        
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
    
    if df.empty:
        print(f"  [ERROR] No valid numerical spectroscopy data could be extracted from this file.")
        print("          Please ensure this is a valid CSV file exported from an FTIR machine.")
        return None
        
    print(f"  - Raw data points: {len(df)}")
    print(f"  - Wavenumber range: {df['wavenumber'].min():.1f} - {df['wavenumber'].max():.1f} cm-1")

    # [SMART FEATURE] Detect Raw Data Type
    max_y = df['y_value'].max()
    if max_y > 10:
        raw_type = 'Transmittance'
        print(f"  - [SMART DETECT] Max Y = {max_y:.1f}. Auto-detected format: Transmittance (%T)")
    else:
        raw_type = 'Absorbance'
        print(f"  - [SMART DETECT] Max Y = {max_y:.2f}. Auto-detected format: Absorbance (A)")

    # Determine Target Type
    target_type = raw_type
    if force_y_type == 'transmittance':
        target_type = 'Transmittance'
        print(f"  - [USER OVERRIDE] Final output will be forced to Transmittance (%T)")
    elif force_y_type == 'absorbance':
        target_type = 'Absorbance'
        print(f"  - [USER OVERRIDE] Final output will be forced to Absorbance (A)")

    # 2. Filter to standard IR range
    print("\n[2/7] Filtering to standard range (400-4000 cm-1)...")
    df = filter_range(df)
    print(f"  - Data points after filtering: {len(df)}")

    # 3 & 4. Process based on raw and target types
    if target_type == 'Transmittance':
        if raw_type == 'Absorbance':
            print("\n[3/7] Cleaning saturated absorbance values...")
            df['y_value'] = df['y_value'].clip(lower=0, upper=3.5)
            print("\n[4/7] Converting Absorbance -> Transmittance & smoothing...")
            raw_y = convert_to_transmittance(df['y_value'].values)
        else:
            print("\n[3/7] Cleaning Transmittance values...")
            df['y_value'] = df['y_value'].clip(lower=0, upper=120)
            print("\n[4/7] Smoothing Transmittance spectrum...")
            raw_y = df['y_value'].values
    else: # target_type == 'Absorbance'
        if raw_type == 'Transmittance':
            print("\n[3/7] Cleaning Transmittance values...")
            df['y_value'] = df['y_value'].clip(lower=0, upper=120)
            print("\n[4/7] Converting Transmittance -> Absorbance & smoothing...")
            raw_y = convert_to_absorbance(df['y_value'].values)
        else:
            print("\n[3/7] Cleaning saturated absorbance values...")
            df['y_value'] = df['y_value'].clip(lower=0, upper=3.5)
            print("\n[4/7] Smoothing Absorbance spectrum...")
            raw_y = df['y_value'].values

    smoothed_y = smooth_spectrum(raw_y)
    wavenumber = df['wavenumber'].values
    
    # 5. Advanced Baseline Correction (AsLS)
    print("\n[5/7] Applying Asymmetric Least Squares (AsLS) Baseline Correction...")
    processed_y, _ = baseline_correction_asls(smoothed_y, data_type=target_type)
    print("  - Baseline flattened successfully.")
    
    # [SMART FEATURE] Quality Score
    snr, quality = calculate_quality_score(raw_y, processed_y)
    print(f"  - [SMART] Spectrum Quality: {quality} (Estimated SNR: {snr:.1f})")

    # 6. Find peaks
    print("\n[6/7] Identifying major absorption peaks...")
    
    # Adjust sensitivity
    prominence = PEAK_PROMINENCE
    if sensitivity == 'high':
        prominence = max(0.5, PEAK_PROMINENCE * 0.4)
        print(f"  - Sensitivity: HIGH")
    elif sensitivity == 'low':
        prominence = PEAK_PROMINENCE * 2.0
        print(f"  - Sensitivity: LOW")
    else:
        print(f"  - Sensitivity: NORMAL")
        
    peaks, peak_wn, peak_y = find_main_peaks(wavenumber, processed_y, data_type=target_type, prominence=prominence)
    print(f"  - Main peaks found: {len(peaks)}")
    for wn, y in zip(peak_wn, peak_y):
        unit = '%T' if target_type == 'Transmittance' else 'A'
        print(f"    -> {wn:.1f} cm-1  ({y:.2f} {unit})")

    # 7. AI Expert System Analysis
    print(f"\n{Colors.CYAN}{Colors.BOLD}[7/7] AI Expert System Analysis...{Colors.RESET}")
    expert = FTIRExpertSystem()
    fg_table, predictions, insights = expert.analyze_spectrum(peak_wn)
    
    if fg_table.empty:
        print(f"  {Colors.RED}- No significant peaks or functional groups found.{Colors.RESET}")
    else:
        table_str = fg_table.to_string(index=False)
        # Safely print, replacing any problematic chars
        print(table_str.encode('ascii', 'replace').decode('ascii'))
        
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}🤖 [AI INSIGHTS]{Colors.RESET}")
        for insight in insights:
            print(f"   {Colors.CYAN}*{Colors.RESET} {insight}")
            
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎯 [AI PREDICTION]{Colors.RESET} Based on spectral analysis, the compound is likely:")
        for pred in predictions:
            if "[Specific Compound]" in pred:
                print(f"   {Colors.GREEN}➔ {Colors.BOLD}{pred}{Colors.RESET}")
            else:
                print(f"   {Colors.GREEN}➔ {pred}{Colors.RESET}")

    # Save table
    table_path = os.path.join(output_dir, f"{name_no_ext}_functional_groups.csv")
    
    try:
        # Append prediction to the bottom of the CSV
        with open(table_path, 'w', encoding='utf-8') as f:
            fg_table.to_csv(f, index=False)
            f.write("\nAI INSIGHTS:\n")
            for insight in insights:
                f.write(f"{insight}\n")
            f.write(f"\nAI PREDICTION:,{ ' OR '.join(predictions) }\n")
        print(f"\n  [OK] Table saved -> {table_path}")
    except PermissionError:
        print(f"\n  [ERROR] Permission denied: Could not save '{name_no_ext}_functional_groups.csv'.")
        print("          Is the file currently open in Excel or another program? Please close it and try again.")
    except Exception as e:
        print(f"\n  [ERROR] Could not save table: {e}")

    # Generate plot
    title = name_no_ext.replace('_', ' ')
    png_path = os.path.join(output_dir, f"{name_no_ext}_FTIR_spectrum.png")
    
    try:
        # Extract top predicted specific compound name
        top_compound = None
        if predictions:
            for pred in predictions:
                if "[Specific Compound]" in pred:
                    top_compound = pred.split("]")[1].split("(")[0].strip()
                    break
                    
        plot_spectrum(wavenumber, processed_y, peaks, peak_wn, peak_y, title, png_path, 
                      data_type=target_type, fg_table=fg_table, top_compound=top_compound)
    except PermissionError:
        print(f"  [ERROR] Permission denied: Could not save plot '{name_no_ext}_FTIR_spectrum.png'.")
        print("          Is the image file currently open in another program?")
    except Exception as e:
        print(f"  [ERROR] Failed to generate plot: {e}")

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