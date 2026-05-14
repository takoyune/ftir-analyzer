# FTIR Spectroscopy Analyzer

An automated, smart Python pipeline for analyzing Fourier Transform Infrared (FTIR) spectroscopy datasets. 

This tool automatically detects whether your data is in Absorbance (A) or Transmittance (%T), performs noise-reduction, applies baseline correction, identifies key functional groups, and generates publication-quality visualization plots.

## Features

- **Smart Detection**: Automatically determines if input data is `%T` or `Absorbance`.
- **Automated Conversion**: Calculates `%T = 10^(2 - A)` if Absorbance data is detected.
- **Spectrum Quality Scoring**: Estimates Signal-to-Noise Ratio (SNR) and provides a qualitative score (Excellent, Good, Fair, Poor).
- **Intelligent Peak Picking**: Uses prominence-based algorithms to find main troughs while ignoring electronic noise.
- **Functional Group Identification**: Cross-references identified peaks with a chemical bond database to identify likely functional groups (e.g., C=O stretches, O-H bonds).
- **Publication-Ready Plots**: Generates high-resolution PNGs with inverted X-axes, annotated peaks, and clean scientific styling.

## Requirements

Ensure you have Python installed along with the following dependencies:

```bash
pip install pandas numpy matplotlib scipy
```

## Usage

You can run the script directly from your terminal. The script now includes a Command Line Interface (CLI) for flexible usage.

### 1. Process a Specific File
If you want to analyze a single `.CSV` file and optionally give it a custom name for the plot title:

```bash
python ftir_analyzer.py --file "path/to/your/data.CSV" --name "My Custom Sample"
```

### 2. Process an Entire Directory
If you have a folder full of `.CSV` files, you can process them all in one go:

```bash
python ftir_analyzer.py --dir "path/to/folder"
```

### 3. Save Outputs to a Specific Folder
By default, the script saves the generated plots (`.png`) and functional group tables (`.csv`) in the same folder as the input data. You can redirect these outputs to a specific folder:

```bash
python ftir_analyzer.py --file "data.CSV" --out "results_folder/"
```

## Understanding the Output

For every input `.CSV` file, the script generates two files:

1. **`[Name]_FTIR_spectrum.png`**: A high-resolution plot of the spectrum.
2. **`[Name]_functional_groups.csv`**: A table listing the identified wavenumbers, their chemical bonds, and functional groups.

*Note: The script is designed to handle `.CSV` files with either comma (`,`) or dot (`.`) decimal separators, and semicolon (`;`) or comma (`,`) column delimiters.*
