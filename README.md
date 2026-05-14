<div align="center">
  <h1>🔬 FTIR Expert System - Premium Edition 🚀</h1>
  <p><i>A production-grade, AI-driven pipeline for analyzing Fourier Transform Infrared (FTIR) spectroscopy datasets.</i></p>
</div>

---

This tool has been completely overhauled from a simple parser into an **Intelligent Spectroscopy Expert System**. Featuring advanced mathematical noise handling, natural language diagnostic insights, dynamic signature matching, and stunning publication-ready dashboards.

## 🌟 Premium Features

### 🧠 Dedicated AI Expert System (`ftir_ai.py`)
- **Natural Language Diagnostics:** Acts like a human spectroscopist. Instead of just listing peaks, the AI generates readable `[AI INSIGHTS]` (e.g., flagging the presence of complex aromatic systems or specific carbonyl groups).
- **Smart Override Logic:** If the AI matches an exact compound fingerprint with high confidence (>70%), it intelligently suppresses generic class guesses to give you a definitive answer.
- **Dynamic Tolerance Matching:** Physically models reality. Sharp peaks require strict matching (`±30 cm⁻¹`), while broad structural bands (like O-H stretches) automatically expand their search windows (`±150 cm⁻¹`) so they are never missed.

### 📐 Advanced Mathematical Preprocessing
- **Asymmetric Least Squares (AsLS) Baseline Correction:** Replaced naive polynomial fitting with the industry-standard AsLS algorithm (`lam=1e10`). It acts as an elastic band that flawlessly flattens background drift and slope without erasing massive, broad spectral bumps.
- **Adaptive Noise & Prominence Filtering:** Dynamically scales its peak-detection thresholds based on the signal-to-noise ratio of your specific file, dropping digital spikes and retaining true molecular signals.

### 📊 Publication-Quality Analytics Dashboard
- **Embedded Grid Layout:** The generated `.png` isn't just a graph anymore. It uses advanced Matplotlib `GridSpec` to render a beautiful data table of your top functional groups directly beneath the spectrum.
- **Live PubChem API Integration:** When the AI identifies a specific chemical, the script actively queries the official PubChem REST API, downloads the 2D molecular structure, and renders it directly into the bottom right corner of your plot!

### 💻 Premium User Experience
- **Native GUI File Explorer:** No more typing paths. Just hit `ENTER` in the terminal and a sleek Tkinter File Chooser pops up.
- **ANSI Color Engine:** The CLI wizard has been completely redesigned with a vibrant cyan, green, and yellow interface that highlights AI insights and predictions in real-time.
- **Smart Format Detection:** Automatically detects if your data is Absorbance or Transmittance based on maximum scale, and seamlessly converts it.

---

## 🚀 Absolute Beginner's Guide (How to Use)

Never used a command prompt or Python before? No problem! Follow these exact steps:

### Step 1: Install Python
1. Download and install **Python 3** from [python.org](https://www.python.org/downloads/).
2. **CRITICAL:** During installation, make sure to check the box that says **"Add Python to PATH"** before clicking Install.

### Step 2: Open the Command Prompt
1. Press the `Windows Key` on your keyboard, type `cmd`, and press `Enter`. 
2. A black window (the Command Prompt) will open.

### Step 3: Install Required Libraries
Copy and paste the following line into the black window and press `Enter`:
```cmd
pip install pandas numpy matplotlib scipy
```
Wait for it to finish downloading (you'll see a bunch of loading bars).

### Step 4: Run the Analyzer 🚀
1. Now, you need to tell the Command Prompt where your script is.
2. Find the folder where you saved `ftir_analyzer.py`.
3. Right-click that folder and select **"Copy as path"**.
4. In the black window, type `cd` followed by a space, then **right-click** to paste your path. It should look something like this:
   ```cmd
   cd "C:\Users\Name\Desktop\ftir-analyzer"
   ```
5. Press `Enter`. Now you are in the right place!
6. Finally, type this and press `Enter`:
   ```cmd
   python ftir_analyzer.py
   ```

---

## ✨ Why this is awesome for everyone:
- **No Coding Needed:** The script talks to you in plain English.
- **Easy File Picking:** You don't have to type long file names. Just hit `Enter` and pick your file like you do in Word or Excel.
- **Auto-Magic Analysis:** It automatically knows if your data is "Absorbance" or "Transmittance".
- **Professional Results:** It saves a beautiful graph and a spreadsheet report automatically in your folder!

---

## 💻 Advanced Command Line Usage
For power-users, you can bypass the interactive wizard by passing arguments directly:
```cmd
python ftir_analyzer.py --file "path/to/data.CSV"
python ftir_analyzer.py --dir "path/to/folder"
```

---

## 📂 Understanding the Outputs

For every analysis, the engine outputs two highly-detailed files:

1. **`[Name]_FTIR_spectrum.png`**: The ultimate analytics dashboard. Contains the shaded FTIR spectrum graph, the top 10 functional groups table, and (if successfully identified) the 2D molecular structure of the predicted compound.
2. **`[Name]_functional_groups.csv`**: The complete raw data table containing wavenumbers, bonds, assigned groups, and confidence scores. It now includes the full AI Diagnostic block and final predictions appended to the bottom.

---
<div align="center">
  <i>Built for the Modern Spectroscopist.</i>
</div>
