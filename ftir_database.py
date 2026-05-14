"""
FTIR Functional Group and Compound Database
============================================
Comprehensive reference table of IR vibrational frequencies with
confidence scores, region labels, and weighted compound prediction rules.

Each FUNCTIONAL_GROUPS entry is a tuple:
  (wn_high, wn_low, bond_name, group_name, confidence)

  confidence: 0.0–1.0 — how diagnostic/distinctive this band is.
              1.0 = extremely characteristic
              0.5 = moderately diagnostic
              0.2 = common / low selectivity

Each COMPOUND_RULES entry is a dict with:
  - compound   : str name
  - requires   : list[str] bond substrings that MUST be present
  - excludes   : list[str] bond substrings that must NOT be present
  - confidence : float (0.0–1.0) base confidence for this rule
  - notes      : str (human-readable explanation)
"""

# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONAL GROUPS TABLE
# Format: (wn_high, wn_low, bond_name, group_name, confidence)
# ─────────────────────────────────────────────────────────────────────────────
FUNCTIONAL_GROUPS = [

    # ── ALCOHOLS & PHENOLS ──────────────────────────────────────────────
    (3700, 3584, "O–H stretch (free)",       "Alcohol (free OH)",            0.95),
    (3584, 3200, "O–H stretch (H-bonded)",   "Alcohol / Phenol",             0.90),
    (1420, 1330, "O–H in-plane bend",        "Alcohol / Phenol",             0.60),
    (1260, 1000, "C–O stretch",              "Alcohol / Ether / Ester",      0.55),
    ( 750,  650, "O–H out-of-plane wag",     "Alcohol (broad)",              0.50),

    # ── CARBOXYLIC ACIDS ────────────────────────────────────────────────
    (3300, 2500, "O–H stretch (broad)",      "Carboxylic Acid",              0.95),
    (1725, 1700, "C=O stretch",              "Carboxylic Acid",              0.90),
    ( 955,  915, "O–H out-of-plane bend",    "Carboxylic Acid (dimer)",      0.80),

    # ── ESTERS ──────────────────────────────────────────────────────────
    (1750, 1735, "C=O stretch",              "Ester",                        0.90),
    (1300, 1150, "C–O–C asym stretch",       "Ester",                        0.80),
    (1150, 1050, "C–O–C sym stretch",        "Ester",                        0.70),

    # ── LACTONES ────────────────────────────────────────────────────────
    (1795, 1760, "C=O stretch",              "Lactone (5-ring)",             0.88),
    (1750, 1730, "C=O stretch",              "Lactone (6-ring)",             0.85),

    # ── ANHYDRIDES ──────────────────────────────────────────────────────
    (1860, 1800, "C=O stretch (asym)",       "Anhydride",                    0.95),
    (1800, 1750, "C=O stretch (sym)",        "Anhydride",                    0.95),

    # ── ACID HALIDES ────────────────────────────────────────────────────
    (1815, 1785, "C=O stretch",              "Acid Halide (Acyl Chloride)",  0.95),

    # ── ALDEHYDES ───────────────────────────────────────────────────────
    (2870, 2695, "C–H stretch (Fermi)",      "Aldehyde",                     0.90),
    (1740, 1720, "C=O stretch",              "Aldehyde",                     0.85),

    # ── KETONES ─────────────────────────────────────────────────────────
    (1725, 1705, "C=O stretch",              "Ketone",                       0.85),
    (1700, 1680, "C=O stretch (conj.)",      "Conjugated Ketone",            0.82),

    # ── AMIDES ──────────────────────────────────────────────────────────
    (1690, 1650, "C=O stretch",              "Amide (Band I)",               0.90),
    (1580, 1520, "N–H bend + C–N stretch",   "Amide (Band II)",              0.85),
    (1310, 1230, "C–N stretch + N–H bend",   "Amide (Band III)",             0.75),

    # ── CARBAMATES / URETHANES ──────────────────────────────────────────
    (1730, 1700, "C=O stretch",              "Carbamate / Urethane",         0.80),
    (1540, 1510, "N–H bend",                 "Carbamate / Urethane",         0.75),

    # ── IMIDES ──────────────────────────────────────────────────────────
    (1800, 1770, "C=O stretch (asym)",       "Imide",                        0.90),
    (1730, 1700, "C=O stretch (sym)",        "Imide",                        0.88),

    # ── AMINES ──────────────────────────────────────────────────────────
    (3500, 3300, "N–H stretch",              "Primary / Secondary Amine",    0.80),
    (3400, 3300, "N–H stretch (asym)",       "Primary Amine",                0.85),
    (3300, 3250, "N–H stretch (sym)",        "Primary Amine",                0.82),
    (1650, 1580, "N–H bend (scissor)",       "Primary Amine",                0.75),
    (1580, 1490, "N–H bend",                 "Secondary Amine",              0.65),
    (1360, 1250, "C–N stretch",              "Aromatic Amine",               0.70),
    (1250, 1020, "C–N stretch",              "Aliphatic Amine",              0.55),
    ( 910,  665, "N–H wag",                  "Primary / Secondary Amine",    0.60),

    # ── NITRO COMPOUNDS ─────────────────────────────────────────────────
    (1560, 1515, "N–O asym stretch",         "Nitro Compound",               0.95),
    (1385, 1345, "N–O sym stretch",          "Nitro Compound",               0.90),

    # ── NITROSO ─────────────────────────────────────────────────────────
    (1590, 1500, "N=O stretch",              "Nitroso Compound",             0.88),

    # ── ALKANES ─────────────────────────────────────────────────────────
    (3000, 2850, "C–H stretch",              "Alkane (sp3 CH)",              0.70),
    (2965, 2950, "C–H asym stretch",         "Alkane (CH3)",                 0.72),
    (2926, 2916, "C–H asym stretch",         "Alkane (CH2)",                 0.70),
    (2872, 2862, "C–H sym stretch",          "Alkane (CH3)",                 0.68),
    (2853, 2843, "C–H sym stretch",          "Alkane (CH2)",                 0.68),
    (1470, 1450, "C–H bend (scissoring)",    "Alkane (CH2)",                 0.55),
    (1390, 1370, "C–H bend (umbrella)",      "Alkane (CH3)",                 0.60),
    ( 730,  720, "C–H rocking",              "Alkane (long chain ≥4C)",      0.72),

    # ── AROMATICS ───────────────────────────────────────────────────────
    (3100, 3000, "=C–H stretch",             "Aromatic C–H",                 0.75),
    (1625, 1575, "C=C ring stretch",         "Aromatic Ring (4 bands)",      0.80),
    (1525, 1475, "C=C ring stretch",         "Aromatic Ring",                0.78),
    (1175, 1125, "C–H in-plane bend",        "Aromatic C–H",                 0.65),
    ( 900,  690, "C–H out-of-plane bend",    "Aromatic Ring (subst. pattern)", 0.80),

    # ── ALKENES ─────────────────────────────────────────────────────────
    (3150, 3000, "=C–H stretch",             "Alkene (sp2 C–H)",             0.75),
    (1680, 1620, "C=C stretch",              "Alkene",                       0.78),
    (1000,  960, "=C–H bend (trans)",        "Alkene (trans)",               0.82),
    ( 915,  885, "=C–H bend (terminal)",     "Alkene (terminal, =CH2)",      0.85),

    # ── ALKYNES ─────────────────────────────────────────────────────────
    (3333, 3267, "≡C–H stretch",             "Terminal Alkyne",              0.95),
    (2260, 2100, "C≡C stretch",              "Alkyne",                       0.90),
    ( 700,  610, "≡C–H bend",               "Terminal Alkyne",              0.85),

    # ── NITRILES ────────────────────────────────────────────────────────
    (2260, 2210, "C≡N stretch",              "Nitrile",                      0.95),

    # ── ISOCYANATES / ISOTHIOCYANATES ───────────────────────────────────
    (2275, 2250, "N=C=O asym stretch",       "Isocyanate",                   0.97),
    (2140, 2050, "N=C=S asym stretch",       "Isothiocyanate",               0.90),

    # ── THIOLS & SULFUR ─────────────────────────────────────────────────
    (2600, 2550, "S–H stretch",              "Thiol",                        0.90),
    (1430, 1300, "S=O stretch",              "Sulfone / Sulfoxide",          0.78),
    (1200, 1145, "S=O sym stretch",          "Sulfone",                      0.80),
    (1075, 1030, "S=O stretch",              "Sulfoxide",                    0.78),

    # ── PHOSPHORUS COMPOUNDS ────────────────────────────────────────────
    (2400, 2300, "P–H stretch",              "Phosphine",                    0.85),
    (1300, 1240, "P=O stretch",              "Phosphine Oxide / Phosphate",  0.85),
    (1050,  950, "P–O–C stretch",            "Phosphate Ester",              0.80),
    ( 900,  750, "P–O stretch",              "Phosphonate",                  0.72),

    # ── SILICON COMPOUNDS ───────────────────────────────────────────────
    (1260, 1200, "Si–CH3 bend",              "Organosilicon (silicone)",     0.88),
    (1100,  980, "Si–O–Si stretch",          "Siloxane",                     0.90),
    ( 840,  800, "Si–C stretch",             "Organosilicon",                0.82),

    # ── HALIDES ─────────────────────────────────────────────────────────
    (1400, 1000, "C–F stretch",              "Fluoroalkane",                 0.75),
    ( 800,  600, "C–Cl stretch",             "Chloroalkane",                 0.82),
    ( 600,  500, "C–Br stretch",             "Bromoalkane",                  0.85),
    ( 500,  400, "C–I stretch",              "Iodoalkane",                   0.85),

    # ── EPOXIDES ────────────────────────────────────────────────────────
    ( 950,  810, "C–O–C asym stretch",       "Epoxide ring",                 0.85),
    ( 880,  750, "Ring deformation",         "Epoxide ring",                 0.80),

    # ── POLYSACCHARIDES / CARBOHYDRATES ─────────────────────────────────
    (3600, 3000, "O–H stretch (broad)",      "Polysaccharide / Carbohydrate", 0.70),
    (1200,  950, "C–O–C glycosidic stretch", "Polysaccharide",               0.78),

    # ── POLYESTERS (PET-like) ────────────────────────────────────────────
    (1730, 1715, "C=O stretch",              "Polyester",                    0.82),
    (1285, 1240, "C–O–C stretch",            "Polyester (aromatic)",         0.80),
    (1125, 1095, "C–O stretch",              "Polyester (aliphatic)",        0.72),

    # ── FINGERPRINT REGION ──────────────────────────────────────────────
    (1000,  400, "—",                        "Fingerprint region / Unassigned", 0.10),
]


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL REGION LABELS (for plot shading)
# Format: { region_name: (wn_low, wn_high, hex_color, alpha) }
# ─────────────────────────────────────────────────────────────────────────────
REGION_LABELS = {
    "O–H / N–H":    (3700, 2500, "#FF6B6B", 0.07),
    "C–H stretch":  (3000, 2700, "#FFD93D", 0.07),
    "Triple bonds": (2700, 1900, "#6BCB77", 0.06),
    "C=O stretch":  (1900, 1600, "#4D96FF", 0.08),
    "C=C / N–H":    (1600, 1350, "#C77DFF", 0.07),
    "C–O / C–N":    (1350, 1000, "#FF9F1C", 0.06),
    "Fingerprint":  (1000,  400, "#AAAAAA", 0.05),
}


# ─────────────────────────────────────────────────────────────────────────────
# COMPOUND PREDICTION RULES
# ─────────────────────────────────────────────────────────────────────────────
COMPOUND_RULES = [
    {
        "compound":   "Aromatic Carboxylic Acid",
        "requires":   ["O–H stretch (broad)", "C=O stretch", "C=C ring stretch"],
        "excludes":   [],
        "confidence": 0.90,
        "notes":      "Broad OH + C=O near 1710 cm⁻¹ + ring bands confirm aromatic acid."
    },
    {
        "compound":   "Aliphatic Carboxylic Acid",
        "requires":   ["O–H stretch (broad)", "C=O stretch"],
        "excludes":   ["C=C ring stretch"],
        "confidence": 0.85,
        "notes":      "Broad 2500–3300 cm⁻¹ OH + C=O 1700–1725 cm⁻¹, no ring bands."
    },
    {
        "compound":   "Anhydride",
        "requires":   ["C=O stretch (asym)", "C=O stretch (sym)"],
        "excludes":   ["O–H stretch (broad)", "O–H stretch (H-bonded)"],
        "confidence": 0.95,
        "notes":      "Twin C=O peaks ~1850 and ~1780 cm⁻¹ are a hallmark of anhydrides."
    },
    {
        "compound":   "Acid Halide (Acyl Chloride)",
        "requires":   ["C=O stretch", "C–Cl stretch"],
        "excludes":   ["O–H stretch (broad)", "O–H stretch (H-bonded)"],
        "confidence": 0.90,
        "notes":      "High C=O (>1785) + C–Cl band."
    },
    {
        "compound":   "Ester",
        "requires":   ["C=O stretch", "C–O–C asym stretch"],
        "excludes":   ["O–H stretch (broad)", "O–H stretch (H-bonded)", "N–H stretch"],
        "confidence": 0.88,
        "notes":      "C=O ~1735 + C–O–C double band in 1000–1300 cm⁻¹."
    },
    {
        "compound":   "Lactone",
        "requires":   ["C=O stretch"],
        "excludes":   ["O–H stretch (broad)", "O–H stretch (H-bonded)"],
        "confidence": 0.80,
        "notes":      "Cyclic ester, C=O elevated to 1750–1795 cm⁻¹."
    },
    {
        "compound":   "Amide (Primary)",
        "requires":   ["C=O stretch", "N–H stretch", "N–H bend (scissor)"],
        "excludes":   ["O–H stretch (broad)"],
        "confidence": 0.88,
        "notes":      "Amide I ~1650, Amide II ~1550, free N–H doublet."
    },
    {
        "compound":   "Amide (Secondary)",
        "requires":   ["C=O stretch", "N–H bend"],
        "excludes":   ["O–H stretch (broad)", "N–H stretch (asym)", "N–H stretch (sym)"],
        "confidence": 0.80,
        "notes":      "Secondary amide shows single N–H band."
    },
    {
        "compound":   "Aldehyde",
        "requires":   ["C=O stretch", "C–H stretch (Fermi)"],
        "excludes":   ["O–H stretch (broad)"],
        "confidence": 0.92,
        "notes":      "Fermi doublet 2720–2820 cm⁻¹ is diagnostic for aldehydes."
    },
    {
        "compound":   "Ketone",
        "requires":   ["C=O stretch"],
        "excludes":   [
            "O–H stretch (broad)", "O–H stretch (H-bonded)",
            "C–O stretch", "C–H stretch (Fermi)",
            "N–H stretch", "C–O–C asym stretch"
        ],
        "confidence": 0.80,
        "notes":      "Isolated C=O ~1715 with no OH, no CO, no Fermi doublet."
    },
    {
        "compound":   "Carbamate / Urethane",
        "requires":   ["C=O stretch", "N–H bend", "C–O–C asym stretch"],
        "excludes":   ["O–H stretch (broad)"],
        "confidence": 0.82,
        "notes":      "C=O 1700–1730 + N–H Amide II + ether C–O band."
    },
    {
        "compound":   "Aromatic Alcohol / Phenol",
        "requires":   ["O–H stretch (H-bonded)", "C–O stretch", "C=C ring stretch"],
        "excludes":   ["C=O stretch"],
        "confidence": 0.88,
        "notes":      "H-bonded OH + C–O ~1230 cm⁻¹ + ring bands, no carbonyl."
    },
    {
        "compound":   "Aliphatic Alcohol",
        "requires":   ["O–H stretch (H-bonded)", "C–O stretch"],
        "excludes":   ["C=O stretch", "C=C ring stretch"],
        "confidence": 0.85,
        "notes":      "Broad OH + C–O stretch 1000–1260 cm⁻¹, no ring."
    },
    {
        "compound":   "Primary Amine",
        "requires":   ["N–H stretch (asym)", "N–H stretch (sym)", "N–H bend (scissor)"],
        "excludes":   ["C=O stretch"],
        "confidence": 0.88,
        "notes":      "Two N–H peaks + scissoring band at 1550–1650 cm⁻¹."
    },
    {
        "compound":   "Secondary Amine",
        "requires":   ["N–H stretch", "C–N stretch"],
        "excludes":   ["C=O stretch", "N–H stretch (sym)"],
        "confidence": 0.78,
        "notes":      "Single N–H band + C–N stretch."
    },
    {
        "compound":   "Nitro Compound",
        "requires":   ["N–O asym stretch", "N–O sym stretch"],
        "excludes":   [],
        "confidence": 0.95,
        "notes":      "Strong doublet: asymmetric ~1530, symmetric ~1350 cm⁻¹."
    },
    {
        "compound":   "Thiol",
        "requires":   ["S–H stretch"],
        "excludes":   [],
        "confidence": 0.90,
        "notes":      "Weak but sharp S–H band 2550–2600 cm⁻¹."
    },
    {
        "compound":   "Sulfone",
        "requires":   ["S=O sym stretch"],
        "excludes":   [],
        "confidence": 0.85,
        "notes":      "Two S=O bands: ~1300 and ~1150 cm⁻¹."
    },
    {
        "compound":   "Sulfoxide",
        "requires":   ["S=O stretch"],
        "excludes":   ["S=O sym stretch"],
        "confidence": 0.80,
        "notes":      "Single strong S=O ~1050 cm⁻¹."
    },
    {
        "compound":   "Ether",
        "requires":   ["C–O–C asym stretch"],
        "excludes":   ["C=O stretch", "O–H stretch (H-bonded)", "O–H stretch (broad)"],
        "confidence": 0.75,
        "notes":      "C–O–C strong band 1060–1150 cm⁻¹, no carbonyl or alcohol."
    },
    {
        "compound":   "Epoxide",
        "requires":   ["C–O–C asym stretch", "Ring deformation"],
        "excludes":   ["C=O stretch"],
        "confidence": 0.85,
        "notes":      "Epoxide ring bands ~910 and ~840 cm⁻¹."
    },
    {
        "compound":   "Terminal Alkyne",
        "requires":   ["≡C–H stretch", "C≡C stretch"],
        "excludes":   [],
        "confidence": 0.95,
        "notes":      "Sharp ≡C–H at 3300 cm⁻¹ + C≡C at 2100–2150 cm⁻¹."
    },
    {
        "compound":   "Internal Alkyne",
        "requires":   ["C≡C stretch"],
        "excludes":   ["≡C–H stretch"],
        "confidence": 0.85,
        "notes":      "C≡C present but no terminal ≡C–H."
    },
    {
        "compound":   "Nitrile",
        "requires":   ["C≡N stretch"],
        "excludes":   [],
        "confidence": 0.95,
        "notes":      "Sharp, strong C≡N ~2200–2260 cm⁻¹."
    },
    {
        "compound":   "Isocyanate",
        "requires":   ["N=C=O asym stretch"],
        "excludes":   [],
        "confidence": 0.97,
        "notes":      "Extremely strong, broad band ~2270 cm⁻¹."
    },
    {
        "compound":   "Aromatic Hydrocarbon",
        "requires":   ["=C–H stretch", "C=C ring stretch"],
        "excludes":   [
            "C=O stretch", "O–H stretch (H-bonded)",
            "O–H stretch (broad)", "C–O stretch"
        ],
        "confidence": 0.85,
        "notes":      "Aromatic C–H + ring overtones + OOP bending 700–900 cm⁻¹."
    },
    {
        "compound":   "Alkene",
        "requires":   ["=C–H stretch", "C=C stretch"],
        "excludes":   ["C=C ring stretch", "C=O stretch"],
        "confidence": 0.82,
        "notes":      "sp2 C–H + C=C 1620–1680 cm⁻¹."
    },
    {
        "compound":   "Aliphatic Hydrocarbon (Alkane)",
        "requires":   ["C–H stretch"],
        "excludes":   [
            "C=O stretch", "O–H stretch (H-bonded)",
            "O–H stretch (broad)", "C=C stretch",
            "=C–H stretch", "C–O stretch"
        ],
        "confidence": 0.78,
        "notes":      "Only sp3 C–H bands, no functional groups."
    },
    {
        "compound":   "Silicone Polymer",
        "requires":   ["Si–O–Si stretch", "Si–CH3 bend"],
        "excludes":   [],
        "confidence": 0.92,
        "notes":      "Strong Si–O at 1000–1100 cm⁻¹ + Si–CH3 at 1260 cm⁻¹."
    },
    {
        "compound":   "Phosphate Ester",
        "requires":   ["P=O stretch", "P–O–C stretch"],
        "excludes":   [],
        "confidence": 0.88,
        "notes":      "P=O ~1260 + P–O–C bands 950–1100 cm⁻¹."
    },
    {
        "compound":   "Polyester",
        "requires":   ["C=O stretch", "C–O–C asym stretch", "C–O–C sym stretch"],
        "excludes":   ["O–H stretch (broad)", "N–H stretch"],
        "confidence": 0.82,
        "notes":      "Polymeric ester: strong C=O + dual C–O–C stretches."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN COMPOUNDS (For Expected Compound Highlighting)
# Format: { "compound name (lowercase)": [(expected_wavenumber, "bond description"), ...] }
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_COMPOUNDS = {
    "benzoic acid": [
        (3000, "Broad O-H stretch (2500-3300)"),
        (1680, "C=O stretch (Aromatic carboxylic acid)"),
        (1600, "C=C Aromatic ring stretch"),
        (1580, "C=C Aromatic ring stretch"),
        (1450, "C=C Aromatic ring stretch"),
        (1290, "C-O stretch"),
        (930, "O-H out-of-plane bend (dimer)"),
        (710, "C-H out-of-plane bend (Mono-substituted benzene)")
    ],
    "peg": [
        (2880, "C-H aliphatic stretch"),
        (1465, "C-H bend (scissoring)"),
        (1340, "C-H bend"),
        (1100, "C-O-C ether stretch (strong, broad)")
    ],
    "polyethylene glycol": [
        (2880, "C-H aliphatic stretch"),
        (1465, "C-H bend (scissoring)"),
        (1340, "C-H bend"),
        (1100, "C-O-C ether stretch (strong, broad)")
    ],
    "ethanol": [
        (3300, "Broad O-H stretch"),
        (2970, "C-H stretch"),
        (1045, "C-O stretch (primary alcohol)"),
        (880, "C-C-O symmetric stretch")
    ],
    "acetone": [
        (1715, "Strong C=O stretch"),
        (2920, "C-H stretch"),
        (1360, "C-H bend (umbrella)"),
        (1220, "C-C-C stretch")
    ],
    "aspirin (acetylsalicylic acid)": [
        (3000, "Broad O-H stretch (carboxylic acid)"),
        (1750, "C=O stretch (ester)"),
        (1680, "C=O stretch (carboxylic acid)"),
        (1600, "C=C aromatic stretch"),
        (1190, "C-O-C ester stretch")
    ],
    "water (h2o)": [
        (3300, "O-H stretch (very broad)"),
        (1640, "H-O-H bend")
    ],
    "paracetamol (acetaminophen)": [
        (3320, "N-H / O-H stretch"),
        (1650, "C=O stretch (amide)"),
        (1560, "N-H bend (Amide II)"),
        (1500, "C=C aromatic stretch")
    ]
}
