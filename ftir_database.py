"""
FTIR Functional Group and Compound Database

Comprehensive reference table of IR vibrational frequencies.
"""

FUNCTIONAL_GROUPS = [
    # ──────────────────────────────────────────
    # ALCOHOLS & PHENOLS
    # ──────────────────────────────────────────
    (3650, 3584, "O–H stretch (free)", "Alcohol (free)"),
    (3550, 3200, "O–H stretch (H-bonded)", "Alcohol / Phenol"),
    (1420, 1330, "O–H bend", "Alcohol / Phenol"),
    (1260, 1000, "C–O stretch", "Alcohol / Ether / Ester"),

    # ──────────────────────────────────────────
    # AMINES & AMIDES
    # ──────────────────────────────────────────
    (3500, 3300, "N–H stretch", "Primary / Secondary Amine or Amide"),
    (1650, 1550, "N–H bend", "Amine / Amide"),
    (1360, 1250, "C–N stretch (Aromatic)", "Aromatic Amine"),
    (1250, 1020, "C–N stretch (Aliphatic)", "Aliphatic Amine"),
    (910, 665,   "N–H wag", "Primary / Secondary Amine"),

    # ──────────────────────────────────────────
    # ALKYNES & NITRILES
    # ──────────────────────────────────────────
    (3330, 3250, "≡C–H stretch", "Terminal Alkyne"),
    (2260, 2220, "C≡N stretch", "Nitrile"),
    (2260, 2100, "C≡C stretch", "Alkyne"),
    (700, 610,   "≡C–H bend", "Terminal Alkyne"),

    # ──────────────────────────────────────────
    # CARBOXYLIC ACIDS & DERIVATIVES
    # ──────────────────────────────────────────
    (3300, 2500, "O–H stretch (broad)", "Carboxylic Acid"),
    (1830, 1800, "C=O stretch (sym)", "Anhydride"),
    (1775, 1740, "C=O stretch (asym)", "Anhydride"),
    (1815, 1785, "C=O stretch", "Acid Halide"),
    (1750, 1735, "C=O stretch", "Ester"),
    (1725, 1700, "C=O stretch", "Carboxylic Acid"),
    (1690, 1650, "C=O stretch", "Amide"),

    # ──────────────────────────────────────────
    # ALDEHYDES & KETONES
    # ──────────────────────────────────────────
    (2830, 2695, "C–H stretch (Fermi)", "Aldehyde"),
    (1740, 1720, "C=O stretch", "Aldehyde"),
    (1725, 1705, "C=O stretch", "Ketone"),

    # ──────────────────────────────────────────
    # ALKANES
    # ──────────────────────────────────────────
    (3000, 2850, "C–H stretch", "Alkane"),
    (1470, 1450, "C–H bend (scissoring)", "Alkane (CH2)"),
    (1385, 1370, "C–H bend (umbrella)", "Alkane (CH3)"),
    (725, 720,   "C–H rocking", "Alkane (long chain)"),
    
    # ──────────────────────────────────────────
    # AROMATICS & ALKENES
    # ──────────────────────────────────────────
    (3150, 3000, "=C–H stretch", "Aromatic / Alkene"),
    (1680, 1620, "C=C stretch", "Alkene"),
    (1600, 1585, "C=C ring stretch", "Aromatic Ring"),
    (1510, 1480, "C=C ring stretch", "Aromatic Ring"),
    (1000, 960,  "=C–H bend (trans)", "Alkene (trans)"),
    (915, 885,   "=C–H bend (terminal)", "Alkene (terminal)"),
    (900, 690,   "=C–H out-of-plane bend", "Aromatic Ring"),

    # ──────────────────────────────────────────
    # SULFUR COMPOUNDS
    # ──────────────────────────────────────────
    (2600, 2550, "S–H stretch", "Thiol"),
    (1420, 1300, "S=O stretch", "Sulfone / Sulfoxide"),
    (1200, 1145, "S=O stretch (sym)", "Sulfone"),
    (1070, 1030, "S=O stretch", "Sulfoxide"),

    # ──────────────────────────────────────────
    # NITROGEN COMPOUNDS (NON-AMINE)
    # ──────────────────────────────────────────
    (1560, 1515, "N–O asym stretch", "Nitro Compound"),
    (1385, 1345, "N–O sym stretch", "Nitro Compound"),
    (1660, 1500, "N=O stretch", "Nitroso Compound"),

    # ──────────────────────────────────────────
    # HALIDES
    # ──────────────────────────────────────────
    (1400, 1000, "C–F stretch", "Fluoroalkane"),
    (800, 600,   "C–Cl stretch", "Chloroalkane"),
    (600, 500,   "C–Br stretch", "Bromoalkane"),
    (500, 400,   "C–I stretch", "Iodoalkane"),

    # ──────────────────────────────────────────
    # PHOSPHORUS & SILICON
    # ──────────────────────────────────────────
    (1300, 1240, "P=O stretch", "Phosphine Oxide / Phosphate"),
    (1250, 1200, "Si–CH3 bend", "Organosilicon"),
    (1100, 1000, "Si–O stretch", "Siloxane / Silicate"),
    
    # ──────────────────────────────────────────
    # FINGERPRINT REGION
    # ──────────────────────────────────────────
    (1000, 400, "—", "Fingerprint region / Unassigned")
]


# Logic Rules to Identify the exact Compound
COMPOUND_RULES = [
    {
        "compound": "Aromatic Carboxylic Acid",
        "requires": ["O–H stretch (broad)", "C=O stretch", "C=C ring stretch"],
        "excludes": []
    },
    {
        "compound": "Aliphatic Carboxylic Acid",
        "requires": ["O–H stretch (broad)", "C=O stretch"],
        "excludes": ["C=C ring stretch"]
    },
    {
        "compound": "Anhydride",
        "requires": ["C=O stretch (sym)", "C=O stretch (asym)"],
        "excludes": ["O–H stretch (broad)"]
    },
    {
        "compound": "Acid Halide",
        "requires": ["C=O stretch", "C–Cl stretch"],
        "excludes": ["O–H stretch (broad)", "O–H stretch (H-bonded)"]
    },
    {
        "compound": "Ester",
        "requires": ["C=O stretch", "C–O stretch"],
        "excludes": ["O–H stretch (broad)", "O–H stretch (H-bonded)"]
    },
    {
        "compound": "Amide",
        "requires": ["C=O stretch", "N–H stretch", "N–H bend"],
        "excludes": []
    },
    {
        "compound": "Aldehyde",
        "requires": ["C=O stretch", "C–H stretch (Fermi)"],
        "excludes": ["O–H stretch (broad)"]
    },
    {
        "compound": "Ketone",
        "requires": ["C=O stretch"],
        "excludes": ["O–H stretch (broad)", "O–H stretch (H-bonded)", "C–O stretch", "C–H stretch (Fermi)"]
    },
    {
        "compound": "Aromatic Alcohol / Phenol",
        "requires": ["O–H stretch (H-bonded)", "C–O stretch", "C=C ring stretch"],
        "excludes": ["C=O stretch"]
    },
    {
        "compound": "Aliphatic Alcohol",
        "requires": ["O–H stretch (H-bonded)", "C–O stretch"],
        "excludes": ["C=O stretch", "C=C ring stretch"]
    },
    {
        "compound": "Primary/Secondary Amine",
        "requires": ["N–H stretch", "C–N stretch"],
        "excludes": ["C=O stretch"]
    },
    {
        "compound": "Nitro Compound",
        "requires": ["N–O asym stretch", "N–O sym stretch"],
        "excludes": []
    },
    {
        "compound": "Thiol",
        "requires": ["S–H stretch"],
        "excludes": []
    },
    {
        "compound": "Sulfone or Sulfoxide",
        "requires": ["S=O stretch"],
        "excludes": []
    },
    {
        "compound": "Ether",
        "requires": ["C–O stretch"],
        "excludes": ["C=O stretch", "O–H stretch (H-bonded)", "O–H stretch (broad)"]
    },
    {
        "compound": "Terminal Alkyne",
        "requires": ["≡C–H stretch", "C≡C stretch"],
        "excludes": []
    },
    {
        "compound": "Internal Alkyne",
        "requires": ["C≡C stretch"],
        "excludes": ["≡C–H stretch"]
    },
    {
        "compound": "Nitrile",
        "requires": ["C≡N stretch"],
        "excludes": []
    },
    {
        "compound": "Aromatic Hydrocarbon",
        "requires": ["=C–H stretch", "C=C ring stretch"],
        "excludes": ["C=O stretch", "O–H stretch (H-bonded)", "O–H stretch (broad)", "C–O stretch"]
    },
    {
        "compound": "Alkene",
        "requires": ["=C–H stretch", "C=C stretch"],
        "excludes": ["C=C ring stretch", "C=O stretch", "O–H stretch (H-bonded)", "O–H stretch (broad)"]
    },
    {
        "compound": "Aliphatic Hydrocarbon (Alkane)",
        "requires": ["C–H stretch"],
        "excludes": ["C=O stretch", "O–H stretch (H-bonded)", "O–H stretch (broad)", "C=C stretch", "=C–H stretch", "C–O stretch"]
    }
]
