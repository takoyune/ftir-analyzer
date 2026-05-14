import pandas as pd
from ftir_database import FUNCTIONAL_GROUPS, COMPOUND_RULES, KNOWN_COMPOUNDS

class FTIRExpertSystem:
    def __init__(self):
        pass

    def analyze_spectrum(self, peak_wn):
        """
        Main entry point for AI analysis of peaks.
        Returns the functional group DataFrame, a list of predictions, and natural language insights.
        """
        fg_table = self._identify_functional_groups(peak_wn)
        predictions, insights = self._predict_compound(fg_table, peak_wn)
        return fg_table, predictions, insights

    def _identify_functional_groups(self, peak_wavenumbers):
        """Match peak wavenumbers to functional groups with confidence."""
        results = []
        for wn in peak_wavenumbers:
            matched = False
            for wn_high, wn_low, bond, group, conf in FUNCTIONAL_GROUPS:
                if wn_low <= wn <= wn_high:
                    results.append({
                        'Wavenumber (cm⁻¹)': f'{wn:.1f}',
                        'Bond': bond,
                        'Functional Group': group,
                        'Confidence': conf
                    })
                    matched = True
                    break
            if not matched:
                results.append({
                    'Wavenumber (cm⁻¹)': round(wn, 1),
                    'Bond': '—',
                    'Functional Group': 'Fingerprint region / Unassigned',
                    'Confidence': 0.0
                })
                
        if not results:
            return pd.DataFrame(columns=['Wavenumber (cm⁻¹)', 'Bond', 'Functional Group', 'Confidence'])
            
        return pd.DataFrame(results)

    def _predict_compound(self, fg_table, peak_wn):
        """Predict compound class based on a weighted scoring system and generate AI insights."""
        found_bonds = fg_table['Bond'].tolist() if not fg_table.empty else []
        found_confs = fg_table['Confidence'].tolist() if not fg_table.empty else []
        
        # Map bond to its highest confidence in the found peaks
        bond_confidence = {}
        for bond, conf in zip(found_bonds, found_confs):
            if bond != '—':
                bond_confidence[bond] = max(bond_confidence.get(bond, 0.0), conf)
                
        scored_predictions = []
        
        # 1. Check General Compound Classes
        for rule in COMPOUND_RULES:
            reqs = rule["requires"]
            excs = rule["excludes"]
            base_conf = rule["confidence"]
            
            if not reqs:
                continue
                
            req_score = 0
            for req in reqs:
                matched_bonds = [b for b in bond_confidence.keys() if req in b]
                if matched_bonds:
                    req_score += max(bond_confidence[b] for b in matched_bonds)
                    
            req_ratio = req_score / len(reqs) if reqs else 0
            
            actual_req_count = sum(1 for req in reqs if any(req in b for b in bond_confidence.keys()))
            coverage = actual_req_count / len(reqs)
            
            exc_matches = sum(1 for exc in excs if any(exc in b for b in bond_confidence.keys()))
            
            score = base_conf * req_ratio * coverage
            
            if exc_matches > 0:
                score *= (0.2 ** exc_matches)  # Heavy penalty
                
            if score > 0.35:
                scored_predictions.append({
                    "name": rule["compound"],
                    "score": score,
                    "type": "Class",
                    "notes": rule["notes"]
                })

        # 2. Check Specific Known Compounds
        for comp_name, expected_peaks in KNOWN_COMPOUNDS.items():
            if not expected_peaks:
                continue
                
            matched_peaks = 0
            total_weight = 0
            current_weight = 0
            
            for expected_wn, desc in expected_peaks:
                weight = 1.0
                total_weight += weight
                
                # Dynamic tolerance: broad bands can shift significantly
                tolerance = 150 if "broad" in desc.lower() else 30
                
                is_matched = any(abs(wn - expected_wn) <= tolerance for wn in peak_wn)
                if is_matched:
                    matched_peaks += 1
                    current_weight += weight
                    
            if total_weight > 0:
                match_ratio = current_weight / total_weight
                if match_ratio >= 0.5:
                    score = match_ratio * 0.95 
                    scored_predictions.append({
                        "name": comp_name.title(),
                        "score": score,
                        "type": "Specific Compound",
                        "notes": f"Matched {matched_peaks}/{len(expected_peaks)} signature peaks."
                    })
                
        scored_predictions.sort(key=lambda x: x["score"], reverse=True)
        
        predictions = []
        insights = []
        
        # 3. Generate AI Insights
        if len(peak_wn) >= 12:
            insights.append("High number of major peaks detected. This often indicates a highly complex molecule, an aromatic system, or a mixture of compounds.")
            
        if any(3200 <= wn <= 3600 for wn in peak_wn):
            insights.append("A broad peak in the 3200-3600 cm⁻¹ region strongly suggests the presence of O-H (alcohol, phenol, water) or N-H bonds.")
            
        if any(1680 <= wn <= 1750 for wn in peak_wn):
            insights.append("A strong, sharp peak in the 1680-1750 cm⁻¹ region indicates a Carbonyl (C=O) group. This is highly characteristic of esters, ketones, aldehydes, or carboxylic acids.")
            
        if any(2100 <= wn <= 2260 for wn in peak_wn):
            insights.append("Peak(s) in the 2100-2260 cm⁻¹ range are characteristic of triple bonds (C≡C, C≡N) or cumulative double bonds.")

        if scored_predictions:
            # --- SMART OVERRIDE LOGIC ---
            # If we find a highly confident Specific Compound, suppress the generic Class predictions
            best_specific = next((p for p in scored_predictions if p['type'] == 'Specific Compound'), None)
            if best_specific and best_specific['score'] >= 0.70:
                # Filter out generic classes, keep only specific compounds or absolutely perfect class matches
                scored_predictions = [p for p in scored_predictions if p['type'] == 'Specific Compound' or p['score'] >= 0.95]
                
            best_score = scored_predictions[0]["score"]
            best_match = scored_predictions[0]
            
            if best_score > 0.85:
                insights.append(f"AI DIAGNOSIS: Extremely high confidence match for {best_match['name']}. The spectral fingerprint is highly distinctive.")
            elif best_score > 0.6:
                insights.append(f"AI DIAGNOSIS: Moderate confidence match for {best_match['name']}. Some expected bands might be obscured, shifted, or mixed with impurities.")
            else:
                insights.append(f"AI DIAGNOSIS: Low confidence matches. The sample may be contaminated, mixed, or a novel compound not well-represented in the database.")
            
            for p in scored_predictions:
                if p["score"] >= 0.6 or (p["score"] >= best_score * 0.75 and p["score"] > 0.4):
                    tag = f"[{p['type']}] " if p['type'] == 'Specific Compound' else ""
                    predictions.append(f"{tag}{p['name']} (Confidence: {p['score']*100:.1f}%) - {p['notes']}")
                    
            predictions = predictions[:4]
                
        if not predictions:
            if any("O–H stretch" in b for b in bond_confidence.keys()):
                predictions.append("Unknown Oxygenated Compound (contains O-H)")
            elif any("C=O stretch" in b for b in bond_confidence.keys()):
                predictions.append("Unknown Carbonyl Compound")
            elif any("C–O stretch" in b for b in bond_confidence.keys()):
                predictions.append("Unknown Oxygenated Compound (contains C-O)")
            else:
                predictions.append("Unidentified / Complex Mixture")
                insights.append("AI DIAGNOSIS: The spectrum lacks distinctive functional groups. This could be a simple aliphatic hydrocarbon, an inorganic compound, or highly degraded data.")
                
        return predictions, insights
