"""
ADDI Data Harmonization — C-Surv Variable Mapping

Maps SANSCOG/TLSA variables to the ADDI (Alzheimer's Disease Data
Initiative) C-Surv taxonomy for interoperability with the AD Workbench.

C-Surv taxonomy (4-level acyclic hierarchy):
    Theme → Domain → Family → Object

This module provides:
    1. A complete mapping of SANSCOG features to C-Surv codes
    2. Export utilities for ADDI-compatible data dictionaries
    3. Validation of required harmonization fields

Reference:
    ADDI Data Harmonization Group. "C-Surv Data Model for
    Neurodegeneration Research Cohorts." alzheimersdata.org
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# C-Surv Taxonomy Mapping: SANSCOG → ADDI
# ---------------------------------------------------------------------------
# Format: sanscog_variable → {theme, domain, family, object, unit, type}

SANSCOG_TO_CSURV = {
    # ── Demographics ──
    'age': {
        'theme': 'Participant',
        'domain': 'Demographics',
        'family': 'Age',
        'object': 'age_at_assessment',
        'unit': 'years',
        'type': 'continuous',
        'description': 'Age at time of cognitive assessment',
    },
    'sex': {
        'theme': 'Participant',
        'domain': 'Demographics',
        'family': 'Sex',
        'object': 'biological_sex',
        'unit': 'binary (0=male, 1=female)',
        'type': 'categorical',
        'description': 'Biological sex assigned at birth',
    },
    'education_years': {
        'theme': 'Participant',
        'domain': 'Demographics',
        'family': 'Education',
        'object': 'education_years_completed',
        'unit': 'years',
        'type': 'continuous',
        'description': 'Total years of formal education completed',
    },
    
    # ── Clinical / Metabolic ──
    'hba1c': {
        'theme': 'Clinical',
        'domain': 'Blood Biochemistry',
        'family': 'Glycemic Markers',
        'object': 'hba1c_percentage',
        'unit': '%',
        'type': 'continuous',
        'description': 'Glycated hemoglobin (HbA1c) percentage',
    },
    'fasting_glucose': {
        'theme': 'Clinical',
        'domain': 'Blood Biochemistry',
        'family': 'Glycemic Markers',
        'object': 'fasting_plasma_glucose',
        'unit': 'mg/dL',
        'type': 'continuous',
        'description': 'Fasting plasma glucose level',
    },
    
    # ── Cardiovascular ──
    'systolic_bp': {
        'theme': 'Clinical',
        'domain': 'Cardiovascular',
        'family': 'Blood Pressure',
        'object': 'systolic_bp_seated',
        'unit': 'mmHg',
        'type': 'continuous',
        'description': 'Systolic blood pressure, seated, mean of 2 readings',
    },
    'diastolic_bp': {
        'theme': 'Clinical',
        'domain': 'Cardiovascular',
        'family': 'Blood Pressure',
        'object': 'diastolic_bp_seated',
        'unit': 'mmHg',
        'type': 'continuous',
        'description': 'Diastolic blood pressure, seated, mean of 2 readings',
    },
    'ldl': {
        'theme': 'Clinical',
        'domain': 'Blood Biochemistry',
        'family': 'Lipid Panel',
        'object': 'ldl_cholesterol',
        'unit': 'mg/dL',
        'type': 'continuous',
        'description': 'Low-density lipoprotein cholesterol',
    },
    
    # ── Physical Function ──
    'physical_activity_min_week': {
        'theme': 'Clinical',
        'domain': 'Physical Function',
        'family': 'Physical Activity',
        'object': 'moderate_vigorous_activity_minutes_per_week',
        'unit': 'min/week',
        'type': 'continuous',
        'description': 'Self-reported moderate-to-vigorous physical activity per week',
    },
    'gait_speed': {
        'theme': 'Clinical',
        'domain': 'Physical Function',
        'family': 'Gait',
        'object': 'gait_speed_usual_pace',
        'unit': 'm/s',
        'type': 'continuous',
        'description': 'Usual-pace gait speed over 4-meter walkway',
    },
    
    # ── Ophthalmology ──
    'oct_rnfl_thickness': {
        'theme': 'Clinical',
        'domain': 'Ophthalmology',
        'family': 'Optical Coherence Tomography',
        'object': 'rnfl_thickness_average',
        'unit': 'μm',
        'type': 'continuous',
        'description': 'Average retinal nerve fiber layer thickness (OCT)',
    },
    
    # ── Cognitive ──
    'mmse_score': {
        'theme': 'Cognition',
        'domain': 'Cognitive Screening',
        'family': 'Global Cognition',
        'object': 'mmse_total_score',
        'unit': 'score (0-30)',
        'type': 'ordinal',
        'description': 'Mini-Mental State Examination total score',
    },
    'cdr_global': {
        'theme': 'Cognition',
        'domain': 'Cognitive Screening',
        'family': 'Dementia Staging',
        'object': 'cdr_global_score',
        'unit': 'score (0-3)',
        'type': 'ordinal',
        'description': 'Clinical Dementia Rating global score',
    },
    'hmse_score': {
        'theme': 'Cognition',
        'domain': 'Cognitive Screening',
        'family': 'Global Cognition',
        'object': 'hmse_total_score',
        'unit': 'score (0-30)',
        'type': 'ordinal',
        'description': 'Hindi Mental State Examination (SANSCOG adaptation)',
    },
    
    # ── Autonomic ──
    'hrv_sdnn': {
        'theme': 'Clinical',
        'domain': 'Autonomic Function',
        'family': 'Heart Rate Variability',
        'object': 'hrv_sdnn_5min',
        'unit': 'ms',
        'type': 'continuous',
        'description': 'Standard deviation of NN intervals (5-min recording)',
    },
    
    # ── Outcome ──
    'mci_2year': {
        'theme': 'Outcome',
        'domain': 'Cognitive Outcome',
        'family': 'Mild Cognitive Impairment',
        'object': 'mci_incident_2year',
        'unit': 'binary (0/1)',
        'type': 'binary',
        'description': 'Incident MCI at 2-year follow-up (Petersen criteria)',
    },
}


def get_csurv_mapping() -> Dict:
    """Return the full SANSCOG → C-Surv variable mapping."""
    return SANSCOG_TO_CSURV.copy()


def export_data_dictionary(
    output_path: str = "data/addi_data_dictionary.json",
    cohort_name: str = "SANSCOG",
    cohort_description: str = "Srinivaspura Aging, Neuro Senescence and COGnition Study",
) -> str:
    """
    Export an ADDI-compatible data dictionary in JSON format.
    
    This can be uploaded to the AD Workbench for dataset registration.
    """
    dictionary = {
        'schema_version': '1.0',
        'cohort': {
            'name': cohort_name,
            'description': cohort_description,
            'country': 'India',
            'region': 'Kolar, Karnataka',
            'type': 'population-based longitudinal',
            'n_participants': 10013,
            'start_year': 2018,
            'follow_up_interval': 'biennial',
        },
        'variables': [],
    }
    
    for var_name, mapping in SANSCOG_TO_CSURV.items():
        entry = {
            'source_variable': var_name,
            'csurv_path': f"{mapping['theme']}/{mapping['domain']}/{mapping['family']}/{mapping['object']}",
            'csurv_theme': mapping['theme'],
            'csurv_domain': mapping['domain'],
            'csurv_family': mapping['family'],
            'csurv_object': mapping['object'],
            'unit': mapping['unit'],
            'data_type': mapping['type'],
            'description': mapping['description'],
        }
        dictionary['variables'].append(entry)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dictionary, f, indent=2)
    
    print(f"  ✓ ADDI data dictionary exported: {output_path}")
    print(f"    {len(dictionary['variables'])} variables mapped to C-Surv taxonomy")
    
    return output_path


def validate_harmonization(df, required_vars: Optional[List[str]] = None) -> Dict:
    """
    Validate that a DataFrame has the required harmonized variables.
    
    Returns a report of present/missing/unmapped variables.
    """
    if required_vars is None:
        required_vars = list(SANSCOG_TO_CSURV.keys())
    
    present = [v for v in required_vars if v in df.columns]
    missing = [v for v in required_vars if v not in df.columns]
    unmapped = [c for c in df.columns if c not in SANSCOG_TO_CSURV]
    
    return {
        'total_required': len(required_vars),
        'present': len(present),
        'missing': len(missing),
        'missing_vars': missing,
        'unmapped_in_data': unmapped[:10],  # first 10
        'completeness': len(present) / max(1, len(required_vars)),
    }


if __name__ == "__main__":
    print("ADDI Data Harmonization — C-Surv Variable Mapping")
    print("=" * 55)
    
    mapping = get_csurv_mapping()
    
    # Show taxonomy tree
    themes = {}
    for var, m in mapping.items():
        key = f"{m['theme']}/{m['domain']}"
        if key not in themes:
            themes[key] = []
        themes[key].append(var)
    
    for path, vars in sorted(themes.items()):
        print(f"\n  {path}:")
        for v in vars:
            print(f"    → {v} [{mapping[v]['type']}]")
    
    print(f"\n  Total: {len(mapping)} variables mapped")
    
    # Export dictionary
    export_data_dictionary("data/addi_data_dictionary.json")
