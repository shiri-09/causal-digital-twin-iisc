"""
Clinician Dashboard — Flask Backend

Serves the landing page + touch-optimized clinician interface for
counterfactual therapy recommendations.

Endpoints:
    GET  /           → Landing page
    GET  /dashboard  → Clinician dashboard UI
    POST /predict    → Patient risk + counterfactual predictions
    GET  /health     → Health check
    GET  /demo-patient → Demo patient data
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from typing import Dict, Optional

from src.dashboard.inference import InferenceEngine


app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)

# Global inference engine
engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Lazy-load the inference engine."""
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine


@app.route('/')
def landing():
    """Serve the landing page."""
    return render_template('landing.html')


@app.route('/dashboard')
def dashboard():
    """Serve the clinician dashboard."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Run counterfactual prediction for a patient.
    
    Expected JSON body:
    {
        "age": 65,
        "sex": 1,
        "education_years": 8,
        "hba1c": 7.2,
        "systolic_bp": 145,
        "diastolic_bp": 88,
        "ldl": 135,
        "physical_activity_min_week": 90,
        "gait_speed": 0.85,
        "oct_rnfl_thickness": null,
        "mmse_score": 25,
        "hrv_sdnn": 38
    }
    """
    try:
        data = request.get_json()
        eng = get_engine()
        result = eng.predict_patient(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': engine is not None,
    })


@app.route('/demo-patient')
def demo_patient():
    """Return a demo patient for testing."""
    return jsonify({
        'age': 65,
        'sex': 1,
        'education_years': 8,
        'hba1c': 7.2,
        'systolic_bp': 145,
        'diastolic_bp': 88,
        'ldl': 135,
        'physical_activity_min_week': 90,
        'gait_speed': 0.85,
        'oct_rnfl_thickness': None,
        'mmse_score': 25,
        'hrv_sdnn': 38,
    })


def run_dashboard(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start the dashboard server."""
    print(f"\n{'='*60}")
    print(f"  Causal Digital Twin — Clinician Dashboard")
    print(f"  Landing:   http://{host}:{port}/")
    print(f"  Dashboard: http://{host}:{port}/dashboard")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard(debug=True)
