/**
 * Causal Digital Twin — Dashboard JavaScript
 * Handles patient form submission and result rendering
 */

const FEATURE_FIELDS = [
    'age', 'sex', 'education_years', 'hba1c', 'systolic_bp',
    'diastolic_bp', 'ldl', 'physical_activity_min_week',
    'gait_speed', 'oct_rnfl_thickness', 'mmse_score', 'hrv_sdnn'
];

/** Load demo patient data */
async function loadDemoPatient() {
    try {
        const resp = await fetch('/demo-patient');
        const data = await resp.json();
        
        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = value !== null ? value : '';
            }
        }
    } catch (err) {
        // Fallback demo data
        const demo = {
            age: 65, sex: 1, education_years: 8, hba1c: 7.2,
            systolic_bp: 145, diastolic_bp: 88, ldl: 135,
            physical_activity_min_week: 90, gait_speed: 0.85,
            oct_rnfl_thickness: '', mmse_score: 25, hrv_sdnn: 38
        };
        for (const [key, value] of Object.entries(demo)) {
            const el = document.getElementById(key);
            if (el) el.value = value;
        }
    }
}

/** Collect form data into patient object */
function collectPatientData() {
    const data = {};
    for (const field of FEATURE_FIELDS) {
        const el = document.getElementById(field);
        if (el && el.value !== '' && el.value !== null) {
            data[field] = parseFloat(el.value);
        } else {
            data[field] = null;
        }
    }
    return data;
}

/** Run prediction */
async function runPrediction(event) {
    event.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');
    
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    btn.disabled = true;
    
    const patientData = collectPatientData();
    
    try {
        const resp = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });
        
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        
        const result = await resp.json();
        renderResults(result);
    } catch (err) {
        console.error('Prediction error:', err);
        alert('Prediction failed. Check console for details.');
    } finally {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        btn.disabled = false;
    }
}

/** Render prediction results */
function renderResults(result) {
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    
    // Inference time
    const timeEl = document.getElementById('inferenceTime');
    timeEl.textContent = `⚡ ${result.inference_time_ms}ms`;
    
    // Baseline risk
    const riskPct = result.baseline_risk_pct;
    const riskEl = document.getElementById('baselineRisk');
    riskEl.textContent = `${riskPct}%`;
    riskEl.style.color = getRiskColor(riskPct);
    
    const bar = document.getElementById('baselineBar');
    bar.style.width = `${Math.min(riskPct, 100)}%`;
    bar.style.background = getRiskGradient(riskPct);
    
    // Interventions
    const listEl = document.getElementById('interventionsList');
    listEl.innerHTML = '';
    
    for (const intervention of result.interventions) {
        const card = createInterventionCard(intervention, riskPct);
        listEl.appendChild(card);
    }
    
    // Combined
    document.getElementById('combinedRisk').textContent = `${result.combined_risk_pct}%`;
    document.getElementById('combinedReduction').textContent = 
        `↓ ${result.combined_reduction_pct}pp`;
}

/** Create an intervention card element */
function createInterventionCard(intervention, baselineRiskPct) {
    const card = document.createElement('div');
    card.className = 'intervention-card';
    
    const reductionClass = intervention.risk_reduction_pp >= 8 ? 'reduction-high' :
                           intervention.risk_reduction_pp >= 4 ? 'reduction-med' : 'reduction-low';
    
    const eClass = intervention.confidence === 'high' ? 'e-high' :
                   intervention.confidence === 'moderate' ? 'e-moderate' : 'e-low';
    
    card.innerHTML = `
        <div class="intervention-header">
            <div class="intervention-name">
                <span class="intervention-icon">${intervention.icon}</span>
                <span>${intervention.name}</span>
            </div>
            <span class="intervention-reduction ${reductionClass}">
                ↓ ${intervention.risk_reduction_pp}pp
            </span>
        </div>
        <div class="intervention-details">
            <div class="intervention-risk">
                <span>New risk: <span class="new-risk-value">${intervention.new_risk_pct}%</span></span>
                <span>•</span>
                <span>Monitor: ${intervention.monitoring}</span>
            </div>
            <span class="e-value-badge ${eClass}">
                E=${intervention.e_value}
            </span>
        </div>
        <div class="intervention-bar">
            <div class="intervention-bar-fill" style="
                width: ${Math.max(2, intervention.risk_reduction_pp / baselineRiskPct * 100)}%;
                background: ${getRiskGradient(intervention.new_risk_pct)};
            "></div>
        </div>
    `;
    
    return card;
}

/** Get risk color based on percentage */
function getRiskColor(pct) {
    if (pct < 20) return '#10b981';
    if (pct < 35) return '#f59e0b';
    return '#ef4444';
}

/** Get risk gradient based on percentage */
function getRiskGradient(pct) {
    if (pct < 20) return 'linear-gradient(90deg, #10b981, #34d399)';
    if (pct < 35) return 'linear-gradient(90deg, #f59e0b, #fbbf24)';
    return 'linear-gradient(90deg, #ef4444, #f87171)';
}

// Auto-load demo on page load for convenience
document.addEventListener('DOMContentLoaded', () => {
    console.log('Causal Digital Twin Dashboard loaded');
});
