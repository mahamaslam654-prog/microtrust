import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const API_URL = "http://127.0.0.1:8000";

const defaultForm = {
  person_age: 30,
  person_income: 360000,
  person_emp_length: 3,
  family_size: 4,
  dependents: 2,
  loan_amnt: 50000,
  loan_int_rate: 24,
  loan_percent_income: 0.14,
  loan_tenor_months: 12,
  payment_streak_months: 8,
  bills_paid_ontime_12mo: 9,
  bills_late_12mo: 3,
  utility_units_per_month: 150,
  monthly_bill_pkr: 4000,
  mobile_txn_per_month: 10,
  peer_defaults: 1,
  cb_cred_hist_length_years: 2,
  has_mobile_wallet: 1,
  has_smartphone: 1,
  has_guarantor: 0,
  income_seasonal: 0,
  is_unbanked: 1,
  wb_account_ownership_pct: 27.3,
  wb_gdp_per_capita_usd: 1478.0,
  wb_gini_index: 33.5,
  wb_bank_branches_per_100k: 10.8,
  occupation: "small shopkeeper",
  city: "lahore",
  urban_type: "urban",
  house_ownership: "rent",
  loan_intent: "working_capital",
  wallet_provider: "jazzcash",
  gender: "male",
  cb_person_default_on_file: "n",
  nepra_income_tier: "low",
  data_source: "api_input",
};

function ScoreGauge({ score }) {
  const color = score >= 65 ? "#2ecc71" : score >= 40 ? "#f39c12" : "#e74c3c";
  const band  = score >= 65 ? "Low Risk" : score >= 40 ? "Medium Risk" : "High Risk";
  return (
    <div style={{ textAlign: "center", padding: "30px" }}>
      <div style={{
        fontSize: "72px", fontWeight: "bold", color,
        border: `6px solid ${color}`, borderRadius: "50%",
        width: "160px", height: "160px", lineHeight: "160px",
        margin: "0 auto"
      }}>
        {score}
      </div>
      <div style={{ fontSize: "22px", fontWeight: "bold", color, marginTop: "12px" }}>
        {band}
      </div>
      <div style={{ color: "#888", marginTop: "4px" }}>Trust Score (0–100)</div>
    </div>
  );
}

export default function App() {
  const [form, setForm]       = useState(defaultForm);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(f => ({ ...f, [name]: isNaN(value) ? value : Number(value) }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res  = await fetch(`${API_URL}/predict`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify(form),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Could not connect to API. Make sure uvicorn is running.");
    }
    setLoading(false);
  };

  const inputStyle = {
    width: "100%", padding: "6px 8px", marginTop: "4px",
    border: "1px solid #ddd", borderRadius: "6px", fontSize: "13px"
  };
  const labelStyle = { fontSize: "12px", color: "#555", marginTop: "10px", display: "block" };

  const fields = [
    ["person_age",              "Age"],
    ["person_income",           "Annual Income (PKR)"],
    ["loan_amnt",               "Loan Amount (PKR)"],
    ["loan_int_rate",           "Interest Rate (%)"],
    ["payment_streak_months",   "Payment Streak (months)"],
    ["bills_paid_ontime_12mo",  "Bills Paid On Time (last 12mo)"],
    ["bills_late_12mo",         "Bills Paid Late (last 12mo)"],
    ["utility_units_per_month", "Utility Units/Month"],
    ["mobile_txn_per_month",    "Mobile Transactions/Month"],
    ["has_mobile_wallet",       "Has Mobile Wallet (1/0)"],
    ["has_guarantor",           "Has Guarantor (1/0)"],
    ["peer_defaults",           "Peer Defaults (0–5)"],
    ["loan_tenor_months",       "Loan Tenor (months)"],
    ["person_emp_length",       "Employment Length (years)"],
  ];

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: "1100px",
                  margin: "0 auto", padding: "24px" }}>

      {/* Header */}
      <div style={{ background: "#1a1a2e", color: "white", padding: "20px 28px",
                    borderRadius: "12px", marginBottom: "24px" }}>
        <h1 style={{ margin: 0, fontSize: "24px" }}>🏦 MicroTrust</h1>
        <p style={{ margin: "4px 0 0", color: "#aaa", fontSize: "14px" }}>
          Alternative Credit Risk Portal for Micro-Finance Officers
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>

        {/* Input Form */}
        <div style={{ background: "white", border: "1px solid #eee",
                      borderRadius: "12px", padding: "20px" }}>
          <h3 style={{ marginTop: 0 }}>Applicant Details</h3>

          {fields.map(([key, label]) => (
            <div key={key}>
              <label style={labelStyle}>{label}</label>
              <input name={key} value={form[key]} onChange={handleChange}
                     type="number" style={inputStyle} />
            </div>
          ))}

          <label style={labelStyle}>Occupation</label>
          <select name="occupation" value={form.occupation}
                  onChange={handleChange} style={inputStyle}>
            {["daily wage laborer","rickshaw/bike driver","street food vendor",
              "small shopkeeper","home-based worker","domestic worker",
              "informal factory worker","construction worker",
              "small farmer","motorcycle mechanic"].map(o => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>

          <label style={labelStyle}>City</label>
          <select name="city" value={form.city}
                  onChange={handleChange} style={inputStyle}>
            {["lahore","karachi","rawalpindi","faisalabad","multan",
              "peshawar","gujranwala","sialkot","bahawalpur"].map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>

          <label style={labelStyle}>NEPRA Income Tier</label>
          <select name="nepra_income_tier" value={form.nepra_income_tier}
                  onChange={handleChange} style={inputStyle}>
            {["lifeline","very_low","low","lower_middle",
              "middle","upper_middle","high","affluent"].map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>

          <button onClick={handleSubmit} disabled={loading}
                  style={{
                    width: "100%", marginTop: "20px", padding: "12px",
                    background: loading ? "#aaa" : "#1a1a2e",
                    color: "white", border: "none", borderRadius: "8px",
                    fontSize: "15px", cursor: loading ? "not-allowed" : "pointer"
                  }}>
            {loading ? "Calculating..." : "Calculate Trust Score"}
          </button>

          {error && (
            <div style={{ marginTop: "12px", color: "#e74c3c",
                          background: "#fdf0f0", padding: "10px",
                          borderRadius: "6px", fontSize: "13px" }}>
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div style={{ background: "white", border: "1px solid #eee",
                      borderRadius: "12px", padding: "20px" }}>
          <h3 style={{ marginTop: 0 }}>Trust Score Result</h3>

          {!result && !loading && (
            <div style={{ color: "#aaa", textAlign: "center",
                          marginTop: "60px", fontSize: "14px" }}>
              Fill in applicant details and click Calculate
            </div>
          )}

          {result && (
            <>
              <ScoreGauge score={result.trust_score} />

              {/* Recommendation */}
              <div style={{
                marginTop: "16px", padding: "12px 16px",
                background: result.color === "green" ? "#eafaf1" :
                            result.color === "amber" ? "#fef9e7" : "#fdf0f0",
                borderRadius: "8px", fontSize: "14px",
                borderLeft: `4px solid ${
                  result.color === "green" ? "#2ecc71" :
                  result.color === "amber" ? "#f39c12" : "#e74c3c"}`
              }}>
                <strong>Recommendation:</strong> {result.recommendation}
              </div>

              {/* SHAP bar chart */}
              <div style={{ marginTop: "20px" }}>
                <h4 style={{ marginBottom: "8px" }}>Top Risk Factors</h4>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart
                    data={result.top5_shap.map(s => ({
                      name  : s.feature.replace(/_/g," ").slice(0,20),
                      impact: Math.abs(s.shap_value),
                      fill  : s.shap_value > 0 ? "#e74c3c" : "#2ecc71"
                    }))}
                    layout="vertical"
                    margin={{ left: 10 }}
                  >
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name"
                           tick={{ fontSize: 10 }} width={130} />
                    <Tooltip />
                    <Bar dataKey="impact" fill="#8e44ad" radius={[0,4,4,0]} />
                  </BarChart>
                </ResponsiveContainer>
                <p style={{ fontSize: "11px", color: "#999", marginTop: "4px" }}>
                  Red = increases default risk | Green = decreases default risk
                </p>
              </div>

              <div style={{ marginTop: "12px", fontSize: "12px", color: "#999" }}>
                Default probability: {(result.p_default * 100).toFixed(1)}%
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}