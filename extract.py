import os
import re
import json
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
import openai
from typing import Dict, Any

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_key

# ─── Utility Functions ─────────────────────────
def clean_numeric_value(value: str) -> float:
    if not value or value == "" or value == "0":
        return 0.0
    cleaned = re.sub(r'[\u20ac$,%\s]', '', str(value))
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    match = re.search(r'-?\d+\.?\d*', cleaned)
    return float(match.group()) if match else 0.0

def calculate_netherlands_tax(taxable_income: float) -> float:
    if taxable_income <= 0:
        return 0.0
    if taxable_income <= 200000:
        return taxable_income * 0.19
    return 200000 * 0.19 + (taxable_income - 200000) * 0.258

def extract_financial_data_with_ai(text: str, tables_data: str) -> Dict[str, Any]:
    PROMPT = """
You are a Corporate Tax Analyzer AI assistant.
Your job is to extract clean, structured financial data from Dutch corporate tax documents. These documents may include trial balances, profit-loss statements, or invoices in either PDF or CSV form.

From the given document text, extract and return the following fields in valid JSON format, for each quarter (Q1, Q2, Q3, Q4) and overall:
{
  "quarters": {
    "Q1": {"revenue": "", "expenditures": "", "depreciation": "", "deductions": "", "net_taxable_income": "", "final_tax_owed": ""},
    "Q2": {...},
    "Q3": {...}
  },
  "overall": {
    "company_name": "",
    "country": "",
    "revenue": "",
    "expenditures": "",
    "depreciation": "",
    "deductions": "",
    "net_taxable_income": "",
    "final_tax_owed": ""
  }
}

Instructions:
- 'net_taxable_income' = revenue - expenditures - depreciation - deductions
- Use Netherlands tax rules:
    - ≤ 200k €: 19%
    - > 200k €: 25.8%
- Return only valid JSON with double quotes.
"""
    combined = f"DOCUMENT TEXT:\n{text}\n\nTABLE DATA:\n{tables_data}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PROMPT.strip()},
                {"role": "user", "content": combined[:15000]}
            ],
            temperature=0,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()
        if result.startswith("```"):
            result = re.sub(r'```json\n?|```', '', result).strip()
        data = json.loads(result)
        return data
    except Exception as e:
        return {"error": str(e)}

def format_currency(amount: str) -> str:
    try:
        num = float(amount)
        return f"-\u20ac{abs(num):,.0f}" if num < 0 else f"\u20ac{num:,.0f}"
    except:
        return "\u20ac0"

def create_results_table(financial_data: Dict[str, Any]) -> pd.DataFrame:
    rows = [[k.replace("_", " ").title(), format_currency(financial_data[k])] for k in [
        "total_revenue", "total_expenses", "depreciation", "deductions",
        "net_taxable_income", "final_tax_owed"]]
    rows.insert(0, ["Company Name", financial_data.get("company_name", "")])
    rows.insert(1, ["Country", financial_data.get("country", "")])
    return pd.DataFrame(rows, columns=["Field", "Value"])      





