import os
import re
import json
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=openai_key)

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
    if not text and not tables_data:
        return {"error": "No text or table data provided"}

    PROMPT = """
You are a Corporate Tax Analyzer AI assistant specialized in extracting financial data from Dutch corporate documents.
Your task is to analyze the provided document and extract financial information, even if the data is not explicitly labeled as quarterly.

IMPORTANT INSTRUCTIONS:
1. ALWAYS return quarterly data (Q1, Q2, Q3) even if the document doesn't explicitly show quarters:
   - If annual data is given, divide it evenly into quarters
   - If monthly data is given, combine into quarters (Jan-Mar = Q1, Apr-Jun = Q2, Jul-Sep = Q3, Oct-Dec = Q4)
   - If only one period is given, use that data for all quarters
   - Never leave quarterly data empty

2. For each quarter, extract:
   - revenue
   - expenditures
   - depreciation
   - deductions
   - net_taxable_income (calculate as: revenue - expenditures - depreciation - deductions)
   - final_tax_owed (calculate using Netherlands tax rules)

3. For the overall section, include:
   - company_name (look in headers, footers, or document metadata)
   - country (assume Netherlands if not specified)
   - All financial fields (use annual totals if available)

4. Netherlands tax rules:
   - ≤ 200k €: 19%
   - > 200k €: 25.8%

5. IMPORTANT: Never return empty values:
   - Use "0" for missing numeric fields
   - Use "Not found" for missing text fields
   - Always return a complete structure

Return the data in this exact JSON format:
{
  "quarters": {
    "Q1": {"revenue": "", "expenditures": "", "depreciation": "", "deductions": "", "net_taxable_income": "", "final_tax_owed": ""},
    "Q2": {...},
    "Q3": {...},
    "Q4": {...}
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

Remember: Always return valid JSON with double quotes and never leave any fields empty.
"""
    combined = f"DOCUMENT TEXT:\n{text}\n\nTABLE DATA:\n{tables_data}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PROMPT.strip()},
                {"role": "user", "content": combined[:15000]}
            ],
            temperature=0,
            max_tokens=2000
        )
        result = response.choices[0].message.content.strip()
        
        # Clean up the response
        if result.startswith("```"):
            result = re.sub(r'```json\n?|```', '', result).strip()
        
        # Safety check: If result is empty or not valid JSON, return default structure
        if not result or not result.strip():
            return {
                "quarters": {
                    "Q1": {"revenue": "0", "expenditures": "0", "depreciation": "0", "deductions": "0", "net_taxable_income": "0", "final_tax_owed": "0"},
                    "Q2": {"revenue": "0", "expenditures": "0", "depreciation": "0", "deductions": "0", "net_taxable_income": "0", "final_tax_owed": "0"},
                    "Q3": {"revenue": "0", "expenditures": "0", "depreciation": "0", "deductions": "0", "net_taxable_income": "0", "final_tax_owed": "0"}
                },
                "overall": {
                    "company_name": "Not found",
                    "country": "Not found",
                    "revenue": "0",
                    "expenditures": "0",
                    "depreciation": "0",
                    "deductions": "0",
                    "net_taxable_income": "0",
                    "final_tax_owed": "0"
                }
            }
        
        try:
            data = json.loads(result)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {result}")
            return {
                "quarters": {
                    "Q1": {"revenue": "0", "expenditures": "0", "depreciation": "0", "deductions": "0", "net_taxable_income": "0", "final_tax_owed": "0"},
                    "Q2": {"revenue": "0", "expenditures": "0", "depreciation": "0", "deductions": "0", "net_taxable_income": "0", "final_tax_owed": "0"},
                    "Q3": {"revenue": "0", "expenditures": "0", "depreciation": "0", "deductions": "0", "net_taxable_income": "0", "final_tax_owed": "0"}
                },
                "overall": {
                    "company_name": "Not found",
                    "country": "Not found",
                    "revenue": "0",
                    "expenditures": "0",
                    "depreciation": "0",
                    "deductions": "0",
                    "net_taxable_income": "0",
                    "final_tax_owed": "0"
                }
            }
            
    except Exception as e:
        print(f"API call error: {str(e)}")
        return {"error": f"Failed to process document: {str(e)}"}

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
    rows.insert(0, ["Company Name", financial_data.get("company_name", "Not found")])
    rows.insert(1, ["Country", financial_data.get("country", "Not found")])
    return pd.DataFrame(rows, columns=["Field", "Value"])      





