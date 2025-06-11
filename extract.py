import os
import re
import json
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
import openai
from typing import Dict, Any
from fastapi import HTTPException

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key loaded: {'Yes' if openai_key else 'No'}")
print(f"OpenAI API Key length: {len(openai_key) if openai_key else 0}")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
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
    print(f"Starting AI extraction with text length: {len(text)} and tables length: {len(tables_data)}")
    PROMPT = """
You are a Corporate Tax Analyzer AI assistant specialized in extracting financial data from Dutch corporate documents.
Your job is to extract clean, structured financial data from Dutch corporate tax documents. These documents may include trial balances, profit-loss statements, or invoices in either PDF or CSV form.

From the given document text, extract and return only the following fields in valid JSON format:
{
  "company_name": "",
  "country": "",
  "total_revenue": "",
  "total_expenses": "",
  "depreciation": "",
  "deductions": "",
  "net_taxable_income": "",
  "final_tax_owed": "",
  "quarters": {
    "Q1": {"revenue": "", "expenses": "", "depreciation": "", "deductions": "", "net_taxable_income": "", "final_tax_owed": ""},
    "Q2": {...},
    "Q3": {...},
    "Q4": {...}
  }
}

Instructions:
1. ALWAYS return quarterly data (Q1, Q2, Q3, Q4) even if the document doesn't explicitly show quarters:
   - If annual data is given, divide it evenly into quarters
   - If monthly data is given, combine into quarters (Jan-Mar = Q1, Apr-Jun = Q2, Jul-Sep = Q3, Oct-Dec = Q4)
   - If only one period is given, use that data for all quarters
2. For the overall section, include:
   - company_name (look in headers, footers, or document metadata)
   - country (assume Netherlands if not specified)
   - All financial fields (use annual totals if available)

- If overall data is not present, set each field to "" or "Not found".
- Do not invent data. Only extract what is present.
- net_taxable_income = total_revenue - total_expenses - depreciation - deductions
- Use Netherlands tax rules for final_tax_owed as stated: 
    Netherlands tax rules:
    - ≤ 200k €: 19%
    - > 200k €: 25.8%

- Return only valid JSON with double quotes.
"""
    combined = f"DOCUMENT TEXT:\n{text}\n\nTABLE DATA:\n{tables_data}"
    try:
        print("Making OpenAI API call...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": PROMPT.strip()},
                    {"role": "user", "content": combined[:15000]}
                ],
                temperature=0,
                max_tokens=2000
            )
            print("OpenAI API call successful")
            result = response.choices[0].message.content.strip()
            if result.startswith("```"):
                result = re.sub(r'```json\n?|```', '', result).strip()
            print(f"Raw AI response: {result[:200]}...")  # Print first 200 chars of response
            data = json.loads(result)
            data = fill_quarters_from_overall(data)
            return data
        except openai.error.AuthenticationError as e:
            print(f"OpenAI Authentication Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API Authentication Error: {str(e)}")
        except openai.error.APIError as e:
            print(f"OpenAI API Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
        except openai.error.RateLimitError as e:
            print(f"OpenAI Rate Limit Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI Rate Limit Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected OpenAI Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected OpenAI Error: {str(e)}")
    except Exception as e:
        print(f"Error in AI extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in AI extraction: {str(e)}")

def fill_quarters_from_overall(data: dict) -> dict:
    # Only fill if quarters are missing or empty
    quarters = data.get("quarters", {})
    # Check if all quarters are empty or missing
    if not any(q and any(v not in ("", "0", 0) for v in q.values()) for q in quarters.values()):
        try:
            revenue = float(data.get("total_revenue", 0) or 0)
            expenses = float(data.get("total_expenses", 0) or 0)
            depreciation = float(data.get("depreciation", 0) or 0)
            deductions = float(data.get("deductions", 0) or 0)
            net_income = float(data.get("net_taxable_income", 0) or 0)
            tax = float(data.get("final_tax_owed", 0) or 0)
        except Exception:
            revenue = expenses = depreciation = deductions = net_income = tax = 0
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            quarters[q] = {
                "revenue": str(int(revenue // 4)) if revenue else "",
                "expenses": str(int(expenses // 4)) if expenses else "",
                "depreciation": str(int(depreciation // 4)) if depreciation else "",
                "deductions": str(int(deductions // 4)) if deductions else "",
                "net_taxable_income": str(int(net_income // 4)) if net_income else "",
                "final_tax_owed": str(int(tax // 4)) if tax else "",
            }
        data["quarters"] = quarters
    return data

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





