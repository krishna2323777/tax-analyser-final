# api.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pdfplumber
from extract import extract_financial_data_with_ai
import openai
import json
from typing import Dict, Any
import re

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    full_text, tables_text = "", ""

    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                tables = page.extract_tables()
                for t in tables:
                    df = pd.DataFrame(t)
                    tables_text += df.to_string(index=False) + "\n"
    elif file.filename.endswith(".csv"):
        import io
        df = pd.read_csv(io.BytesIO(content), header=None)
        key_rows = []
        for idx, row in df.iterrows():
            row_str = " ".join([str(x) for x in row if pd.notnull(x)])
            if any(
                kw in row_str.lower()
                for kw in [
                    "revenue", "sales", "expenses", "depreciation", "deductions",
                    "net income", "profit", "loss", "tax"
                ]
            ):
                key_rows.append(row_str)
        full_text = "\n".join(key_rows)
        tables_text = df.to_string(index=False)
    else:
        return {"error": "Unsupported file type"}

    result = extract_financial_data_with_ai(full_text, tables_text)
    print("AI RAW RESPONSE:", result)
    return result