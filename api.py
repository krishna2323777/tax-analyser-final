# api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pdfplumber
from extract import extract_financial_data_with_ai
import openai
import json
from typing import Dict, Any
import re
import io

app = FastAPI()

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Corporate Tax Analyzer API is running"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        content = await file.read()
        full_text, tables_text = "", ""

        if file.filename.lower().endswith(".pdf"):
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

                        tables = page.extract_tables()
                        for t in tables:
                            if len(t) > 1:
                                headers = t[0]
                                rows = t[1:]
                                try:
                                    df = pd.DataFrame(rows, columns=headers)
                                except Exception as e:
                                    print(f"Error creating DataFrame from table: {e}")
                                    df = pd.DataFrame(t)  # fallback to raw
                                tables_text += df.to_string(index=False) + "\n"
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

        elif file.filename.lower().endswith(".csv"):
            try:
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
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or CSV")

        if not full_text and not tables_text:
            raise HTTPException(status_code=400, detail="No readable content found in the file")

        result = extract_financial_data_with_ai(full_text, tables_text)
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")