"""Shared retrieval and prompt helpers for the terminal and web chat clients."""

import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
import decimal
import re
import pandas as pd

DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = 10
SYSTEM_PROMPT = (
    "You are an expert executive data analyst and academic consultant specializing in institutional feedback analysis. "
    "Your objective is to provide polished, highly structured, and actionable analytical reports based exclusively on retrieved context."
)


def build_prompt(question: str, retrieved_chunks: list[str]) -> str:
    """Combines user question and context chunks into a professional, structured prompt."""
    context = "\n\n".join(retrieved_chunks)
    return f"""Context Data:
{context}

User Query: {question}

Role: AI Procurement & Document Auditor

Task: Analyze ALL uploaded files (contracts, purchase orders, supply orders) and generate a fully populated, highly detailed Master Summary Report. Ensure NO values are left as "Not specified" if they exist in the text.

Output Format Guidelines:

### 1. Executive Master Table
Present a unified comparison table for all uploaded documents using the exact parameters below:

| File / Document Name | Document Ref / PO # | Buyer / Client Name | Seller / Vendor Name | Product / Service Scope | Total Value (INR) | Contract / Delivery Period |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [Exact File Name] | [Ref Number] | [Full Buyer Name & Org] | [Full Vendor Name] | [Detailed Item Scope] | [Full Amount with Taxes] | [Start Date to End Date] |

---

### 2. Comprehensive File-by-File Breakdown
For EACH uploaded file, list all extracted information in structured sections:

#### File: [Exact File Name]
- Document Metadata:
  - Document Ref / PO Number:
  - Document Date:
  - Procurement Mode / Channel: (e.g., GeM Direct Purchase, L1 Comparison, Corporate PO)
- Buyer / Client Details:
  - Organization Name:
  - Department / Zone:
  - Designation & Contact Person:
  - Email & Phone Number:
  - Full Address:
  - GSTIN & PAN (if available):
- Seller / Vendor Details:
  - Company Name:
  - Contact Person / Email / Phone:
  - Registered Address:
  - GSTIN & PAN:
  - MSME Registration / Category:
- Financial & Scope Breakdown:
  - Line Items / Service Details (Item description, quantity, unit rate):
  - Base Value (Excl. Tax):
  - Tax Rate & Tax Value:
  - Total Order Value (Incl. Tax):
  - Payment Terms & Billing Cycle:
- Operational & SLA Terms:
  - Delivery / Service Timeline:
  - SLA / Penalty / Liquidated Damages (LD) Clauses:
  - Warranty / Maintenance Terms:

---

### 3. Summary Statistics
- Total Portfolio Value (INR): [Sum of all orders combined]
- Primary Vendors Identified: [List of vendors]
- Active Execution Window: [Earliest start date to latest end date]

Respond to: {question}
"""


def answer_question(question: str, collection: chromadb.Collection, client: OpenAI = None):
    """Retrieve relevant chunks and generate an answer for a terminal client."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    retrieved_chunks = results["documents"][0]
    sources = sorted({meta["source"] for meta in results["metadatas"][0]})
    if not retrieved_chunks:
        return "I couldn't find anything relevant in the documents.", sources

    response = (client or OpenAI()).chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, retrieved_chunks)},
        ],
    )
    
    return response.choices[0].message.content,"" #sources

def _pick_value_and_label(columns: list[str], rows: list[list], exclude: set[int] = frozenset()):
    """Pick the best (value_idx, label_idx) pair from the result set, optionally
    excluding certain columns from being chosen as the value (used to find a
    *second* measure distinct from the first chart's)."""

    def is_numeric(v):
        return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)

    def looks_like_id(name: str) -> bool:
        n = name.lower()
        return n.endswith("id") or n in ("id",)

    sample = rows[0]

    value_idx = None
    for i, val in enumerate(sample):
        if i in exclude:
            continue
        if is_numeric(val) and not looks_like_id(columns[i]):
            value_idx = i
            break
    if value_idx is None:
        for i, val in enumerate(sample):
            if i in exclude:
                continue
            if is_numeric(val):
                value_idx = i
                break
    if value_idx is None:
        return None, None

    sample_rows = rows[:50]

    def distinct_count(i):
        return len({str(r[i]) for r in sample_rows})

    candidates = [i for i in range(len(columns)) if i != value_idx]
    label_idx = candidates[0]
    best_score = (distinct_count(label_idx), 0 if not looks_like_id(columns[label_idx]) else -1)
    for i in candidates[1:]:
        score = (distinct_count(i), 0 if not looks_like_id(columns[i]) else -1)
        if score > best_score:
            best_score = score
            label_idx = i

    return value_idx, label_idx


def suggest_charts(df: pd.DataFrame) -> list:
    """
    Builds multiple chart options (different types and secondary measures) 
    for side-by-side dashboard rendering.
    
    - Handles pyodbc decimal.Decimal / NUMERIC types as numeric.
    - Excludes ID-like columns and Contact/Phone numbers from chart values.
    - Selects label axis based on the categorical column with highest variance/uniqueness.
    """
    if df.empty:
        return []

    df = df.copy()

    # 1. Standardize SQL Server DECIMAL/NUMERIC (pyodbc decimal.Decimal) to float
    for col in df.columns:
        if df[col].dtype == object:
            # Check if non-null elements are instances of decimal.Decimal
            sample = df[col].dropna()
            if not sample.empty and isinstance(sample.iloc[0], decimal.Decimal):
                df[col] = df[col].astype(float)

    # 2. Identify and filter out ID-like and Contact Number columns
    id_pattern = re.compile(r'(^id|_id|id$|code|uuid|guid)', re.IGNORECASE)
    contact_pattern = re.compile(r'(phone|contact|mobile|cell|fax|telephone)', re.IGNORECASE)

    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out numeric columns that represent IDs or Contact Numbers
    numeric_measures = []
    for col in all_numeric_cols:
        col_str = str(col)
        # Exclude matching patterns
        if id_pattern.search(col_str) or contact_pattern.search(col_str):
            continue
        # Exclude numeric columns that look like phone numbers (e.g., 10 digits) or continuous sequence IDs
        if df[col].nunique() == len(df) and df[col].dtype == 'int64':
            continue
            
        numeric_measures.append(col)

    # 3. Identify categorical / label columns (excluding IDs & Contacts)
    candidate_labels = []
    for col in df.columns:
        col_str = str(col)
        if id_pattern.search(col_str) or contact_pattern.search(col_str):
            continue
        # Allow object/string, category, or datetime columns
        if df[col].dtype in ['object', 'category', 'datetime64[ns]']:
            candidate_labels.append(col)

    # If no categorical columns exist, fallback to non-measure numeric columns or index
    if not candidate_labels:
        candidate_labels = [col for col in df.columns if col not in numeric_measures]
    
    if not candidate_labels:
        df['Index'] = df.index.astype(str)
        candidate_labels = ['Index']

    # 4. Pick label axis by whichever candidate column varies most across rows
    # (highest number of unique values, capped below row count to avoid unique key noise)
    label_axis = max(candidate_labels, key=lambda c: df[c].nunique())

    # If no valid measures are found, return empty
    if not numeric_measures:
        return []

    primary_measure = numeric_measures[0]
    secondary_measure = numeric_measures[1] if len(numeric_measures) > 1 else None

    # 5. Build multiple chart configurations
    chart_options = []

    # View 1: Primary Measure - Bar Chart
    chart_options.append({
        "title": f"{primary_measure} by {label_axis} (Bar)",
        "chart_type": "bar",
        "x_axis": label_axis,
        "y_axis": [primary_measure],
        "data": df[[label_axis, primary_measure]].to_dict(orient="records")
    })

    # View 2: Primary Measure - Line / Trend Chart
    chart_options.append({
        "title": f"{primary_measure} Trend across {label_axis} (Line)",
        "chart_type": "line",
        "x_axis": label_axis,
        "y_axis": [primary_measure],
        "data": df[[label_axis, primary_measure]].to_dict(orient="records")
    })

    # View 3: Multi-Measure Comparison (if secondary measure exists)
    if secondary_measure:
        chart_options.append({
            "title": f"Comparison: {primary_measure} vs {secondary_measure}",
            "chart_type": "grouped_bar",
            "x_axis": label_axis,
            "y_axis": [primary_measure, secondary_measure],
            "data": df[[label_axis, primary_measure, secondary_measure]].to_dict(orient="records")
        })
        
        # View 4: Scatter Plot for correlation between two measures
        chart_options.append({
            "title": f"{primary_measure} vs {secondary_measure} Correlation",
            "chart_type": "scatter",
            "x_axis": primary_measure,
            "y_axis": secondary_measure,
            "tooltip_label": label_axis,
            "data": df[[label_axis, primary_measure, secondary_measure]].to_dict(orient="records")
        })
    else:
        # Alternative View 3 for single measure: Donut / Pie Chart (if unique categories <= 10)
        if df[label_axis].nunique() <= 10:
            chart_options.append({
                "title": f"Distribution of {primary_measure} by {label_axis}",
                "chart_type": "pie",
                "label": label_axis,
                "value": primary_measure,
                "data": df[[label_axis, primary_measure]].to_dict(orient="records")
            })

    return chart_options
    """Build a small set of chart options (different types, and a second
    measure if one exists) instead of a single guess, so the dashboard can
    show several views of the same result side by side.

    - Treats SQL Server DECIMAL/NUMERIC columns (returned by pyodbc as
      decimal.Decimal, not float) as numeric.
    - Avoids charting ID-like columns (StudentId, MarksId, ...) as values.
    - Picks the label axis by whichever column varies most across rows.
    """
    if not rows or len(columns) < 2:
        return []

    def is_numeric(v):
        return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)

    def value_for(v):
        return float(v) if isinstance(v, Decimal) else (v if is_numeric(v) else 0)

    def label_for(v):
        return str(v) if v is not None else "—"

    value_idx, label_idx = _pick_value_and_label(columns, rows)
    if value_idx is None:
        return []

    sample_rows = rows[:50]
    labels = [label_for(r[label_idx]) for r in sample_rows]
    values = [value_for(r[value_idx]) for r in sample_rows]
    measure_name = columns[value_idx]

    charts = [
        {"type": "bar", "title": f"{measure_name} by {columns[label_idx]}",
         "labels": labels, "label": measure_name, "values": values},
        {"type": "line", "title": f"{measure_name} trend",
         "labels": labels, "label": measure_name, "values": values},
    ]

    # A pie chart only reads well with a modest number of categories, and
    # only makes sense for genuinely additive quantities (not e.g. an
    # average or a percentage), so keep it simple: offer it whenever the
    # category count is small.
    distinct_labels = len(set(labels))
    if 2 <= distinct_labels <= 8:
        charts.append({
            "type": "pie", "title": f"{measure_name} share by {columns[label_idx]}",
            "labels": labels, "label": measure_name, "values": values,
        })

    # A second chart for another numeric measure, if the result has one
    # (e.g. MarksObtained alongside FullMarks).
    value_idx2, label_idx2 = _pick_value_and_label(columns, rows, exclude={value_idx})
    if value_idx2 is not None:
        labels2 = [label_for(r[label_idx2]) for r in sample_rows]
        values2 = [value_for(r[value_idx2]) for r in sample_rows]
        measure_name2 = columns[value_idx2]
        charts.append({
            "type": "bar", "title": f"{measure_name2} by {columns[label_idx2]}",
            "labels": labels2, "label": measure_name2, "values": values2,
        })

    return charts


def main():
    """Run a simple terminal chat session."""
    load_dotenv()
    if not os.path.isdir(DB_DIR):
        raise RuntimeError("No index found. Run ingest.py first.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print("Ask questions about your documents. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer, sources = answer_question(question, collection)
        print(f"\nAssistant:\n{answer}")
        #if sources:
           # print(f"\nSources: {', '.join(sources)}")


if __name__ == "__main__":
    main()