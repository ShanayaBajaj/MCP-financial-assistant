# %%
import os
import streamlit as st
import sqlite3
import requests
import yfinance as yf
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set environment variables 
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
# Company map

company_map = {
    "reliance": {"name": "Reliance Industries", "ticker": "RELIANCE.NS"},
    "tcs": {"name": "TCS", "ticker": "TCS.NS"},
    "hdfc": {"name": "HDFC Bank", "ticker": "HDFCBANK.NS"},
    "infosys": {"name": "Infosys", "ticker": "INFY.NS"},
    "icici": {"name": "ICICI Bank", "ticker": "ICICIBANK.NS"},
    "hul": {"name": "HUL", "ticker": "HINDUNILVR.NS"},
    "sbi": {"name": "SBI", "ticker": "SBIN.NS"},
    "bharti": {"name": "Bharti Airtel", "ticker": "BHARTIARTL.NS"},
    "bajaj": {"name": "Bajaj Finance", "ticker": "BAJFINANCE.NS"},
    "asian": {"name": "Asian Paints", "ticker": "ASIANPAINT.NS"}
}



# Create the database file
conn = sqlite3.connect("company_data.db")
cursor = conn.cursor()

# Create the table (only if it doesn't already exist)
cursor.execute("""
CREATE TABLE IF NOT EXISTS company_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    sector TEXT,
    market_cap TEXT,
    latest_revenue TEXT,
    notes TEXT
)
""")

# Insert data into the table
companies = [
    ("Reliance Industries", "Conglomerate", "19T INR", "9T INR", "Focus on renewable energy"),
    ("TCS", "IT Services", "12T INR", "2T INR", "High margins, digital leadership"),
    ("HDFC Bank", "Banking", "11T INR", "1.5T INR", "Stable asset quality and retail strength"),
    ("Infosys", "IT Services", "7T INR", "1.3T INR", "Global digital capabilities"),
    ("ICICI Bank", "Banking", "7T INR", "1.2T INR", "Improved profitability via digital"),
    ("HUL", "FMCG", "6T INR", "0.6T INR", "Strong brand portfolio"),
    ("SBI", "Banking", "8T INR", "1.8T INR", "Largest PSB with improving asset quality"),
    ("Bharti Airtel", "Telecom", "5T INR", "1.5T INR", "Pan-India 4G coverage"),
    ("Bajaj Finance", "NBFC", "4T INR", "0.7T INR", "Diversified lending portfolio"),
    ("Asian Paints", "Paints", "3T INR", "0.4T INR", "Growth driven by rural demand")
]

cursor.executemany("""
INSERT INTO company_info (name, sector, market_cap, latest_revenue, notes)
VALUES (?, ?, ?, ?, ?)
""", companies)

conn.commit()
conn.close()

print(" Database created and data inserted successfully!")

# Context manager
class MCPContextManager:
    def __init__(self):
        self.context = []

    def add(self, tool_name: str, tool_input: str, tool_output: str):
        """Add a record of tool usage to the context."""
        self.context.append({
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output
        })

    def reset(self):
        """Clear the context (e.g., at the start of a new query)."""
        self.context = []

    def show(self) -> str:
        """Format the context as readable text for LLM input."""
        return "\n\n".join(
            f"Tool: {item['tool']}\nInput: {item['input']}\nOutput: {item['output']}"
            for item in self.context
        )

    def to_dict(self):
        """Return context as list of dictionaries (for logging/debugging)."""
        return self.context


ctx = MCPContextManager()

def clean_context_outputs():
    """Extracts only the outputs from context, excluding tool names and inputs."""
    return "\n\n".join(item["output"] for item in ctx.to_dict())


# Functions from your backend
def get_stock_price(symbol):
    try:
        price = round(yf.Ticker(symbol).history(period="1d")['Close'][0], 2)
        result = f"{price} INR"
    except Exception as e:
        result = f"Error: {str(e)}"
    ctx.add("get_stock_price", symbol, result)
    return result

def get_company_info(name):
    try:
        conn = sqlite3.connect("company_data.db"); cur = conn.cursor()
        cur.execute("SELECT * FROM company_info WHERE name LIKE ?", (f"%{name}%",))
        row = cur.fetchone(); conn.close()
        result = "\n".join(f"{k}: {v}" for k, v in dict(zip(["id","name","sector","market_cap","latest_revenue","notes"], row)).items()) if row else "Company not found."
    except Exception as e:
        result = f"DB error: {e}"
    ctx.add("get_company_info", name, result)
    return result

def web_search(query):
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params={
            "key": os.getenv("GOOGLE_API_KEY"), "cx": os.getenv("GOOGLE_CSE_ID"), "q": query})
        items = r.json().get("items", [])
        out = "\n".join([f"- {i['title']}: {i['snippet']}" for i in items[:3]]) or "No results found."
    except Exception as e:
        out = f"Web search error: {e}"
    ctx.add("web_search", query, out)
    return out

def chunk_pdf(path, size=100):
    try:
        text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
        return [text[i:i+size] for i in range(0, len(text), size)] if text.strip() else []
    except Exception as e:
        print("PDF error:", e); return []

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pdf_chunks = chunk_pdf("Top10_Companies_Report.pdf")
index = None
if pdf_chunks:
    emb = embedding_model.encode(pdf_chunks)
    index = faiss.IndexFlatL2(emb.shape[1]); index.add(np.array(emb))

def retrieve_relevant_chunks(q, k=2):
    try:
        if index:
            _, I = index.search(np.array(embedding_model.encode([q])), k)
            return [pdf_chunks[i] for i in I[0]]
        return ["No index available."]
    except Exception as e:
        return [f"Error: {e}"]

#  Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

#  Streamlit UI
st.set_page_config(page_title="MCP Financial Assistant",)
st.title(" Financial Assistant")
st.write("Ask about a company to get stock price, DB info, report summary, and latest news.")

user_input = st.text_input("Enter a top 10 company under NSE (reliance, hdfc bank, tcs, airtel, infosys, icici bank, hul, sbi, bajaj finance and asian paints):").strip().lower()

if st.button("Ask assistant") and user_input:
    ctx.context = []  # Reset context each run
    company = company_map.get(user_input)

    if company:
        company_name = company['name']
        ticker = company['ticker']
        st.info(f"Running agent for: **{company_name} ({ticker})**")

        question = f"What does the report say about {company_name}?"
        retrieved_chunks = retrieve_relevant_chunks(question)
        rag_context = "\n".join(retrieved_chunks)
        ctx.add("RAG_retrieve", question, rag_context)

        price = get_stock_price(ticker)
        db_info = get_company_info(company_name)
        web_result = web_search(f"{company_name} latest news")

        # Build Gemini prompt
        prompt = f"""You are an expert financial assistant.Based on the following information, write a clear and concise summary about {company_name}:

            {clean_context_outputs()}
                """
        response = model.generate_content(prompt)


        # Show results
        st.subheader("Report Summary")
        st.text(rag_context)

        st.subheader("Stock Price")
        st.text(price)

        st.subheader("Database Info")
        st.text(db_info)

        st.subheader("Web Search")
        st.text(web_result)

        st.subheader("AI Summary")
        st.write(response.text)

    else:
        st.error("Unknown company. Please try again.")


# %%


