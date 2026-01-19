# Agentic Insurance & Pension Assistant

This project implements a Streamlit-based AI assistant for insurance and pension queries. It uses Vertex AI generative models and BigQuery RAG for pension-related responses.

---

## File Overview

### 1. app.py
- Main Streamlit app.
- Handles:
  - User authentication
  - Session state
  - Chat input/output
  - Domain classification and response routing
- Calls functions from vertex_agent.py.

### 2. vertex_agent.py
- Core AI logic.
- Responsibilities:
  - Vertex AI model initialization (init_vertex_ai)
  - Loading user CSV data from GCS (load_user_data)
  - Classifying user queries (classify_domain)
  - Generating insurance responses (generate_response)
  - Generating pensions responses with RAG (generate_pension_response)
  - BigQuery integration for pension chunks
- Libraries used: vertexai, pandas, google-cloud-storage, google-cloud-bigquery, streamlit

### 3. bq_init.py (Offline / One-time script)
- Builds BigQuery embedding table from FCA PDF documents.
- Not part of the runtime app.
- Steps:
  1. Load PDFs from GCS.
  2. Chunk the text.
  3. Generate embeddings in batches using Vertex AI.
  4. Create BigQuery dataset and table (if not exists).
  5. Upload embeddings in batch to BigQuery.
- Libraries used: pdfplumber, vertexai, google-cloud-storage, google-cloud-bigquery
- Note: Run manually when adding new documents; the runtime app (app.py) reads from BigQuery.

---

## Notes
- bq_init.py is offline only; do not include it in app runtime.
- Pension RAG queries (retrieve_pension_chunks) rely on the BigQuery table populated by bq_init.py.


Just for fun : App is currently hosted here -> https://my-chatbot-537553334366.europe-central2.run.app/
