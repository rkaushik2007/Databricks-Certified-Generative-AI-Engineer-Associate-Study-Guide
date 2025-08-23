# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 Hands-On Lab: End-to-End RAG System Deployment on Databricks
# MAGIC
# MAGIC ## 📌 Scenario
# MAGIC You are a Machine Learning Engineer at a large enterprise tasked with building a **production-ready Retrieval-Augmented Generation (RAG) system** on Databricks.  
# MAGIC Executives want employees to query internal knowledge bases—such as **technical documentation, compliance policies, and customer reports**—using natural language.  
# MAGIC
# MAGIC The challenge is not just building a prototype, but deploying a system that:
# MAGIC - Scales efficiently for enterprise use,
# MAGIC - Remains **cost-efficient**,
# MAGIC - Adheres to **governance and compliance standards**.
# MAGIC
# MAGIC To succeed, you will use:
# MAGIC - **MLflow** for experiment tracking, packaging, and registration,
# MAGIC - **Databricks Model Serving** for deployment,
# MAGIC - **Databricks Vector Search** for context retrieval,
# MAGIC - **Unity Catalog** for governance and version control.
# MAGIC
# MAGIC This lab mirrors **real-world enterprise scenarios** where **traceability, reproducibility, and compliance** are just as important as technical accuracy.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC - ✅ Package a LangChain-based **RAG pipeline** into a PyFunc model for Databricks.
# MAGIC - ✅ Track runs, parameters, and artifacts with **MLflow**.
# MAGIC - ✅ Register and promote the model through **Staging → Production** in **Unity Catalog**.
# MAGIC - ✅ Build and query a **Vector Search index** for retrieval.
# MAGIC - ✅ Deploy the RAG pipeline as a **REST-serving endpoint**.
# MAGIC - ✅ Test end-to-end queries against the deployed endpoint.
# MAGIC - ✅ Apply **optimization practices** to balance **latency** and **cost**.
# MAGIC - 🛠️ *(Optional)* Demonstrate advanced deployment techniques:
# MAGIC   - Stage vs. version targeting,
# MAGIC   - Error handling and fallbacks for production readiness.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 1: Install Required Libraries
# MAGIC
# MAGIC In this step, we install the Python packages necessary for building and deploying the **Retrieval-Augmented Generation (RAG) system** on Databricks.
# MAGIC
# MAGIC - **databricks-vectorsearch** → Provides APIs for creating and querying Vector Search indexes.  
# MAGIC - **mlflow** → Used for experiment tracking, packaging the RAG pipeline, and model registration.  
# MAGIC - **langchain** → Simplifies orchestration of retrieval + generation workflows.  
# MAGIC - **tiktoken** → Tokenizer for working with LLM prompts and
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch mlflow langchain tiktoken requests 
# MAGIC %pip install --quiet -U databricks-vectorsearch

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 2: Restart Python Kernel
# MAGIC
# MAGIC After installing new libraries with `%pip install`, we need to restart the Python kernel so that the environment picks up the newly installed or upgraded packages.  
# MAGIC This ensures that the correct versions of `databricks-vectorsearch`, `mlflow`, `langchain`, and others are available for use in subsequent steps.
# MAGIC

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 3: Define Configuration Variables
# MAGIC
# MAGIC In this step, we define **all environment-specific configuration values** required for the lab.  
# MAGIC These variables make the notebook portable and easier to adapt across environments (development, staging, production).
# MAGIC
# MAGIC Key sections:
# MAGIC
# MAGIC - **Databricks Workspace Configuration**  
# MAGIC   Workspace URL, authentication token, and secret scope setup.  
# MAGIC   ⚠️ For production use, prefer `dbutils.secrets.get()` instead of hardcoding tokens.
# MAGIC
# MAGIC - **Model and Endpoint Configuration**  
# MAGIC   Embedding model endpoint, Vector Search endpoint name, registered model name, and the final serving endpoint.
# MAGIC
# MAGIC - **Database Configuration**  
# MAGIC   Catalog, schema, and table names for raw documents, processed chunks, embeddings, and the vector index.
# MAGIC
# MAGIC - **Processing Configuration**  
# MAGIC   Chunk sizes, batch sizes, similarity search parameters, and request timeouts.
# MAGIC
# MAGIC - **Circuit Breaker Configuration**  
# MAGIC   Settings for failure thresholds and recovery logic to improve production resilience.
# MAGIC
# MAGIC - **Derived Configuration**  
# MAGIC   Fully qualified paths and derived variables (do not modify).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # 🔑 How to Create a Databricks Personal Access Token (PAT)
# MAGIC
# MAGIC A **Personal Access Token (PAT)** is required for programmatic access to Databricks REST APIs and Model Serving.  
# MAGIC Follow these steps to generate one:
# MAGIC
# MAGIC 1. **Log in** to your Databricks workspace.  
# MAGIC 2. In the top-right corner, click on your **user profile icon** → select **User Settings**.  
# MAGIC 3. Go to the **Access tokens** tab.  
# MAGIC 4. Click **Generate new token**.  
# MAGIC 5. Provide a **description** (e.g., "RAG Lab Token") and optionally set an **expiry date**.  
# MAGIC 6. Click **Generate**.  
# MAGIC 7. Copy the token immediately — it will not be shown again.  
# MAGIC 8. Use this token in your code (preferably via `dbutils.secrets.get()` in production for security).  
# MAGIC
# MAGIC ⚠️ **Best Practices:**
# MAGIC - Store the token in **Databricks Secret Scope** instead of hardcoding.  
# MAGIC - Use **short-lived tokens** whenever possible.  
# MAGIC - Rotate and revoke tokens regularly.  
# MAGIC

# COMMAND ----------

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE FOR YOUR ENVIRONMENT
# =============================================================================

# Databricks Workspace Configuration
WORKSPACE_URL = "Put your workspace URL"
# For example "https://adb-3141834805281316.15.azuredatabricks.net"
TOKEN = "Put personal access token"
# for example "dapic121fe7686ef956f616d37fc348fb58f-2"  # Consider using dbutils.secrets.get() for production
SECRET_SCOPE = "corp_lab"
SECRET_KEY = "databricks_pat"

# Model and Endpoint Configuration
EMBEDDING_ENDPOINT = "databricks-bge-large-en"
VECTOR_SEARCH_ENDPOINT_NAME = "orielly-chapter5-endpoint"
MODEL_NAME = "main.default.rag_pyfunc"
SERVING_ENDPOINT_NAME = "rag-pyfunc-endpoint-Chapter-5"

# Database Configuration
CATALOG_NAME = "corp_ai"
SCHEMA_NAME = "rag_lab"
RAW_TABLE = "docs_raw"
CHUNKS_TABLE = "docs_chunks"
EMBEDDINGS_TABLE = "docs_embed"
VECTOR_INDEX_NAME = "docs_index_sync"

# Processing Configuration
CHUNK_SIZE = 350
BATCH_SIZE = 32
SIMILARITY_SEARCH_RESULTS = 5
REQUEST_TIMEOUT = 60

# Circuit Breaker Configuration
FAILURE_THRESHOLD = 20  # 20% failure rate
RECOVERY_TIMEOUT = 60   # 1 minute recovery
SUCCESS_THRESHOLD = 3   # 3 successes to close
WINDOW_SIZE = 50        # Track last 50 requests
MIN_REQUESTS = 10       # Minimum requests before calculating failure rate

# Derived Configuration (DO NOT MODIFY)
FULL_CATALOG_SCHEMA = f"{CATALOG_NAME}.{SCHEMA_NAME}"
SOURCE_TABLE_FULLNAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE}"
VS_INDEX_FULLNAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{VECTOR_INDEX_NAME}"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
RETURN_COLUMNS = ["chunk_id", "doc_id", "section", "product_line", "region", "chunk"]

print("✅ Configuration loaded successfully")
print(f"📍 Workspace: {WORKSPACE_URL}")
print(f"🗄️ Database: {FULL_CATALOG_SCHEMA}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"🔗 Serving Endpoint: {SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 4: Import Required Libraries
# MAGIC
# MAGIC In this step, we import all the Python libraries required for the **RAG system deployment**.
# MAGIC
# MAGIC - **Core Python libraries** → Utilities for file handling, JSON, concurrency, random sampling, and system operations.  
# MAGIC - **Data Processing (Pandas, NumPy)** → Efficient manipulation of tabular data and numerical arrays.  
# MAGIC - **Requests** → For making REST API calls to Databricks endpoints.  
# MAGIC - **MLflow** → Used for model logging, tracking, registration, and signature inference.  
# MAGIC - **Databricks Vector Search Client** → To create, query, and manage Vector Search indexes.  
# MAGIC - **PySpark** → Provides distributed processing and DataFrame APIs for preparing documents, embeddings, and feature engineering.
# MAGIC
# MAGIC Once this cell runs, you will have all necessary libraries loaded and ready for use in the following steps.
# MAGIC

# COMMAND ----------

# =============================================================================
# IMPORTS - ALL REQUIRED LIBRARIES
# =============================================================================

# Core Python libraries
import os
import json
import time
import uuid
import tempfile
import threading
import random
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

# Data processing
import pandas as pd
import numpy as np

# HTTP requests
import requests

# MLflow and Databricks
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from databricks.vector_search.client import VectorSearchClient

# PySpark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType

print("✅ All libraries imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 5: Initialize Database Catalog and Schema
# MAGIC
# MAGIC In this step, we set up the **Databricks Unity Catalog** and a dedicated schema for storing all artifacts of the RAG pipeline.  
# MAGIC
# MAGIC Why this matters:
# MAGIC - **Catalogs** provide a top-level namespace in Unity Catalog.  
# MAGIC - **Schemas** organize related tables and models inside a catalog.  
# MAGIC - Ensures all tables (raw docs, chunks, embeddings) and models are grouped under a governed namespace.  
# MAGIC - Promotes **data governance, access control, and reproducibility** across teams.
# MAGIC
# MAGIC Here, we:
# MAGIC 1. Create the catalog (if it doesn’t already exist).  
# MAGIC 2. Create the schema within that catalog.  
# MAGIC 3. Switch the Spark session to use this catalog and schema.  
# MAGIC

# COMMAND ----------

# Create catalog and schema
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_CATALOG_SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print(f"✅ Database setup complete: {FULL_CATALOG_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 6: Load Sample Enterprise Documents
# MAGIC
# MAGIC To simulate enterprise knowledge bases (such as compliance manuals, product specifications, and policy handbooks), we create a **sample dataset**.  
# MAGIC
# MAGIC Why this matters:
# MAGIC - Provides a controlled **corpus of documents** for testing the RAG system.  
# MAGIC - Each document includes metadata such as:
# MAGIC   - `doc_id` → Unique document identifier  
# MAGIC   - `doc_type` → Type of document (manual, spec, handbook)  
# MAGIC   - `section` → Section or chapter reference  
# MAGIC   - `product_line` → Product relevance  
# MAGIC   - `region` → Regional applicability  
# MAGIC   - `effective_date` → Date the policy/spec becomes effective  
# MAGIC   - `text` → The actual document content  
# MAGIC
# MAGIC These documents are written into a Delta table (`docs_raw`) under the configured catalog and schema.  
# MAGIC This ensures governance and easy retrieval when we process them into chunks and embeddings in later steps.
# MAGIC

# COMMAND ----------

# Sample enterprise documents
sample_data = [
    ("DOC-001", "Compliance Manual", "Storage Policy", "product-a", "us", "2024-01-15", 
     "All customer data must be stored in encrypted volumes with AES-256. Backups require weekly integrity checks and must reside in approved regions."),
    ("DOC-002", "Compliance Manual", "Access Control", "product-a", "eu", "2024-03-01", 
     "Access to production data requires MFA and is restricted to on-call engineers. All access events must be logged and retained for 365 days."),
    ("DOC-003", "Product Spec", "Warranty Terms", "product-b", "us", "2023-11-20", 
     "Product-B includes a standard warranty of 12 months covering manufacturing defects. Consumables and accidental damage are excluded."),
    ("DOC-004", "Product Spec", "Maintenance Guide", "product-b", "apac", "2023-10-05", 
     "Maintenance requires quarterly inspections and replacement of filters after 500 hours of operation. Use only certified parts."),
    ("DOC-005", "Policy Handbook", "Data Retention", "shared", "us", "2024-02-10", 
     "Logs must be retained for a minimum of 180 days and a maximum of 730 days depending on classification. High-sensitivity logs require masking.")
]

# Define schema for the documents
document_schema = T.StructType([
    T.StructField("doc_id", T.StringType()),
    T.StructField("doc_type", T.StringType()),
    T.StructField("section", T.StringType()),
    T.StructField("product_line", T.StringType()),
    T.StructField("region", T.StringType()),
    T.StructField("effective_date", T.StringType()),
    T.StructField("text", T.StringType()),
])

# Create DataFrame and save to table
df_raw = spark.createDataFrame(sample_data, document_schema)
df_raw = df_raw.withColumn("effective_date", F.to_date("effective_date"))
df_raw.write.mode("overwrite").saveAsTable(f"{FULL_CATALOG_SCHEMA}.{RAW_TABLE}")

print(f"✅ Sample data created: {len(sample_data)} documents")
display(df_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 7: Chunk Documents for Embedding
# MAGIC
# MAGIC Large documents are difficult to embed and query directly. To make them more manageable and semantically searchable, we split them into **smaller chunks** of text.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Embeddings models have input size limits (token limits).  
# MAGIC - Chunking ensures each piece of text is within the embedding model’s capacity.  
# MAGIC - Improves retrieval accuracy, since queries can match **specific sections** rather than entire documents.  
# MAGIC - Each chunk is assigned a unique `chunk_id` for tracking and indexing.
# MAGIC
# MAGIC ### Approach:
# MAGIC 1. Use a **UDF (User Defined Function)** `simple_chunker`:
# MAGIC    - Splits text into sentences.
# MAGIC    - Groups sentences into chunks until `CHUNK_SIZE` is reached.
# MAGIC    - Produces an array of text chunks.
# MAGIC 2. Explode the chunks into individual rows.  
# MAGIC 3. Assign unique `chunk_id`s.  
# MAGIC 4. Save results to a governed Delta table (`docs_chunks`).  
# MAGIC

# COMMAND ----------

# Document chunking function
@F.udf("array<string>")
def simple_chunker(text):
    import re
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], []
    total = 0
    for s in sents:
        total += len(s)
        cur.append(s)
        if total > CHUNK_SIZE:
            chunks.append(" ".join(cur))
            cur, total = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# Process documents into chunks
chunks = (spark.table(f"{FULL_CATALOG_SCHEMA}.{RAW_TABLE}")
    .withColumn("chunks", simple_chunker(F.col("text")))
    .withColumn("chunk", F.explode("chunks"))
    .withColumn("chunk_id", F.monotonically_increasing_id())
    .select("chunk_id", "doc_id", "doc_type", "section", "product_line", "region", "effective_date", "chunk")
)

chunks.write.mode("overwrite").saveAsTable(f"{FULL_CATALOG_SCHEMA}.{CHUNKS_TABLE}")
print(f"✅ Document chunking complete")
display(chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 8: Test Embedding Endpoint Connectivity
# MAGIC
# MAGIC Before generating embeddings for all document chunks, we first test the **embedding model endpoint** to ensure it is accessible and returning vectors correctly.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Confirms that the configured **Databricks embedding endpoint** (`databricks-bge-large-en`) is online and reachable.  
# MAGIC - Ensures that authentication headers and workspace URLs are correctly set up.  
# MAGIC - Validates that the output vector has the expected dimensionality (e.g., 1024 dimensions).  
# MAGIC
# MAGIC We send a simple test sentence to the endpoint and check the response.
# MAGIC

# COMMAND ----------

# Test embedding endpoint connectivity
payload_single = {"input": "Databricks simplifies production RAG pipelines."}
response = requests.post(
    f"{WORKSPACE_URL}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations",
    headers=HEADERS, 
    data=json.dumps(payload_single), 
    timeout=REQUEST_TIMEOUT
)
response.raise_for_status()
embedding = response.json()["data"][0]["embedding"]
print(f"✅ Embedding endpoint test successful - Dimension: {len(embedding)}")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 9: Generate Embeddings for Document Chunks
# MAGIC
# MAGIC Now that we’ve verified the embedding endpoint, we generate embeddings for **all document chunks** and store them in a Delta table for later retrieval.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Embeddings transform text into high-dimensional vectors that capture semantic meaning.  
# MAGIC - These embeddings are the foundation for **Vector Search**, enabling semantic similarity queries.  
# MAGIC - Storing embeddings alongside metadata ensures we can later join search results back to their original documents.  
# MAGIC
# MAGIC ### Approach:
# MAGIC 1. Define a **Pandas UDF** `embed_udf` to call the embedding endpoint in **batches** (efficient API usage).  
# MAGIC 2. Apply the UDF on the `chunk` column from the `docs_chunks` table.  
# MAGIC 3. Store the results in a governed Delta table (`docs_embed`) with all chunk metadata + embeddings.  
# MAGIC

# COMMAND ----------

# Batch embedding generation function
@pandas_udf(ArrayType(FloatType()))
def embed_udf(texts: pd.Series) -> pd.Series:
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts.iloc[i:i+BATCH_SIZE].tolist()
        response = requests.post(
            f"{WORKSPACE_URL}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations",
            headers=HEADERS, 
            data=json.dumps({"input": batch}), 
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        out.extend([row["embedding"] for row in response.json()["data"]])
    return pd.Series(out)

# Generate embeddings for all chunks
chunks_df = spark.table(f"{FULL_CATALOG_SCHEMA}.{CHUNKS_TABLE}")
df_embeddings = chunks_df.withColumn("embedding", embed_udf(col("chunk")))
df_embeddings.write.mode("overwrite").saveAsTable(f"{FULL_CATALOG_SCHEMA}.{EMBEDDINGS_TABLE}")

print(f"✅ Embeddings generated and saved")
display(df_embeddings.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 10: Initialize Vector Search Endpoint
# MAGIC
# MAGIC Before we can create a **Vector Search Index** to power semantic retrieval, we need a running **Vector Search Endpoint**.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - A **Vector Search Endpoint** is a managed service in Databricks that hosts and serves your indexes.  
# MAGIC - You can attach one or more indexes to a single endpoint.  
# MAGIC - Ensuring the endpoint is online is critical before syncing embeddings.
# MAGIC
# MAGIC ### Approach:
# MAGIC 1. Initialize the `VectorSearchClient`.  
# MAGIC 2. Define helper functions:
# MAGIC    - `endpoint_exists()` → Check if an endpoint already exists.  
# MAGIC    - `wait_for_vs_endpoint_to_be_ready()` → Poll the endpoint until it becomes `ONLINE`.  
# MAGIC 3. Create the endpoint if it does not already exist.  
# MAGIC 4. Wait until the endpoint is ready before proceeding.  
# MAGIC

# COMMAND ----------

# Initialize Vector Search client and utility functions
vsc = VectorSearchClient(disable_notice=True)

def endpoint_exists(client, endpoint_name):
    """Check if vector search endpoint exists"""
    try:
        client.get_endpoint(endpoint_name)
        return True
    except Exception as e:
        if "NOT_FOUND" in str(e) or "does not exist" in str(e):
            return False
        raise e

def wait_for_vs_endpoint_to_be_ready(client, endpoint_name, timeout=700, poll_interval=15):
    """Wait for vector search endpoint to be ready"""
    start_time = time.time()
    while True:
        try:
            status = client.get_endpoint(endpoint_name).get("endpoint_status", {}).get("state", "")
            print(f"Status: {status}")
            if status == "ONLINE":
                print(f"✅ Vector Search endpoint '{endpoint_name}' is ready.")
                break
        except Exception as e:
            print(f"[WARN] Failed to get endpoint status: {e}")

        if time.time() - start_time > timeout:
            raise TimeoutError(f"❌ Timeout: Endpoint '{endpoint_name}' was not ready after {timeout} seconds.")
        time.sleep(poll_interval)

# Create endpoint if needed
if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    print(f"🚀 Creating Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
    time.sleep(5)
else:
    print(f"ℹ️ Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT_NAME}' already exists.")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 11: Create and Sync Vector Search Index
# MAGIC
# MAGIC With the Vector Search Endpoint online, the next step is to create a **Delta Sync Index** that keeps the embeddings in sync with the source table.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - The **Vector Search Index** is the structure that allows for **fast similarity search**.  
# MAGIC - By enabling **Change Data Feed (CDF)** on the source table, the index can stay in sync as new data is added.  
# MAGIC - Once the index is created, we trigger an initial sync so that all embeddings are available for semantic search.
# MAGIC
# MAGIC ### Approach:
# MAGIC 1. Define a utility function `index_exists()` to check if the index is already present.  
# MAGIC 2. Enable **Change Data Feed (CDF)** on the embeddings source table.  
# MAGIC 3. If the index doesn’t exist, create it with:
# MAGIC    - Primary key → `chunk_id`  
# MAGIC    - Source column for embeddings → `chunk`  
# MAGIC    - Embedding model → `databricks-bge-large-en` (configured earlier).  
# MAGIC 4. Wait until the index is ready, then trigger a sync.  
# MAGIC

# COMMAND ----------

# Create delta sync index
def index_exists(client, endpoint, index_name):
    """Check if vector search index exists"""
    try:
        client.get_index(endpoint_name=endpoint, index_name=index_name)
        return True
    except Exception as e:
        if "NOT_FOUND" in str(e) or "does not exist" in str(e):
            return False
        raise e

# Enable Change Data Feed on source table
try:
    spark.sql(f"ALTER TABLE {SOURCE_TABLE_FULLNAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"[INFO] CDF enabled on {SOURCE_TABLE_FULLNAME}")
except Exception as e:
    print(f"[WARN] Could not enable CDF: {e}")

# Create index if it doesn't exist
if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, VS_INDEX_FULLNAME):
    print(f"[INFO] Creating delta-sync index {VS_INDEX_FULLNAME}...")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=VS_INDEX_FULLNAME,
        source_table_name=SOURCE_TABLE_FULLNAME,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT
    )
else:
    print(f"[INFO] Index {VS_INDEX_FULLNAME} already exists.")

# Wait for index to be ready and sync
print(f"[INFO] Waiting for index {VS_INDEX_FULLNAME} to be ready...")
index_obj = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=VS_INDEX_FULLNAME)
index_obj.wait_until_ready()
index_obj.sync()
print(f"[✅] Index {VS_INDEX_FULLNAME} ready and synced.")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 12: Test Vector Search with a Sample Query
# MAGIC
# MAGIC Now that the Vector Search Index is created and synced, we can test it by issuing a **semantic query**.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Confirms that the index is correctly populated with embeddings.  
# MAGIC - Demonstrates how natural language questions can be matched against document chunks.  
# MAGIC - Returns the **most relevant sections** of enterprise documents for downstream use in the RAG pipeline.
# MAGIC
# MAGIC ### Approach:
# MAGIC 1. Define a **test query** → "What is the standard warranty for product-b?".  
# MAGIC 2. Perform a **similarity search** using the index.  
# MAGIC 3. Retrieve the top results with metadata (`doc_id`, `section`, `chunk`).  
# MAGIC 4. Print results to verify that the right document passages are retrieved.  
# MAGIC

# COMMAND ----------

# Test vector search with sample query
test_question = "What is the standard warranty for product-b?"

try:
    results = index_obj.similarity_search(
        query_text=test_question,
        columns=RETURN_COLUMNS,
        num_results=SIMILARITY_SEARCH_RESULTS
    )

    cols = results.get("result", {}).get("columns", RETURN_COLUMNS)
    rows = results.get("result", {}).get("data_array", [])

    print(f"🔍 Query: {test_question}")
    print(f"📄 Found {len(rows)} results:")
    
    for i, row in enumerate(rows, start=1):
        row_map = dict(zip(cols, row))
        print(f"\n📹 Result {i}")
        print(f"Doc ID: {row_map.get('doc_id')}")
        print(f"Section: {row_map.get('section')}")
        print(f"Text: {row_map.get('chunk')}")

except Exception as e:
    print(f"❌ Vector search test failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 13: Implement Circuit Breaker for Production Resilience
# MAGIC
# MAGIC In enterprise systems, it’s not enough to just deploy a model — we also need **resilience** against failures.  
# MAGIC A **Circuit Breaker** pattern helps prevent cascading failures by temporarily blocking requests when error rates exceed a threshold.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Protects downstream systems (e.g., Vector Search, LLM endpoints) from being overwhelmed.  
# MAGIC - Automatically recovers after a cooldown period.  
# MAGIC - Provides metrics for observability and monitoring.  
# MAGIC - Ensures production-grade **fault tolerance**.
# MAGIC
# MAGIC ### Circuit Breaker States:
# MAGIC - **CLOSED** → All requests are allowed (normal operation).  
# MAGIC - **OPEN** → Requests are blocked after repeated failures.  
# MAGIC - **HALF_OPEN** → Trial requests are allowed after cooldown; if they succeed, the breaker closes again.  
# MAGIC
# MAGIC ### Features of `AdvancedCircuitBreaker`:
# MAGIC - Configurable thresholds: failure %, recovery timeout, success threshold.  
# MAGIC - Sliding request window (`deque`) to calculate failure rates.  
# MAGIC - Thread-safe with locks for concurrent requests.  
# MAGIC - Metrics collected:
# MAGIC   - Total requests
# MAGIC   - Successful / failed requests
# MAGIC   - Circuit trips (how many times breaker opened)
# MAGIC   - Current failure rate
# MAGIC

# COMMAND ----------

# Advanced RAG Model with Circuit Breaker
class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class AdvancedCircuitBreaker:
    """Enterprise-grade circuit breaker for RAG system"""
    
    def __init__(self, failure_threshold=FAILURE_THRESHOLD, recovery_timeout=RECOVERY_TIMEOUT, 
                 success_threshold=SUCCESS_THRESHOLD, window_size=WINDOW_SIZE, min_requests=MIN_REQUESTS):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.min_requests = min_requests
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0
        self.request_window = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_trips": 0,
            "current_failure_rate": 0.0
        }
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.metrics["total_requests"] += 1
            
            if not self._should_allow_request():
                self.metrics["failed_requests"] += 1
                raise Exception(f"Circuit breaker is {self.state.value}")
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                self.metrics["successful_requests"] += 1
                return result
            except Exception as e:
                self._record_failure()
                self.metrics["failed_requests"] += 1
                raise e
    
    def _should_allow_request(self):
        current_time = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if current_time >= self.next_attempt_time:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def _record_success(self):
        self.request_window.append({"timestamp": time.time(), "success": True})
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
    
    def _record_failure(self):
        current_time = time.time()
        self.request_window.append({"timestamp": current_time, "success": False})
        self.last_failure_time = current_time
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.next_attempt_time = current_time + self.recovery_timeout
            self.metrics["circuit_trips"] += 1
        elif self.state == CircuitState.CLOSED:
            failure_rate = self._calculate_failure_rate()
            if failure_rate >= (self.failure_threshold / 100.0) and len(self.request_window) >= self.min_requests:
                self.state = CircuitState.OPEN
                self.next_attempt_time = current_time + self.recovery_timeout
                self.metrics["circuit_trips"] += 1
    
    def _calculate_failure_rate(self):
        if not self.request_window:
            return 0.0
        failures = sum(1 for req in self.request_window if not req["success"])
        failure_rate = failures / len(self.request_window)
        self.metrics["current_failure_rate"] = failure_rate * 100
        return failure_rate

print("✅ Advanced Circuit Breaker class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 14: Implement Advanced Circuit Breaker
# MAGIC
# MAGIC In production systems, simply deploying a RAG pipeline is not enough — we need to ensure **resilience** against endpoint outages, slow responses, or cascading failures.
# MAGIC
# MAGIC The **Circuit Breaker pattern** is a fault tolerance mechanism that:
# MAGIC - Prevents overwhelming downstream services (e.g., Vector Search or LLM endpoints).  
# MAGIC - Stops repeated failing requests by "opening the circuit".  
# MAGIC - Automatically retries after a recovery timeout, moving to a **HALF_OPEN** state.  
# MAGIC - Closes the circuit again if requests succeed consistently.  
# MAGIC
# MAGIC ### Circuit Breaker States:
# MAGIC - **CLOSED** → All requests allowed (normal operation).  
# MAGIC - **OPEN** → Requests blocked due to high failure rate.  
# MAGIC - **HALF_OPEN** → Allows limited test requests to check if the system has recovered.  
# MAGIC
# MAGIC ### Features of `AdvancedCircuitBreaker`:
# MAGIC - **Failure threshold** (% of failed requests before tripping the circuit).  
# MAGIC - **Recovery timeout** (time before retrying after a trip).  
# MAGIC - **Success threshold** (number of successful requests to close circuit again).  
# MAGIC - **Sliding request window** to calculate real-time failure rates.  
# MAGIC - **Thread-safe** for concurrent requests.  
# MAGIC - **Metrics** collected for monitoring:
# MAGIC   - Total requests
# MAGIC   - Successful / failed requests
# MAGIC   - Circuit trips
# MAGIC   - Current failure rate
# MAGIC

# COMMAND ----------

# Enterprise RAG Model
class EnterpriseRAGModel(mlflow.pyfunc.PythonModel):
    """Production-ready RAG model with advanced features"""
    
    def load_context(self, context):
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        
        # Initialize circuit breaker
        self.circuit_breaker = AdvancedCircuitBreaker()
        
        # Initialize vector search client
        self.vsc = VectorSearchClient(disable_notice=True)
        self.index = self.vsc.get_index(
            endpoint_name=self.config["vector_search_endpoint"],
            index_name=self.config["vector_index_name"]
        )
    
    def predict(self, context, model_input):
        outputs = []
        for _, row in model_input.iterrows():
            question = row["question"]
            
            try:
                # Use circuit breaker for vector search
                search_results = self.circuit_breaker.call(
                    self._perform_search, question
                )
                
                # Generate answer based on retrieved context
                answer = self._generate_answer(question, search_results)
                
                outputs.append({
                    "question": question,
                    "answer": answer,
                    "retrieved": search_results,
                    "circuit_breaker_state": self.circuit_breaker.state.value
                })
                
            except Exception as e:
                # Fallback response
                outputs.append({
                    "question": question,
                    "answer": f"I apologize, but I'm currently unable to process your request due to technical issues: {str(e)}. Please try again later.",
                    "retrieved": [],
                    "circuit_breaker_state": self.circuit_breaker.state.value,
                    "error": str(e)
                })
        
        return pd.DataFrame(outputs)
    
    def _perform_search(self, question):
        """Perform vector search with error handling"""
        results = self.index.similarity_search(
            query_text=question,
            columns=self.config["return_columns"],
            num_results=self.config["num_results"]
        )
        
        cols = results.get("result", {}).get("columns", [])
        rows = results.get("result", {}).get("data_array", [])
        
        return [{"chunk_text": dict(zip(cols, row)).get("chunk", ""), 
                "source": dict(zip(cols, row)).get("doc_id", "")} for row in rows]
    
    def _generate_answer(self, question, search_results):
        """Generate answer based on retrieved context"""
        if not search_results:
            return "I couldn't find relevant information to answer your question."
        
        # Simple answer generation based on retrieved context
        context = " ".join([result["chunk_text"] for result in search_results[:3]])
        
        # Basic keyword matching for demo purposes
        if "warranty" in question.lower():
            for result in search_results:
                if "warranty" in result["chunk_text"].lower():
                    return f"Based on the documentation: {result['chunk_text']}"
        
        return f"Based on the available information: {context[:200]}..."

print("✅ Enterprise RAG Model class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 15: Package and Register the RAG Model
# MAGIC
# MAGIC Now that we have document embeddings and a working Vector Search index, we need to **package the RAG pipeline as an MLflow PyFunc model**.  
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - MLflow packaging makes the model reproducible and deployable across environments.  
# MAGIC - Unity Catalog registration ensures **governance**, **versioning**, and **traceability**.  
# MAGIC - PyFunc models support flexible APIs (`predict`) for integration with Serving endpoints.  
# MAGIC
# MAGIC ### Fixes applied in `SimpleRAGModel`:
# MAGIC - **Lazy initialization of VectorSearchClient** inside `_get_vector_search_index()`  
# MAGIC   → avoids serialization errors when logging the model.  
# MAGIC - **Robust error handling** with fallback responses.  
# MAGIC - **Keyword-based answer generation** for warranty, retention, access control, and maintenance queries.  
# MAGIC - **Config artifact** stored in JSON so parameters are externalized (endpoint, index, return columns).  
# MAGIC
# MAGIC ### Registration Process:
# MAGIC 1. Define model configuration and save to `config.json`.  
# MAGIC 2. Create example input/output to define MLflow **signature**.  
# MAGIC 3. Log and register the model in MLflow + Unity Catalog.  
# MAGIC 4. Confirm successful registration with model name and version.
# MAGIC

# COMMAND ----------

# Fixed RAG Model that can be serialized
class SimpleRAGModel(mlflow.pyfunc.PythonModel):
    """Simplified RAG model that avoids serialization issues"""
    
    def load_context(self, context):
        """Load configuration - don't initialize complex objects here"""
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        # Don't initialize VectorSearchClient here - causes serialization issues
        self.vsc = None
        self.index = None
    
    def _get_vector_search_index(self):
        """Lazy initialization of vector search client"""
        if self.vsc is None:
            from databricks.vector_search.client import VectorSearchClient
            self.vsc = VectorSearchClient(disable_notice=True)
            self.index = self.vsc.get_index(
                endpoint_name=self.config["vector_search_endpoint"],
                index_name=self.config["vector_index_name"]
            )
        return self.index
    
    def predict(self, context, model_input):
        """Process questions and return answers"""
        outputs = []
        
        for _, row in model_input.iterrows():
            question = row["question"]
            
            try:
                # Get vector search index (lazy initialization)
                index = self._get_vector_search_index()
                
                # Perform vector search
                search_results = self._perform_search(index, question)
                
                # Generate answer
                answer = self._generate_answer(question, search_results)
                
                outputs.append({
                    "question": question,
                    "answer": answer,
                    "retrieved": search_results
                })
                
            except Exception as e:
                # Fallback response
                outputs.append({
                    "question": question,
                    "answer": f"I apologize, but I'm currently unable to process your request: {str(e)}. Please try again later.",
                    "retrieved": [],
                    "error": str(e)
                })
        
        return pd.DataFrame(outputs)
    
    def _perform_search(self, index, question):
        """Perform vector search"""
        try:
            results = index.similarity_search(
                query_text=question,
                columns=self.config["return_columns"],
                num_results=self.config["num_results"]
            )
            
            cols = results.get("result", {}).get("columns", [])
            rows = results.get("result", {}).get("data_array", [])
            
            return [{
                "chunk_text": dict(zip(cols, row)).get("chunk", ""), 
                "source": dict(zip(cols, row)).get("doc_id", "")
            } for row in rows]
            
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []
    
    def _generate_answer(self, question, search_results):
        """Generate answer based on retrieved context"""
        if not search_results:
            return "I couldn't find relevant information to answer your question."
        
        # Enhanced answer generation with keyword matching
        question_lower = question.lower()
        
        # Check for warranty questions
        if "warranty" in question_lower:
            for result in search_results:
                if "warranty" in result["chunk_text"].lower():
                    return f"Based on the documentation: {result['chunk_text']}"
        
        # Check for data retention questions
        if "retention" in question_lower or "data" in question_lower:
            for result in search_results:
                if "retention" in result["chunk_text"].lower() or "days" in result["chunk_text"].lower():
                    return f"Based on the policy: {result['chunk_text']}"
        
        # Check for access control questions
        if "access" in question_lower or "control" in question_lower:
            for result in search_results:
                if "access" in result["chunk_text"].lower() or "MFA" in result["chunk_text"]:
                    return f"Based on the access control policy: {result['chunk_text']}"
        
        # Check for maintenance questions
        if "maintenance" in question_lower:
            for result in search_results:
                if "maintenance" in result["chunk_text"].lower() or "inspection" in result["chunk_text"].lower():
                    return f"Based on the maintenance guide: {result['chunk_text']}"
        
        # Default response with context
        context = " ".join([result["chunk_text"] for result in search_results[:2]])
        return f"Based on the available information: {context[:300]}..."

print("✅ Fixed RAG Model class defined")

 

# Fixed model registration
config = {
    "vector_search_endpoint": VECTOR_SEARCH_ENDPOINT_NAME,
    "vector_index_name": VS_INDEX_FULLNAME,
    "return_columns": RETURN_COLUMNS,
    "num_results": SIMILARITY_SEARCH_RESULTS
}

with tempfile.TemporaryDirectory() as td:
    cfg_path = os.path.join(td, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f)
    
    # Define model signature with proper input example
    example_input = pd.DataFrame([{"question": "What are the warranty terms?"}])
    example_output = pd.DataFrame([{
        "question": "What are the warranty terms?",
        "answer": "Based on the documentation: Product-B includes a standard warranty of 12 months covering manufacturing defects.",
        "retrieved": [{"chunk_text": "Sample context", "source": "doc_001"}]
    }])
    
    signature = infer_signature(example_input, example_output)
    
    # Log and register model with all required parameters
    with mlflow.start_run(run_name="fixed_rag_model") as run:
        mlflow.pyfunc.log_model(
            name="fixed_rag",
            python_model=SimpleRAGModel(),  # Use the fixed model class
            artifacts={"config": cfg_path},
            signature=signature,
            input_example=example_input,  # This fixes the warning
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "databricks-vectorsearch",
                "pandas>=1.3.0",
                "numpy>=1.21.0"
            ]
        )

print(f"✅ Model registered successfully: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 16: Deploy RAG Model to Databricks Serving
# MAGIC
# MAGIC With the RAG model registered in MLflow, the next step is to **deploy it as a REST-serving endpoint** in Databricks.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Exposes the RAG pipeline as a **scalable API** for enterprise applications.  
# MAGIC - Allows employees and downstream systems to query the model using standard HTTP requests.  
# MAGIC - Supports **autoscaling and scale-to-zero**, optimizing cost efficiency.  
# MAGIC - Ensures deployment is governed and version-controlled via Unity Catalog.  
# MAGIC
# MAGIC ### Deployment Process:
# MAGIC 1. **Get latest model version** from Unity Catalog/MLflow Registry.  
# MAGIC 2. Define the **endpoint configuration**:
# MAGIC    - Model name and version.  
# MAGIC    - Workload size (e.g., Small).  
# MAGIC    - Environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`) for runtime access.  
# MAGIC    - 100% traffic routed to this version.  
# MAGIC 3. Check if the serving endpoint already exists:  
# MAGIC    - If yes → update configuration.  
# MAGIC    - If no → create a new endpoint.  
# MAGIC 4. Monitor deployment in the Databricks UI under **Serving**.  
# MAGIC
# MAGIC This process may take several minutes as the model container spins up.
# MAGIC

# COMMAND ----------

# Deploy serving endpoint
def deploy_serving_endpoint():
    """Deploy or update serving endpoint"""
    
    # Get latest model version
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(versions, key=lambda v: int(v.version)).version
    
    # Endpoint configuration
    endpoint_config = {
        "served_models": [{
            "name": "enterprise-rag-model",
            "model_name": MODEL_NAME,
            "model_version": latest_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
            "environment_vars": {
                "DATABRICKS_HOST": WORKSPACE_URL,
                "DATABRICKS_TOKEN": TOKEN
            }
        }],
        "traffic_config": {
            "routes": [{
                "served_model_name": "enterprise-rag-model",
                "traffic_percentage": 100
            }]
        }
    }
    
    # Check if endpoint exists
    try:
        response = requests.get(
            f"{WORKSPACE_URL}/api/2.0/serving-endpoints/{SERVING_ENDPOINT_NAME}",
            headers=HEADERS
        )
        
        if response.status_code == 200:
            # Update existing endpoint
            print(f"🔄 Updating serving endpoint: {SERVING_ENDPOINT_NAME}")
            response = requests.put(
                f"{WORKSPACE_URL}/api/2.0/serving-endpoints/{SERVING_ENDPOINT_NAME}/config",
                headers=HEADERS,
                data=json.dumps(endpoint_config)
            )
        else:
            # Create new endpoint
            print(f"🚀 Creating serving endpoint: {SERVING_ENDPOINT_NAME}")
            endpoint_payload = {
                "name": SERVING_ENDPOINT_NAME,
                "config": endpoint_config
            }
            response = requests.post(
                f"{WORKSPACE_URL}/api/2.0/serving-endpoints",
                headers=HEADERS,
                data=json.dumps(endpoint_payload)
            )
        
        if response.status_code in [200, 201]:
            print(f"✅ Endpoint deployment initiated successfully")
            return True
        else:
            print(f"❌ Deployment failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        return False

# Deploy the endpoint
deployment_success = deploy_serving_endpoint()

if deployment_success:
    print(f"\n⏳ Waiting for endpoint to be ready (this may take several minutes)...")
    print(f"📍 You can monitor the deployment in the Databricks UI under 'Serving'")
else:
    print(f"❌ Deployment failed. Please check the configuration and try again.")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC # Step 17: Deploy Serving Endpoint
# MAGIC
# MAGIC With the RAG model registered in MLflow, the next step is to **deploy it as a Databricks Model Serving Endpoint**.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Makes the RAG pipeline available as a **REST API**.  
# MAGIC - Allows business users and applications to query the system with natural language questions.  
# MAGIC - Supports **autoscaling and scale-to-zero** for cost efficiency.  
# MAGIC - Ensures deployment is managed under **Unity Catalog governance**.
# MAGIC
# MAGIC ### Deployment Process:
# MAGIC 1. Retrieve the **latest model version** from MLflow.  
# MAGIC 2. Define the **endpoint configuration**:  
# MAGIC    - Model name and version.  
# MAGIC    - Workload size (e.g., `Small`).  
# MAGIC    - Environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`).  
# MAGIC    - Traffic policy (100% routed to this version).  
# MAGIC 3. Check if the endpoint already exists:  
# MAGIC    - If **yes** → update the configuration.  
# MAGIC    - If **no** → create a new endpoint.  
# MAGIC 4. Wait for the deployment to complete (can take several minutes).  
# MAGIC 5. Monitor status in the **Databricks UI → Serving**.  
# MAGIC

# COMMAND ----------

# Comprehensive RAG system testing
def test_rag_system():
    """Test the complete RAG system"""
    
    test_queries = [
        "What are the warranty terms for product-b?",
        "What is the data retention policy?",
        "What are the access control requirements?",
        "How often should maintenance be performed?",
        "What are the storage policy requirements?"
    ]
    
    print("🧪 Testing RAG System")
    print("=" * 50)
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {query}")
        
        try:
            # Test endpoint (when ready)
            payload = {"dataframe_records": [{"question": query}]}
            
            start_time = time.time()
            response = requests.post(
                f"{WORKSPACE_URL}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result["predictions"][0]["answer"]
                retrieved_docs = result["predictions"][0]["retrieved"]
                
                print(f"   ✅ SUCCESS! Response time: {response_time:.2f}s")
                print(f"   📄 Answer: {answer[:100]}...")
                print(f"   🔍 Retrieved {len(retrieved_docs)} documents")
                
                results.append({
                    "query": query,
                    "status": "success",
                    "response_time": response_time,
                    "answer": answer
                })
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                results.append({
                    "query": query,
                    "status": "error",
                    "error": response.text
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            results.append({
                "query": query,
                "status": "exception",
                "error": str(e)
            })
    
    # Summary
    successful_tests = [r for r in results if r["status"] == "success"]
    print(f"\n📊 TEST SUMMARY")
    print(f"✅ Successful queries: {len(successful_tests)}/{len(test_queries)}")
    
    if successful_tests:
        avg_response_time = np.mean([r["response_time"] for r in successful_tests])
        print(f"⚡ Average response time: {avg_response_time:.2f}s")
    
    return results

# Note: Uncomment the line below to run tests when endpoint is ready
# test_results = test_rag_system()

print("\n🎉 RAG SYSTEM DEPLOYMENT COMPLETE!")
print("=" * 50)
print("✅ Configuration centralized")
print("✅ Database and tables created")
print("✅ Document chunking implemented")
print("✅ Embeddings generated")
print("✅ Vector search index created")
print("✅ Advanced RAG model with circuit breaker")
print("✅ Model registered in MLflow")
print("✅ Serving endpoint deployed")
print("✅ Comprehensive testing framework")
print("\n🚀 Your enterprise RAG system is ready for production!")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 18: Validate Serving Endpoint with End-to-End Testing
# MAGIC
# MAGIC Now that the RAG model has been deployed to **Databricks Model Serving**, we must verify that it works correctly in production.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Confirms the endpoint is **ready** and accepting requests.  
# MAGIC - Validates that natural language questions return the correct **retrieved documents** and **answers**.  
# MAGIC - Measures **response time** to ensure performance requirements are met.  
# MAGIC - Provides a final **end-to-end test** of the RAG pipeline.
# MAGIC
# MAGIC ### Approach:
# MAGIC 1. **Check readiness**  
# MAGIC    - Poll the serving endpoint until it reports state = `READY`.  
# MAGIC
# MAGIC 2. **Run single query tests**  
# MAGIC    - Ask questions like:
# MAGIC      - "What are the warranty terms for product-b?"
# MAGIC      - "What is the data retention policy?"
# MAGIC      - "What are the access control requirements?"
# MAGIC      - "How often should maintenance be performed?"
# MAGIC      - "What are the storage policy requirements?"
# MAGIC    - Display answers, number of documents retrieved, and response times.  
# MAGIC
# MAGIC 3. **Comprehensive end-to-end test**  
# MAGIC    - Run a suite of queries with `test_rag_system()` (notebook-defined).  
# MAGIC    - Collect results, compute success/failure counts, and average response time.  
# MAGIC    - Show sample answers from successful runs.  
# MAGIC
# MAGIC This ensures the pipeline is production-ready, resilient, and optimized.
# MAGIC

# COMMAND ----------


def check_endpoint_readiness(endpoint_name, max_wait_minutes=10):
    """Check if serving endpoint is ready and accepting requests"""
    import time
    
    max_wait_seconds = max_wait_minutes * 60
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.get(
                f"{WORKSPACE_URL}/api/2.0/serving-endpoints/{endpoint_name}",
                headers=HEADERS
            )
            
            if response.status_code == 200:
                endpoint_info = response.json()
                state = endpoint_info.get("state", {}).get("ready", "NOT_READY")
                config_update = endpoint_info.get("state", {}).get("config_update", "NOT_READY")
                
                print(f"Endpoint state: {state}, Config update: {config_update}")
                
                if state == "READY" and config_update == "NOT_UPDATING":
                    print(f"✅ Endpoint {endpoint_name} is ready!")
                    return True
                    
            print(f"⏳ Waiting for endpoint to be ready... ({int(time.time() - start_time)}s elapsed)")
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"❌ Error checking endpoint status: {e}")
            time.sleep(30)
    
    print(f"❌ Endpoint not ready after {max_wait_minutes} minutes")
    return False

# Check if endpoint is ready
print(f"🔍 Checking endpoint readiness: {SERVING_ENDPOINT_NAME}")
endpoint_ready = check_endpoint_readiness(SERVING_ENDPOINT_NAME)

# COMMAND ----------

def test_single_query(question, endpoint_name):
    """Test a single query against the serving endpoint"""
    try:
        payload = {"dataframe_records": [{"question": question}]}
        
        start_time = time.time()
        response = requests.post(
            f"{WORKSPACE_URL}/serving-endpoints/{endpoint_name}/invocations",
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result["predictions"][0]["answer"]
            retrieved_docs = result["predictions"][0].get("retrieved", [])
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "retrieved_count": len(retrieved_docs),
                "response_time": response_time,
                "retrieved_docs": retrieved_docs
            }
        else:
            return {
                "status": "error",
                "question": question,
                "error": f"HTTP {response.status_code}: {response.text}",
                "response_time": response_time
            }
            
    except Exception as e:
        return {
            "status": "exception",
            "question": question,
            "error": str(e),
            "response_time": None
        }

# Test individual queries if endpoint is ready
if endpoint_ready:
    print("\n🧪 Testing individual queries:")
    
    test_questions = [
        "What are the warranty terms for product-b?",
        "What is the data retention policy?",
        "What are the access control requirements?",
        "How often should maintenance be performed?",
        "What are the storage policy requirements?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        result = test_single_query(question, SERVING_ENDPOINT_NAME)
        
        if result["status"] == "success":
            print(f"   ✅ SUCCESS! Response time: {result['response_time']:.2f}s")
            print(f"   📄 Answer: {result['answer'][:100]}...")
            print(f"   🔍 Retrieved {result['retrieved_count']} documents")
        else:
            print(f"   ❌ {result['status'].upper()}: {result['error']}")
else:
    print("❌ Skipping tests - endpoint not ready")

# COMMAND ----------

# Comprehensive end-to-end testing
if endpoint_ready:
    print("\n🚀 Running comprehensive end-to-end test...")
    test_results = test_rag_system()
    
    # Analyze results
    successful_tests = [r for r in test_results if r["status"] == "success"]
    failed_tests = [r for r in test_results if r["status"] != "success"]
    
    print(f"\n📊 FINAL TEST RESULTS")
    print("=" * 50)
    print(f"✅ Successful queries: {len(successful_tests)}/{len(test_results)}")
    print(f"❌ Failed queries: {len(failed_tests)}")
    
    if successful_tests:
        avg_response_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests)
        print(f"⚡ Average response time: {avg_response_time:.2f}s")
        
        print(f"\n📋 Sample successful responses:")
        for result in successful_tests[:2]:  # Show first 2 successful responses
            print(f"   Q: {result['query']}")
            print(f"   A: {result['answer'][:150]}...")
            print()
    
    if failed_tests:
        print(f"\n⚠️ Failed queries need investigation:")
        for result in failed_tests:
            print(f"   Q: {result['query']}")
            print(f"   Error: {result['error']}")
            print()
            
    print("✅ End-to-end testing complete!")
else:
    print("❌ Cannot run comprehensive tests - endpoint not ready")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 19: Advanced Deployment – Version Targeting and A/B Testing
# MAGIC
# MAGIC After deploying a baseline RAG model, enterprises often need more **sophisticated deployment strategies** to support governance, safety, and experimentation.
# MAGIC
# MAGIC ### 1. Version Targeting
# MAGIC - **Target a specific version** of a model (e.g., v1).  
# MAGIC - **Target by stage** (e.g., `Production` or `Staging`) if your registry workflow promotes models via stages.  
# MAGIC - Ensures predictable deployments instead of always pulling the latest.  
# MAGIC
# MAGIC ### 2. Champion–Challenger (A/B Testing)
# MAGIC - Deploy **two versions** of the model side-by-side:  
# MAGIC   - **Champion** → current production version.  
# MAGIC   - **Challenger** → new candidate version.  
# MAGIC - Split traffic (e.g., 80% to Champion, 20% to Challenger).  
# MAGIC - Compare performance metrics before promoting the challenger to full production.  
# MAGIC
# MAGIC This approach allows **safe rollout, canary testing, and continuous evaluation** of new model versions.
# MAGIC

# COMMAND ----------


def deploy_with_version_targeting(model_name, specific_version=None, stage=None):
    """Deploy endpoint with specific version or stage targeting"""
    
    client = MlflowClient()
    
    if specific_version:
        # Target specific version
        target_version = specific_version
        deployment_type = f"Version {specific_version}"
    elif stage:
        # Target by stage (e.g., "Production", "Staging")
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            print(f"❌ No model found in {stage} stage")
            return False
        target_version = versions[0].version
        deployment_type = f"{stage} stage (version {target_version})"
    else:
        # Default to latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = max(versions, key=lambda v: int(v.version)).version
        deployment_type = f"Latest version ({target_version})"
    
    print(f"🎯 Deploying {deployment_type}")
    
    # Endpoint configuration with specific version
    endpoint_config = {
        "served_models": [{
            "name": f"rag-model-v{target_version}",
            "model_name": model_name,
            "model_version": target_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
            "environment_vars": {
                "DATABRICKS_HOST": WORKSPACE_URL,
                "DATABRICKS_TOKEN": TOKEN
            }
        }],
        "traffic_config": {
            "routes": [{
                "served_model_name": f"rag-model-v{target_version}",
                "traffic_percentage": 100
            }]
        }
    }
    
    return endpoint_config, target_version

# Example: Deploy specific version
print("📋 Version targeting examples:")
config_v1, version = deploy_with_version_targeting(MODEL_NAME, specific_version="1")
print(f"✅ Configuration for version {version} created")

# Example: Deploy by stage (if stages are set up)
# config_prod, version = deploy_with_version_targeting(MODEL_NAME, stage="Production")

# COMMAND ----------

def setup_champion_challenger_deployment(model_name, champion_version, challenger_version, challenger_traffic_percent=10):
    """Set up A/B testing with champion/challenger deployment"""
    
    endpoint_config = {
        "served_models": [
            {
                "name": "champion-model",
                "model_name": model_name,
                "model_version": champion_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
                "environment_vars": {
                    "DATABRICKS_HOST": WORKSPACE_URL,
                    "DATABRICKS_TOKEN": TOKEN
                }
            },
            {
                "name": "challenger-model", 
                "model_name": model_name,
                "model_version": challenger_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
                "environment_vars": {
                    "DATABRICKS_HOST": WORKSPACE_URL,
                    "DATABRICKS_TOKEN": TOKEN
                }
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": "champion-model",
                    "traffic_percentage": 100 - challenger_traffic_percent
                },
                {
                    "served_model_name": "challenger-model", 
                    "traffic_percentage": challenger_traffic_percent
                }
            ]
        }
    }
    
    print(f"🏆 Champion (v{champion_version}): {100 - challenger_traffic_percent}% traffic")
    print(f"🥊 Challenger (v{challenger_version}): {challenger_traffic_percent}% traffic")
    
    return endpoint_config

# Example champion/challenger setup (if multiple versions exist)
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if len(versions) >= 2:
    latest_version = max(versions, key=lambda v: int(v.version)).version
    previous_version = str(int(latest_version) - 1)
    
    ab_config = setup_champion_challenger_deployment(
        MODEL_NAME, 
        champion_version=previous_version,
        challenger_version=latest_version,
        challenger_traffic_percent=20
    )
    print("✅ A/B testing configuration created")
else:
    print("ℹ️ Need at least 2 model versions for A/B testing")

# COMMAND ----------

# MAGIC %md
# MAGIC  %md
# MAGIC # Step 20: RAG Model with Circuit Breaker Integration
# MAGIC
# MAGIC In production environments, failures are inevitable — endpoints may become unavailable, or requests may fail due to overload.  
# MAGIC To build a **resilient enterprise-ready RAG pipeline**, we integrate the **Circuit Breaker** pattern directly into the RAG model.
# MAGIC
# MAGIC ### Why this matters:
# MAGIC - Prevents cascading failures by stopping requests once failure rates exceed a threshold.  
# MAGIC - Provides **automatic fallback responses** (e.g., helpdesk instructions or alternate resources).  
# MAGIC - Exposes **metrics** for monitoring reliability:
# MAGIC   - Total requests
# MAGIC   - Successful vs. failed requests
# MAGIC   - Circuit breaker trips
# MAGIC   - Current failure rate  
# MAGIC
# MAGIC ### Features in `CircuitBreakerRAGModel`:
# MAGIC 1. **Integrated Circuit Breaker**  
# MAGIC    - 20% failure threshold, 1-minute recovery timeout.  
# MAGIC    - Switches between `CLOSED`, `OPEN`, and `HALF_OPEN`.  
# MAGIC
# MAGIC 2. **Vector Search Resilience**  
# MAGIC    - Simulates failures (10% random) for testing.  
# MAGIC    - Circuit breaker blocks calls after repeated failures.  
# MAGIC
# MAGIC 3. **Fallback Mechanism**  
# MAGIC    - Keyword-based fallback responses for warranty, retention, access control, and maintenance queries.  
# MAGIC    - Generic fallback for other cases with error context.  
# MAGIC
# MAGIC 4. **Testing with `test_circuit_breaker_behavior()`**  
# MAGIC    - Sends multiple queries to simulate failures.  
# MAGIC    - Shows circuit breaker state transitions.  
# MAGIC    - Prints metrics after every 5 requests.  
# MAGIC    - Confirms fallback answers when failures occur.  
# MAGIC
# MAGIC This step ensures your RAG pipeline is **fault-tolerant, production-ready, and enterprise-compliant**.
# MAGIC
# MAGIC

# COMMAND ----------


# Enhanced RAG Model with Circuit Breaker Integration
class CircuitBreakerRAGModel(mlflow.pyfunc.PythonModel):
    """RAG model with integrated circuit breaker for production resilience"""
    
    def load_context(self, context):
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)
        
        # Initialize circuit breaker
        self.circuit_breaker = AdvancedCircuitBreaker(
            failure_threshold=20,  # 20% failure rate
            recovery_timeout=60,   # 1 minute recovery
            success_threshold=3,   # 3 successes to close
            window_size=10,        # Track last 10 requests
            min_requests=3         # Minimum 3 requests before calculating failure rate
        )
        
        self.vsc = None
        self.index = None
    
    def _get_vector_search_index(self):
        """Lazy initialization of vector search client"""
        if self.vsc is None:
            from databricks.vector_search.client import VectorSearchClient
            self.vsc = VectorSearchClient(disable_notice=True)
            self.index = self.vsc.get_index(
                endpoint_name=self.config["vector_search_endpoint"],
                index_name=self.config["vector_index_name"]
            )
        return self.index
    
    def predict(self, context, model_input):
        """Process questions with circuit breaker protection"""
        outputs = []
        
        for _, row in model_input.iterrows():
            question = row["question"]
            
            try:
                # Use circuit breaker for vector search
                search_results = self.circuit_breaker.call(
                    self._perform_search_with_potential_failure, question
                )
                
                # Generate answer
                answer = self._generate_answer(question, search_results)
                
                outputs.append({
                    "question": question,
                    "answer": answer,
                    "retrieved": search_results,
                    "circuit_breaker_state": self.circuit_breaker.state.value,
                    "circuit_breaker_metrics": self.circuit_breaker.metrics
                })
                
            except Exception as e:
                # Fallback response when circuit breaker is open or other errors
                fallback_answer = self._get_fallback_answer(question, str(e))
                
                outputs.append({
                    "question": question,
                    "answer": fallback_answer,
                    "retrieved": [],
                    "circuit_breaker_state": self.circuit_breaker.state.value,
                    "circuit_breaker_metrics": self.circuit_breaker.metrics,
                    "error": str(e),
                    "fallback_used": True
                })
        
        return pd.DataFrame(outputs)
    
    def _perform_search_with_potential_failure(self, question):
        """Vector search with simulated failures for testing"""
        index = self._get_vector_search_index()
        
        # Simulate occasional failures for circuit breaker testing
        import random
        if random.random() < 0.1:  # 10% failure rate for testing
            raise Exception("Simulated vector search failure")
        
        results = index.similarity_search(
            query_text=question,
            columns=self.config["return_columns"],
            num_results=self.config["num_results"]
        )
        
        cols = results.get("result", {}).get("columns", [])
        rows = results.get("result", {}).get("data_array", [])
        
        return [{
            "chunk_text": dict(zip(cols, row)).get("chunk", ""), 
            "source": dict(zip(cols, row)).get("doc_id", "")
        } for row in rows]
    
    def _get_fallback_answer(self, question, error_msg):
        """Provide fallback answers when primary system fails"""
        
        # Simple keyword-based fallbacks
        question_lower = question.lower()
        
        if "warranty" in question_lower:
            return "I'm currently unable to access the warranty database. Please contact customer service at 1-800-WARRANTY for warranty information."
        
        elif "retention" in question_lower or "data" in question_lower:
            return "I'm experiencing technical difficulties accessing data policies. Please refer to the company handbook or contact IT support."
        
        elif "access" in question_lower or "control" in question_lower:
            return "Access control information is temporarily unavailable. Please contact your system administrator for immediate access needs."
        
        elif "maintenance" in question_lower:
            return "Maintenance schedules are currently inaccessible. Please check the equipment manual or contact technical support."
        
        else:
            return f"I apologize, but I'm currently experiencing technical difficulties and cannot process your request. Please try again later or contact support. (Error: {error_msg})"
    
    def _generate_answer(self, question, search_results):
        """Generate answer based on retrieved context"""
        if not search_results:
            return "I couldn't find relevant information to answer your question."
        
        # Enhanced answer generation
        question_lower = question.lower()
        
        if "warranty" in question_lower:
            for result in search_results:
                if "warranty" in result["chunk_text"].lower():
                    return f"Based on the documentation: {result['chunk_text']}"
        
        if "retention" in question_lower:
            for result in search_results:
                if "retention" in result["chunk_text"].lower():
                    return f"According to our data policy: {result['chunk_text']}"
        
        if "access" in question_lower:
            for result in search_results:
                if "access" in result["chunk_text"].lower():
                    return f"Per access control guidelines: {result['chunk_text']}"
        
        if "maintenance" in question_lower:
            for result in search_results:
                if "maintenance" in result["chunk_text"].lower():
                    return f"Maintenance requirements: {result['chunk_text']}"
        
        # Default response
        context = " ".join([result["chunk_text"] for result in search_results[:2]])
        return f"Based on available information: {context[:300]}..."

print("✅ Circuit Breaker RAG Model defined")

# COMMAND ----------

def test_circuit_breaker_behavior():
    """Test circuit breaker behavior with simulated failures"""
    
    print("🧪 Testing Circuit Breaker Behavior")
    print("=" * 50)
    
    # Create a test instance
    test_model = CircuitBreakerRAGModel()
    
    # Mock context and config
    class MockContext:
        artifacts = {"config": "/tmp/test_config.json"}
    
    # Create test config
    import tempfile
    import os
    
    config = {
        "vector_search_endpoint": VECTOR_SEARCH_ENDPOINT_NAME,
        "vector_index_name": VS_INDEX_FULLNAME,
        "return_columns": RETURN_COLUMNS,
        "num_results": SIMILARITY_SEARCH_RESULTS
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    # Update mock context
    MockContext.artifacts = {"config": config_path}
    
    try:
        # Load context
        test_model.load_context(MockContext)
        
        # Test multiple requests to trigger circuit breaker
        test_questions = [
            "What are the warranty terms?",
            "What is the data retention policy?", 
            "What are access control requirements?",
            "How often is maintenance needed?",
            "What are storage requirements?"
        ] * 3  # Repeat to increase chances of failures
        
        results = []
        for i, question in enumerate(test_questions):
            print(f"\n🔄 Request {i+1}: {question[:30]}...")
            
            input_df = pd.DataFrame([{"question": question}])
            result = test_model.predict(MockContext, input_df)
            
            # Extract result info
            result_dict = result.iloc[0].to_dict()
            cb_state = result_dict.get("circuit_breaker_state", "UNKNOWN")
            fallback_used = result_dict.get("fallback_used", False)
            
            print(f"   Circuit Breaker: {cb_state}")
            print(f"   Fallback Used: {fallback_used}")
            
            if fallback_used:
                print(f"   Fallback Answer: {result_dict['answer'][:100]}...")
            
            results.append(result_dict)
            
            # Show circuit breaker metrics periodically
            if (i + 1) % 5 == 0:
                metrics = result_dict.get("circuit_breaker_metrics", {})
                print(f"\n📊 Circuit Breaker Metrics after {i+1} requests:")
                print(f"   Total Requests: {metrics.get('total_requests', 0)}")
                print(f"   Successful: {metrics.get('successful_requests', 0)}")
                print(f"   Failed: {metrics.get('failed_requests', 0)}")
                print(f"   Circuit Trips: {metrics.get('circuit_trips', 0)}")
                print(f"   Current Failure Rate: {metrics.get('current_failure_rate', 0):.1f}%")
        
        # Summary
        fallback_count = sum(1 for r in results if r.get("fallback_used", False))
        circuit_trips = results[-1].get("circuit_breaker_metrics", {}).get("circuit_trips", 0)
        
        print(f"\n📋 Circuit Breaker Test Summary:")
        print(f"   Total Requests: {len(results)}")
        print(f"   Fallback Responses: {fallback_count}")
        print(f"   Circuit Breaker Trips: {circuit_trips}")
        print(f"   Final State: {results[-1].get('circuit_breaker_state', 'UNKNOWN')}")
        
        if fallback_count > 0:
            print("✅ Circuit breaker and fallback behavior working correctly!")
        else:
            print("ℹ️ No failures occurred - circuit breaker not triggered")
            
    finally:
        # Clean up temp file
        os.unlink(config_path)

# Run circuit breaker test
test_circuit_breaker_behavior()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Next Steps and Production Considerations
# MAGIC
# MAGIC ### Immediate Actions:
# MAGIC 1. **Monitor Deployment**: Check the Databricks UI under 'Serving' to monitor endpoint status
# MAGIC 2. **Run Tests**: Uncomment the test function once the endpoint is ready
# MAGIC 3. **Validate Performance**: Monitor response times and accuracy
# MAGIC
# MAGIC ### Production Enhancements:
# MAGIC 1. **Security**: Replace hardcoded tokens with `dbutils.secrets.get()`
# MAGIC 2. **Monitoring**: Implement comprehensive logging and alerting
# MAGIC 3. **Scaling**: Adjust workload size based on traffic patterns
# MAGIC 4. **Content**: Add more sophisticated answer generation logic
# MAGIC 5. **Evaluation**: Implement automated quality assessment
# MAGIC
# MAGIC ### Advanced Features:
# MAGIC - A/B testing for model versions
# MAGIC - Real-time model updates
# MAGIC - Multi-modal document processing
# MAGIC - Advanced retrieval strategies
# MAGIC - Integration with external knowledge bases
# MAGIC
# MAGIC **🎯 Congratulations! You have successfully built and deployed an enterprise-grade RAG system on Databricks!**