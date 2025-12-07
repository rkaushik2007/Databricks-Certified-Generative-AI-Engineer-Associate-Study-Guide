# Databricks notebook source
# MAGIC %md
# MAGIC # Hands-On Lab: Building a Scalable Vector Search and Retrieval System
# MAGIC
# MAGIC ## Scenario
# MAGIC
# MAGIC You are a **generative AI engineer** responsible for designing a high‑performance retrieval‑augmented generation (RAG) system that supports thousands of daily queries from analysts across your organization. Users rely on the system to search technical documents, internal wikis, and operational reports. Recently, leadership approved a redesign of the system to improve retrieval accuracy, accelerate response time, and support higher traffic volumes.
# MAGIC
# MAGIC Your task is to build and optimize the retrieval layer using **Databricks Mosaic AI and Vector Search**. The lab requires you to:
# MAGIC
# MAGIC 1. Create an embedding pipeline using a Databricks‑hosted embedding model.
# MAGIC 2. Build a Vector Search index and populate it with documents.
# MAGIC 3. Configure a retriever to power semantic search.
# MAGIC 4. Serve the LLM and embedding model using Mosaic AI Model Serving.
# MAGIC 5. Implement batching and adjust context length to improve throughput.
# MAGIC 6. Profile RAG performance and identify bottlenecks.
# MAGIC 7. Optimize system behavior using the tuning strategies introduced in Chapter 9, including context‑length tuning, embedding dimensionality adjustments, batching optimization, and index‑level performance improvements.
# MAGIC
# MAGIC This hands‑on scenario mirrors real enterprise workloads, where retrieval performance and scalability directly influence the usefulness of generative AI systems. By completing this lab, you will apply the full set of concepts from Chapter 9—including embedding selection, index creation, batch inference, and profiling—to build a robust RAG pipeline.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC - ✅ Design and implement a scalable Vector Search index.
# MAGIC - ✅ Choose and apply appropriate embedding models.
# MAGIC - ✅ Serve embedding and LLM endpoints using Mosaic AI.
# MAGIC - ✅ Configure and tune batching, context length, and concurrency.
# MAGIC - ✅ Diagnose performance issues using profiling and metrics.
# MAGIC - ✅ Apply optimization strategies to improve accuracy and throughput.
# MAGIC - ✅ Connect system bottlenecks to root‑cause adjustments using the patterns introduced in Chapter 9, such as identifying slow vector search, diagnosing model execution delays, detecting inefficient batching, and resolving orchestration or data‑retrieval slowdowns.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC Before starting this lab, ensure you have:
# MAGIC
# MAGIC 1. **Databricks Workspace** with Unity Catalog enabled
# MAGIC 2. **Cluster** with Databricks Runtime 14.3 LTS ML or higher
# MAGIC 3. **Vector Search** endpoint created in your workspace
# MAGIC 4. **Model Serving** permissions enabled
# MAGIC 5. Access to **Foundation Model APIs** (for embedding and LLM models)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Lab Duration
# MAGIC
# MAGIC **Estimated Time:** 90-120 minutes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────┐
# MAGIC │                        RAG System Architecture                          │
# MAGIC ├─────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                         │
# MAGIC │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
# MAGIC │  │   Documents  │───▶│  Embedding   │───▶│   Vector Search Index    │  │
# MAGIC │  │   (Delta)    │    │   Pipeline   │    │   (Mosaic AI)            │  │
# MAGIC │  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
# MAGIC │                                                    │                    │
# MAGIC │                                                    ▼                    │
# MAGIC │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
# MAGIC │  │  User Query  │───▶│  Retriever   │───▶│   LLM Model Serving      │  │
# MAGIC │  │              │    │              │    │   (Response Generation)  │  │
# MAGIC │  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
# MAGIC │                                                                         │
# MAGIC └─────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 1: Environment Setup and Configuration
# MAGIC
# MAGIC In this section, we will:
# MAGIC - Install required libraries
# MAGIC - Configure workspace settings
# MAGIC - Set up catalog and schema for our data
# MAGIC
# MAGIC This establishes the foundation for our Vector Search and RAG implementation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Required Packages
# MAGIC
# MAGIC The following cell installs the Python packages required for this lab:
# MAGIC
# MAGIC | Package | Purpose |
# MAGIC |---------|---------|
# MAGIC | `databricks-vectorsearch` | Client library for creating and querying Vector Search indices |
# MAGIC | `databricks-sdk` | Unified SDK for interacting with Databricks workspace APIs |
# MAGIC | `langchain` | Framework for building LLM-powered applications |
# MAGIC | `langchain-community` | Community integrations including Databricks connectors |
# MAGIC | `langchain-text-splitters` | Text chunking utilities for document processing |
# MAGIC | `langchain-core` | Core abstractions for prompts, output parsers, and runnables |
# MAGIC | `tiktoken` | OpenAI's tokenizer for counting tokens in text |
# MAGIC
# MAGIC **Why these packages?**
# MAGIC - Vector Search requires the `databricks-vectorsearch` client to create indices and perform similarity searches
# MAGIC - LangChain provides a unified interface for embedding models, LLMs, and retrieval pipelines
# MAGIC - The `dbutils.library.restartPython()` ensures the newly installed packages are available in the current session

# COMMAND ----------

# Install required packages for Vector Search and RAG pipeline
# Note: Using compatible versions for Databricks Runtime 14.3+ LTS ML
%pip install databricks-vectorsearch databricks-sdk langchain>=0.2.0 langchain-community>=0.2.0 langchain-text-splitters>=0.2.0 langchain-core>=0.2.0 tiktoken --quiet

# Restart Python to ensure new packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Required Libraries
# MAGIC
# MAGIC The following cell imports all necessary libraries organized by their function:
# MAGIC
# MAGIC **Standard Libraries:**
# MAGIC - `os`, `time`, `json`, `datetime` - System utilities for file operations, timing, and data handling
# MAGIC - `typing` - Type hints for better code documentation
# MAGIC
# MAGIC **PySpark & Delta Lake:**
# MAGIC - `SparkSession` - Entry point for Spark functionality
# MAGIC - `pyspark.sql.functions` - DataFrame transformation functions (col, lit, concat, etc.)
# MAGIC - `pyspark.sql.types` - Schema definitions for structured data
# MAGIC
# MAGIC **Databricks SDK & Vector Search:**
# MAGIC - `WorkspaceClient` - Interact with Databricks workspace (list endpoints, manage resources)
# MAGIC - `VectorSearchClient` - Create, manage, and query Vector Search indices
# MAGIC
# MAGIC **MLflow:**
# MAGIC - `mlflow` - Experiment tracking and model registry
# MAGIC - `MlflowClient` - Programmatic access to MLflow tracking server
# MAGIC
# MAGIC **LangChain Components:**
# MAGIC - `RecursiveCharacterTextSplitter` - Splits documents into chunks while respecting natural boundaries
# MAGIC - `DatabricksEmbeddings` - Wrapper for Databricks-hosted embedding models
# MAGIC - `ChatDatabricks` - Wrapper for Databricks-hosted LLM endpoints
# MAGIC - `DatabricksVectorSearch` - LangChain integration with Databricks Vector Search
# MAGIC
# MAGIC **Data Analysis:**
# MAGIC - `pandas`, `numpy` - Data manipulation and numerical operations for profiling

# COMMAND ----------

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

# Standard library imports
import os
import time
import json
import warnings
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
import io

# Suppress deprecation warnings from LangChain and other libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Suppress Databricks SDK authentication notices in notebook output
# These notices recommend Service Principal auth for production but are informational only

# Method 1: Environment variable to disable notices globally
os.environ["DATABRICKS_SDK_UPSTREAM"] = "true"
os.environ["DATABRICKS_SDK_NO_NOTICE"] = "true"

# Method 2: Suppress logging-based notices before any SDK imports
import logging
logging.getLogger("databricks.sdk").setLevel(logging.ERROR)
logging.getLogger("databricks.vector_search").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)

# Method 3: Patch the print function to filter notices
_original_print = print
def _filtered_print(*args, **kwargs):
    """Filter out [NOTICE] messages from print output."""
    if args:
        text = str(args[0])
        if "[NOTICE]" in text or "Using a notebook authentication token" in text:
            return  # Suppress the notice
    return _original_print(*args, **kwargs)

# Apply the filter (comment out if you want to see notices)
import builtins
builtins.print = _filtered_print

# PySpark and Delta Lake for distributed data processing
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat, monotonically_increasing_id, udf, current_timestamp
from pyspark.sql.types import StringType, ArrayType, FloatType, StructType, StructField, IntegerType

# Databricks SDK for workspace management and Vector Search
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

# MLflow for experiment tracking and model serving
import mlflow
from mlflow.tracking import MlflowClient

# LangChain components for RAG pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

# Data analysis libraries for performance profiling
import pandas as pd
import numpy as np

print("✅ All libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Workspace Settings
# MAGIC
# MAGIC This cell defines all the configuration parameters for the lab. These settings control:
# MAGIC
# MAGIC **Unity Catalog Settings:**
# MAGIC - `CATALOG_NAME` - The Unity Catalog catalog where tables will be created
# MAGIC - `SCHEMA_NAME` - The schema (database) within the catalog for organizing our tables
# MAGIC
# MAGIC **Table Paths:**
# MAGIC - `SOURCE_TABLE_PATH` - Delta table storing the original source documents
# MAGIC - `CHUNKS_TABLE_PATH` - Delta table storing chunked documents with embeddings
# MAGIC
# MAGIC **Vector Search Settings:**
# MAGIC - `VECTOR_SEARCH_ENDPOINT_NAME` - The compute endpoint that hosts Vector Search indices
# MAGIC - `VECTOR_INDEX_PATH` - Full path to the Vector Search index
# MAGIC
# MAGIC **Model Settings:**
# MAGIC - `EMBEDDING_MODEL_NAME` - The Databricks-hosted embedding model (BGE-large produces 1024-dim vectors)
# MAGIC - `LLM_MODEL_NAME` - The Databricks-hosted LLM for response generation
# MAGIC
# MAGIC **Tuning Parameters (Chapter 9 Focus):**
# MAGIC - `CHUNK_SIZE` - Maximum characters per chunk (affects context granularity)
# MAGIC - `CHUNK_OVERLAP` - Characters shared between adjacent chunks (maintains context continuity)
# MAGIC - `EMBEDDING_DIMENSION` - Vector size (must match the embedding model output)
# MAGIC - `TOP_K_RESULTS` - Number of similar documents to retrieve
# MAGIC - `BATCH_SIZE` - Documents processed together (affects throughput vs. memory)
# MAGIC
# MAGIC **⚠️ Important:** Update `CATALOG_NAME` and `VECTOR_SEARCH_ENDPOINT_NAME` to match your workspace.

# COMMAND ----------

# ============================================================
# CONFIGURATION - Update these values for your environment
# ============================================================

# Unity Catalog settings
CATALOG_NAME = "main"  # Your Unity Catalog name
SCHEMA_NAME = "rag_lab"  # Schema for this lab

# Table names
SOURCE_TABLE_NAME = "technical_documents"
CHUNKS_TABLE_NAME = "document_chunks"

# Vector Search settings
VECTOR_SEARCH_ENDPOINT_NAME = "rag_lab_endpoint"  # Your VS endpoint name
VECTOR_INDEX_NAME = "document_chunks_index"

# Model Serving settings
EMBEDDING_MODEL_NAME = "databricks-bge-large-en"  # Databricks hosted embedding model
# Available LLM endpoints in this workspace (discovered via endpoint listing):
# - databricks-meta-llama-3-3-70b-instruct (recommended)
# - databricks-meta-llama-3-1-405b-instruct
# - databricks-claude-sonnet-4
# - databricks-gemma-3-12b
LLM_MODEL_NAME = "databricks-meta-llama-3-3-70b-instruct"  # Llama 3.3 70B

# Performance tuning parameters
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
EMBEDDING_DIMENSION = 1024  # BGE-large embedding dimension
TOP_K_RESULTS = 5  # Number of results to retrieve
BATCH_SIZE = 32  # Batch size for embedding generation

# Full table paths
SOURCE_TABLE_PATH = f"{CATALOG_NAME}.{SCHEMA_NAME}.{SOURCE_TABLE_NAME}"
CHUNKS_TABLE_PATH = f"{CATALOG_NAME}.{SCHEMA_NAME}.{CHUNKS_TABLE_NAME}"
VECTOR_INDEX_PATH = f"{CATALOG_NAME}.{SCHEMA_NAME}.{VECTOR_INDEX_NAME}"

print(f"📁 Catalog: {CATALOG_NAME}")
print(f"📁 Schema: {SCHEMA_NAME}")
print(f"📄 Source Table: {SOURCE_TABLE_PATH}")
print(f"📄 Chunks Table: {CHUNKS_TABLE_PATH}")
print(f"🔍 Vector Index: {VECTOR_INDEX_PATH}")
print(f"🤖 Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"🤖 LLM Model: {LLM_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Clients and Create Schema
# MAGIC
# MAGIC This cell performs three critical setup tasks:
# MAGIC
# MAGIC **1. Initialize SparkSession:**
# MAGIC - `SparkSession` is the entry point for all Spark functionality
# MAGIC - In Databricks, a session is pre-configured, so we use `getOrCreate()` to access it
# MAGIC - This enables distributed data processing for our document pipeline
# MAGIC
# MAGIC **2. Initialize Databricks Clients:**
# MAGIC - `WorkspaceClient` - Provides access to workspace-level APIs (endpoints, jobs, clusters)
# MAGIC - `VectorSearchClient` - Specialized client for creating and managing Vector Search indices
# MAGIC - `MlflowClient` - Programmatic access to MLflow for experiment tracking
# MAGIC
# MAGIC **3. Create Unity Catalog Schema:**
# MAGIC - Creates the catalog if it doesn't exist (requires appropriate permissions)
# MAGIC - Creates the schema within the catalog for organizing our tables
# MAGIC - Sets the current catalog and schema context for subsequent SQL operations
# MAGIC
# MAGIC **Why Unity Catalog?**
# MAGIC Unity Catalog provides centralized governance, fine-grained access control, and data lineage tracking - essential for production RAG systems.

# COMMAND ----------

# ============================================================
# INITIALIZE CLIENTS AND CREATE SCHEMA
# ============================================================

# Get or create SparkSession (pre-configured in Databricks environment)
spark = SparkSession.builder.getOrCreate()

# Initialize Databricks workspace client for API access
workspace_client = WorkspaceClient()

# Initialize Vector Search client for index management
# disable_notice=True suppresses authentication method warnings in notebook output
vector_search_client = VectorSearchClient(disable_notice=True)

# Initialize MLflow client for experiment tracking
mlflow_client = MlflowClient()

# Create Unity Catalog and Schema if they don't exist
# Note: You need CREATE CATALOG permission for the first command
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

# Set the current catalog and schema context
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print(f"✅ Schema '{CATALOG_NAME}.{SCHEMA_NAME}' is ready!")
print(f"✅ All clients initialized successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Discover Available Model Serving Endpoints
# MAGIC
# MAGIC Before using LLM endpoints, we need to verify which Foundation Models are available in your workspace. This is important because:
# MAGIC
# MAGIC **Why Discovery is Necessary:**
# MAGIC - Different Databricks regions have different model availability
# MAGIC - Azure Databricks has different models than AWS Databricks
# MAGIC - Some models may be disabled by workspace administrators
# MAGIC - Custom endpoints (fine-tuned models) may also be available
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Uses the `WorkspaceClient` to list all serving endpoints
# MAGIC 2. Categorizes them into Foundation Models (Databricks-hosted) and Custom Models
# MAGIC 3. Shows the ready status of each endpoint
# MAGIC
# MAGIC **After Running:**
# MAGIC - Identify an LLM endpoint for response generation (e.g., `databricks-meta-llama-3-3-70b-instruct`)
# MAGIC - Verify the embedding model is available (e.g., `databricks-bge-large-en`)
# MAGIC - Update `LLM_MODEL_NAME` in the configuration cell if needed

# COMMAND ----------

# ============================================================
# DISCOVER AVAILABLE MODEL SERVING ENDPOINTS
# ============================================================

# Re-import WorkspaceClient (in case Python was restarted)
from databricks.sdk import WorkspaceClient

# Create workspace client instance
w = WorkspaceClient()

print("📋 Available Model Serving Endpoints:\n")
print("-" * 60)

# Separate foundation models from custom endpoints
foundation_models = []
custom_models = []

# Iterate through all serving endpoints in the workspace
for endpoint in w.serving_endpoints.list():
    endpoint_name = endpoint.name
    # Check if endpoint is ready to serve requests
    state = endpoint.state.ready if endpoint.state else "UNKNOWN"

    # Categorize based on naming convention
    if endpoint_name.startswith("databricks-"):
        foundation_models.append((endpoint_name, state))
    else:
        custom_models.append((endpoint_name, state))

# Display Foundation Model endpoints
print("🏢 Foundation Model Endpoints (Databricks-hosted):")
if foundation_models:
    for name, state in sorted(foundation_models):
        status_icon = "✅" if state else "⏳"
        print(f"   {status_icon} {name}")
else:
    print("   ⚠️  No Foundation Model endpoints found")

# Display Custom Model endpoints
print("\n🔧 Custom Model Endpoints:")
if custom_models:
    for name, state in sorted(custom_models):
        status_icon = "✅" if state else "⏳"
        print(f"   {status_icon} {name}")
else:
    print("   (none)")

print("\n" + "-" * 60)
print("💡 Update LLM_MODEL_NAME in the configuration cell above")
print("   to use one of the available Foundation Model endpoints.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 2: Sample Data Generation (Prerequisite)
# MAGIC
# MAGIC Before building our RAG system, we need realistic sample data that simulates enterprise content. This section creates a diverse document corpus that will be used throughout the lab.
# MAGIC
# MAGIC ### Why Sample Data Matters
# MAGIC
# MAGIC The quality and diversity of your document corpus directly impacts RAG system performance:
# MAGIC - **Diverse content types** test the embedding model's ability to capture different semantic patterns
# MAGIC - **Varying document lengths** help tune chunking parameters
# MAGIC - **Multiple categories** enable testing of filtered retrieval
# MAGIC - **Realistic content** ensures the lab reflects production scenarios
# MAGIC
# MAGIC ### Document Categories
# MAGIC
# MAGIC We generate three types of documents commonly found in enterprise knowledge bases:
# MAGIC
# MAGIC | Category | Count | Purpose | Example Queries |
# MAGIC |----------|-------|---------|-----------------|
# MAGIC | **Technical Documents** | 4 | API docs, architecture guides, troubleshooting | "How do I authenticate?", "What is the data pipeline?" |
# MAGIC | **Internal Wikis** | 3 | Process docs, team guidelines, policies | "What are code review practices?", "Who is on-call?" |
# MAGIC | **Operational Reports** | 4 | Incident reports, performance summaries | "What was the outage impact?", "What is Q4 uptime?" |
# MAGIC
# MAGIC ### Document Schema
# MAGIC
# MAGIC Each document contains:
# MAGIC - `doc_id` - Unique identifier for tracking and retrieval
# MAGIC - `title` - Human-readable document title
# MAGIC - `category` - Classification for filtered search
# MAGIC - `content` - The actual document text (will be chunked)
# MAGIC - `author` - Document owner for governance
# MAGIC - `last_updated` - Timestamp for freshness tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Technical Documents
# MAGIC
# MAGIC Technical documents include API references, architecture guides, and troubleshooting manuals. These documents typically contain:
# MAGIC - Structured information (endpoints, parameters, error codes)
# MAGIC - Step-by-step procedures
# MAGIC - Technical terminology that requires precise semantic matching
# MAGIC
# MAGIC The embedding model must capture both the technical vocabulary and the procedural nature of these documents.

# COMMAND ----------

# ============================================================
# SAMPLE DATA GENERATION - Technical Documents
# ============================================================

technical_documents = [
    {
        "doc_id": "TECH-001",
        "title": "User Authentication API Reference",
        "category": "technical",
        "content": """The User Authentication API provides secure endpoints for user login, registration, and session management.

Authentication Flow:
1. Client sends POST request to /api/v2/auth/login with credentials
2. Server validates credentials against the identity provider
3. Upon success, server returns JWT token with 24-hour expiration
4. Client includes token in Authorization header for subsequent requests

Endpoints:
- POST /api/v2/auth/login: Authenticate user and receive JWT token
- POST /api/v2/auth/register: Create new user account
- POST /api/v2/auth/refresh: Refresh expired JWT token
- POST /api/v2/auth/logout: Invalidate current session
- GET /api/v2/auth/profile: Retrieve authenticated user profile

Error Codes:
- 401: Invalid credentials or expired token
- 403: Insufficient permissions
- 429: Rate limit exceeded (max 100 requests/minute)

Security Considerations:
All authentication endpoints require HTTPS. Tokens are signed using RS256 algorithm. Failed login attempts are logged and may trigger account lockout after 5 consecutive failures.""",
        "author": "Platform Team",
        "last_updated": "2024-01-15"
    },
    {
        "doc_id": "TECH-002",
        "title": "Data Pipeline Architecture Guide",
        "category": "technical",
        "content": """This document describes the enterprise data pipeline architecture used for processing and transforming data across our analytics platform.

Architecture Overview:
The pipeline follows a medallion architecture pattern with Bronze, Silver, and Gold layers:

Bronze Layer (Raw Data):
- Ingests data from 50+ source systems
- Stores data in original format with minimal transformation
- Retention period: 90 days
- Storage format: Delta Lake with Z-ordering on timestamp

Silver Layer (Cleaned Data):
- Applies data quality rules and standardization
- Deduplication using composite keys
- Schema enforcement and type casting
- Incremental processing with watermarking

Gold Layer (Business-Ready):
- Aggregated metrics and KPIs
- Dimensional models for reporting
- Optimized for query performance
- Materialized views for common queries

Performance Metrics:
- Daily data volume: 2.5 TB
- Average latency: 15 minutes end-to-end
- SLA: 99.5% availability""",
        "author": "Data Engineering Team",
        "last_updated": "2024-02-20"
    },
    {
        "doc_id": "TECH-003",
        "title": "Kubernetes Deployment Troubleshooting",
        "category": "technical",
        "content": """Troubleshooting guide for common Kubernetes deployment issues in our production environment.

Common Issues and Solutions:

1. Pod CrashLoopBackOff:
   - Check logs: kubectl logs <pod-name> --previous
   - Verify resource limits are not too restrictive
   - Check liveness/readiness probe configurations
   - Ensure environment variables are correctly set

2. ImagePullBackOff:
   - Verify image name and tag exist in registry
   - Check imagePullSecrets are configured
   - Ensure network connectivity to container registry

3. Pending Pods:
   - Check node resources: kubectl describe nodes
   - Verify PersistentVolumeClaims are bound
   - Check node selectors and tolerations

4. Service Not Accessible:
   - Verify selector labels match pod labels
   - Check endpoint status: kubectl get endpoints
   - Validate network policies allow traffic

Debugging Commands:
- kubectl describe pod <pod-name>
- kubectl get events --sort-by='.lastTimestamp'
- kubectl exec -it <pod-name> -- /bin/sh""",
        "author": "DevOps Team",
        "last_updated": "2024-03-10"
    },
    {
        "doc_id": "TECH-004",
        "title": "Machine Learning Model Deployment Guide",
        "category": "technical",
        "content": """This guide covers the end-to-end process for deploying machine learning models to production using MLflow and Databricks Model Serving.

Model Registration:
1. Train your model using any ML framework (scikit-learn, PyTorch, TensorFlow)
2. Log the model to MLflow with appropriate signature
3. Register the model in Unity Catalog
4. Add model version description and tags

Deployment Options:
- Real-time serving: Low-latency predictions via REST API
- Batch inference: Process large datasets using Spark
- Streaming inference: Real-time predictions on streaming data

Model Serving Configuration:
- Compute size: Small (4 CPU, 16GB RAM) to Large (16 CPU, 64GB RAM)
- Scale to zero: Enable for cost optimization
- Auto-scaling: Configure min/max replicas based on traffic

Monitoring:
- Track prediction latency and throughput
- Monitor model drift using statistical tests
- Set up alerts for performance degradation
- Log predictions for audit and debugging""",
        "author": "ML Platform Team",
        "last_updated": "2024-03-25"
    }
]

print(f"✅ Created {len(technical_documents)} technical documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Internal Wiki Documents
# MAGIC
# MAGIC These documents represent internal knowledge base articles covering processes, guidelines, and best practices used across the organization.

# COMMAND ----------

# ============================================================
# SAMPLE DATA GENERATION - Internal Wiki Documents
# ============================================================

wiki_documents = [
    {
        "doc_id": "WIKI-001",
        "title": "Code Review Best Practices",
        "category": "wiki",
        "content": """This wiki outlines the code review process and best practices for our engineering teams.

Code Review Checklist:
□ Code follows team style guidelines
□ Unit tests cover new functionality
□ No hardcoded secrets or credentials
□ Error handling is comprehensive
□ Documentation is updated
□ Performance implications considered

Review Process:
1. Author creates pull request with clear description
2. Automated checks run (linting, tests, security scan)
3. At least 2 reviewers must approve
4. Address all comments before merging
5. Squash commits and merge to main branch

Response Time Expectations:
- Initial review: Within 24 hours
- Follow-up reviews: Within 4 hours
- Urgent fixes: Within 2 hours

Common Feedback Categories:
- Logic errors or edge cases
- Performance concerns
- Security vulnerabilities
- Code readability improvements
- Missing test coverage""",
        "author": "Engineering Standards Team",
        "last_updated": "2024-02-01"
    },
    {
        "doc_id": "WIKI-002",
        "title": "On-Call Rotation Guidelines",
        "category": "wiki",
        "content": """Guidelines for engineers participating in the on-call rotation for production systems.

On-Call Responsibilities:
- Monitor alerting channels during shift
- Acknowledge alerts within 15 minutes
- Escalate issues that cannot be resolved within 1 hour
- Document all incidents in the incident tracker
- Participate in post-incident reviews

Shift Schedule:
- Primary on-call: 7 days, 24/7 coverage
- Secondary on-call: Backup for escalations
- Handoff meeting: Every Monday at 10 AM

Escalation Path:
1. Primary on-call engineer
2. Secondary on-call engineer
3. Team lead
4. Engineering manager
5. VP of Engineering (critical incidents only)

Tools and Access:
- PagerDuty for alerting
- Slack #incidents channel
- Runbook repository in Confluence
- VPN access for remote debugging""",
        "author": "SRE Team",
        "last_updated": "2024-01-20"
    },
    {
        "doc_id": "WIKI-003",
        "title": "Data Governance Policies",
        "category": "wiki",
        "content": """This document outlines data governance policies for handling sensitive information.

Data Classification Levels:
1. Public: No restrictions on access or sharing
2. Internal: Available to all employees
3. Confidential: Restricted to specific teams
4. Restricted: Highly sensitive, need-to-know basis

PII Handling Requirements:
- Encrypt PII at rest and in transit
- Mask PII in non-production environments
- Log all access to PII data
- Retain PII only as long as necessary
- Obtain consent before collecting PII

Data Retention Policies:
- Transaction data: 7 years
- User activity logs: 90 days
- Analytics data: 2 years
- Backup data: 30 days after deletion

Compliance Requirements:
- GDPR for EU customers
- CCPA for California residents
- SOC 2 Type II certification
- Annual security audits""",
        "author": "Compliance Team",
        "last_updated": "2024-03-01"
    }
]

print(f"✅ Created {len(wiki_documents)} wiki documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Operational Reports
# MAGIC
# MAGIC Operational reports represent time-sensitive, factual content that analysts frequently query. These include:
# MAGIC - **Incident Reports** - Post-mortems with root cause analysis and impact metrics
# MAGIC - **Performance Summaries** - Quarterly metrics, SLA compliance, system health
# MAGIC - **Status Updates** - Weekly progress reports, project milestones
# MAGIC
# MAGIC **Key Characteristics:**
# MAGIC - Contains specific numbers, dates, and metrics
# MAGIC - Often queried with time-based filters ("Q4 performance", "last incident")
# MAGIC - Requires precise retrieval to avoid mixing data from different time periods
# MAGIC
# MAGIC **RAG Challenge:** The system must distinguish between similar reports from different time periods and return the most relevant one based on the query context.

# COMMAND ----------

# ============================================================
# SAMPLE DATA GENERATION - Operational Reports
# ============================================================

operational_reports = [
    {
        "doc_id": "OPS-001",
        "title": "Q4 2024 System Performance Report",
        "category": "operational",
        "content": """Quarterly performance report for production systems covering October-December 2024.

Executive Summary:
Overall system availability exceeded targets with 99.97% uptime. Response times improved by 15% compared to Q3 due to infrastructure optimizations.

Key Metrics:
- Uptime: 99.97% (target: 99.9%)
- Average response time: 145ms (target: 200ms)
- Error rate: 0.02% (target: 0.1%)
- Peak concurrent users: 125,000
- Total API calls: 2.3 billion

Infrastructure Changes:
- Migrated 40% of workloads to new Kubernetes cluster
- Upgraded database instances to latest generation
- Implemented CDN for static assets
- Added 3 new edge locations

Incidents:
- Total incidents: 12
- P1 incidents: 1 (database failover)
- P2 incidents: 4
- Mean time to resolution: 23 minutes

Recommendations:
1. Increase database connection pool size
2. Implement request queuing for traffic spikes
3. Add circuit breakers for external dependencies""",
        "author": "Platform Operations",
        "last_updated": "2024-01-05"
    },
    {
        "doc_id": "OPS-002",
        "title": "Incident Report: Payment Processing Outage",
        "category": "operational",
        "content": """Incident Report - Payment Processing Service Outage
Date: March 15, 2024
Duration: 47 minutes
Severity: P1

Impact:
- Payment processing unavailable for 47 minutes
- Approximately 12,000 transactions affected
- Estimated revenue impact: $450,000
- Customer complaints: 234

Timeline:
14:23 - Monitoring alerts triggered for payment service errors
14:25 - On-call engineer acknowledged alert
14:32 - Root cause identified: database connection exhaustion
14:45 - Temporary fix applied: increased connection pool
15:10 - Service fully restored
15:30 - All queued transactions processed

Root Cause:
A deployment at 14:15 introduced a connection leak in the payment service. Each request opened a new database connection without properly closing it, exhausting the connection pool within 8 minutes.

Corrective Actions:
1. Reverted problematic deployment
2. Added connection pool monitoring
3. Implemented connection timeout settings
4. Updated deployment checklist to include connection testing
5. Scheduled post-incident review for March 18""",
        "author": "Incident Response Team",
        "last_updated": "2024-03-16"
    },
    {
        "doc_id": "OPS-003",
        "title": "Weekly Infrastructure Status Update",
        "category": "operational",
        "content": """Weekly Infrastructure Status Report - Week of March 18, 2024

Overall Status: GREEN ✅

Compute Resources:
- CPU utilization: 45% average (healthy)
- Memory utilization: 62% average (healthy)
- Disk I/O: Normal levels
- Network throughput: 2.3 Gbps average

Database Health:
- Primary cluster: Healthy
- Read replicas: 3/3 healthy
- Replication lag: <100ms
- Storage utilization: 72%

Kubernetes Clusters:
- Production: 48/50 nodes healthy
- Staging: 12/12 nodes healthy
- 2 nodes in production under maintenance

Upcoming Maintenance:
- March 22: Database version upgrade (2 AM - 4 AM)
- March 25: Network switch replacement (minimal impact)
- March 28: SSL certificate renewal

Action Items:
- Investigate memory growth in analytics service
- Plan capacity increase for Q2 traffic projections
- Review and update disaster recovery runbooks""",
        "author": "Infrastructure Team",
        "last_updated": "2024-03-18"
    },
    {
        "doc_id": "OPS-004",
        "title": "Cost Optimization Analysis Report",
        "category": "operational",
        "content": """Monthly Cloud Cost Optimization Report - February 2024

Total Cloud Spend: $847,000 (5% under budget)

Cost Breakdown by Service:
- Compute (EC2/VMs): $412,000 (49%)
- Storage (S3/Blob): $178,000 (21%)
- Database (RDS/SQL): $156,000 (18%)
- Networking: $67,000 (8%)
- Other services: $34,000 (4%)

Optimization Achievements:
1. Reserved instance coverage increased to 72%
2. Implemented auto-scaling for dev environments
3. Archived 15TB of cold data to glacier storage
4. Consolidated 12 underutilized instances

Savings Realized: $123,000 (13% reduction from baseline)

Recommendations for Next Month:
1. Migrate remaining workloads to spot instances where applicable
2. Implement S3 intelligent tiering for analytics data
3. Right-size database instances based on usage patterns
4. Enable auto-shutdown for non-production environments

Cost Anomalies Detected:
- Unusual spike in data transfer costs on Feb 15
- Investigation revealed: Large dataset export for compliance audit""",
        "author": "FinOps Team",
        "last_updated": "2024-03-05"
    }
]

print(f"✅ Created {len(operational_reports)} operational reports")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine and Save Documents to Delta Table
# MAGIC
# MAGIC This cell performs the following operations:
# MAGIC
# MAGIC **1. Combine All Documents:**
# MAGIC - Merges technical documents, wiki documents, and operational reports into a single list
# MAGIC - This unified corpus will be processed through the embedding pipeline
# MAGIC
# MAGIC **2. Create Spark DataFrame:**
# MAGIC - Converts the Python list of dictionaries to a distributed Spark DataFrame
# MAGIC - Enables parallel processing for large document collections
# MAGIC
# MAGIC **3. Add Metadata Columns:**
# MAGIC - `created_at` - Timestamp for tracking when documents were ingested
# MAGIC - `embedding_status` - Status flag for tracking embedding generation progress
# MAGIC
# MAGIC **4. Save as Delta Table:**
# MAGIC - Writes to Unity Catalog with `overwrite` mode for idempotent execution
# MAGIC - `overwriteSchema` allows schema evolution if document structure changes
# MAGIC - Delta format enables ACID transactions and time travel
# MAGIC
# MAGIC **Why Delta Lake?**
# MAGIC - Vector Search requires Delta tables with Change Data Feed enabled
# MAGIC - Delta provides automatic versioning and rollback capabilities
# MAGIC - Optimized for both batch and streaming updates

# COMMAND ----------

# ============================================================
# COMBINE ALL DOCUMENTS AND SAVE TO DELTA TABLE
# ============================================================

# Combine all document types into a single corpus
all_documents = technical_documents + wiki_documents + operational_reports

# Convert to Spark DataFrame for distributed processing
documents_df = spark.createDataFrame(all_documents)

# Add metadata columns for tracking and governance
documents_df = documents_df.withColumn("created_at", current_timestamp()) \
                           .withColumn("embedding_status", lit("pending"))

# Display document statistics
print(f"📊 Total documents: {documents_df.count()}")
print(f"\n📋 Document categories:")
documents_df.groupBy("category").count().show()

# Save to Delta table in Unity Catalog
# Using overwrite mode for idempotent execution (safe to re-run)
documents_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(SOURCE_TABLE_PATH)

print(f"\n✅ Documents saved to {SOURCE_TABLE_PATH}")

# Display the saved table
display(spark.table(SOURCE_TABLE_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 3: Building the Embedding Pipeline
# MAGIC
# MAGIC The embedding pipeline is the foundation of any RAG system. It transforms text documents into numerical vectors that capture semantic meaning, enabling similarity-based retrieval.
# MAGIC
# MAGIC ### Pipeline Overview
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
# MAGIC │   Documents  │───▶│   Chunking   │───▶│  Embedding   │───▶│  Delta Table │
# MAGIC │   (Source)   │    │  (Splitting) │    │  (Vectors)   │    │  (Storage)   │
# MAGIC └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Why Chunking Matters
# MAGIC
# MAGIC Documents are typically too long to embed as a single unit:
# MAGIC - **Token limits**: Embedding models have maximum input lengths (typically 512-8192 tokens)
# MAGIC - **Retrieval precision**: Smaller chunks enable more precise matching
# MAGIC - **Context relevance**: Large chunks may contain irrelevant information that dilutes the embedding
# MAGIC
# MAGIC ### Key Decisions in This Section
# MAGIC
# MAGIC | Decision | Our Choice | Rationale |
# MAGIC |----------|------------|-----------|
# MAGIC | Chunk size | 1000 chars | Balances context richness with retrieval precision |
# MAGIC | Chunk overlap | 200 chars | Prevents losing context at chunk boundaries |
# MAGIC | Embedding model | BGE-large-en | High-quality, 1024-dim vectors, optimized for English |
# MAGIC | Batch size | 32 | Optimizes throughput while managing memory |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Document Chunking Strategy
# MAGIC
# MAGIC We use `RecursiveCharacterTextSplitter` from LangChain, which is the recommended splitter for most use cases.
# MAGIC
# MAGIC **How RecursiveCharacterTextSplitter Works:**
# MAGIC 1. Attempts to split on the first separator (`\n\n` - paragraph breaks)
# MAGIC 2. If chunks are still too large, tries the next separator (`\n` - line breaks)
# MAGIC 3. Continues through separators (`. `, ` `, `""`) until chunks fit within `chunk_size`
# MAGIC 4. Adds `chunk_overlap` characters from the previous chunk to maintain context
# MAGIC
# MAGIC **Separator Hierarchy:**
# MAGIC ```python
# MAGIC separators=["\n\n", "\n", ". ", " ", ""]
# MAGIC #           ↑       ↑     ↑    ↑    ↑
# MAGIC #      Paragraphs Lines Sentences Words Characters
# MAGIC ```
# MAGIC
# MAGIC **Metadata Preservation:**
# MAGIC Each chunk retains metadata from the parent document:
# MAGIC - `chunk_id` - Unique identifier for the chunk
# MAGIC - `doc_id` - Reference to the source document
# MAGIC - `chunk_index` - Position within the document (useful for context reconstruction)
# MAGIC - `total_chunks` - Total chunks from this document
# MAGIC - `char_count` - Character count for analysis
# MAGIC
# MAGIC **Chapter 9 Tuning Insight:** Chunk size significantly impacts retrieval quality. Too small = fragmented context. Too large = diluted relevance. Start with 1000 chars and adjust based on your content type.

# COMMAND ----------

# ============================================================
# DOCUMENT CHUNKING
# ============================================================

# Initialize the text splitter with our tuning parameters
# RecursiveCharacterTextSplitter tries to split on natural boundaries
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,           # Maximum characters per chunk
    chunk_overlap=CHUNK_OVERLAP,     # Characters shared between adjacent chunks
    length_function=len,             # Use character count (not token count)
    separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
)

def chunk_document(doc_id: str, title: str, content: str, category: str) -> List[Dict]:
    """
    Split a document into chunks while preserving metadata.

    Args:
        doc_id: Unique document identifier
        title: Document title (preserved in each chunk)
        content: Full document text to be chunked
        category: Document category for filtered retrieval

    Returns:
        List of chunk dictionaries with metadata
    """
    # Split the content into chunks
    chunks = text_splitter.split_text(content)

    # Create chunk records with metadata
    return [
        {
            "chunk_id": f"{doc_id}_chunk_{i}",  # Unique chunk identifier
            "doc_id": doc_id,                   # Parent document reference
            "title": title,                     # Preserved for display
            "category": category,               # For filtered retrieval
            "chunk_index": i,                   # Position in document
            "total_chunks": len(chunks),        # Total chunks from this doc
            "content": chunk,                   # The actual chunk text
            "char_count": len(chunk)            # For analysis
        }
        for i, chunk in enumerate(chunks)
    ]

# Load documents from Delta table
documents = spark.table(SOURCE_TABLE_PATH).collect()

# Process each document through the chunking pipeline
all_chunks = []
for doc in documents:
    chunks = chunk_document(
        doc_id=doc["doc_id"],
        title=doc["title"],
        content=doc["content"],
        category=doc["category"]
    )
    all_chunks.extend(chunks)

# Display chunking statistics
print(f"📄 Original documents: {len(documents)}")
print(f"📦 Total chunks created: {len(all_chunks)}")
print(f"📊 Average chunks per document: {len(all_chunks) / len(documents):.1f}")

# Analyze chunk size distribution (important for tuning)
chunk_sizes = [c["char_count"] for c in all_chunks]
print(f"\n📏 Chunk size statistics:")
print(f"   Min: {min(chunk_sizes)} chars")
print(f"   Max: {max(chunk_sizes)} chars")
print(f"   Avg: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Embeddings Using Databricks Foundation Model
# MAGIC
# MAGIC Embeddings are dense vector representations that capture the semantic meaning of text. We use the Databricks-hosted `databricks-bge-large-en` model.
# MAGIC
# MAGIC **About BGE-large-en:**
# MAGIC | Property | Value | Implication |
# MAGIC |----------|-------|-------------|
# MAGIC | Dimensions | 1024 | Higher dimensionality = more semantic nuance |
# MAGIC | Max tokens | 512 | Longer chunks may be truncated |
# MAGIC | Language | English | Optimized for English text |
# MAGIC | Architecture | BERT-based | Bidirectional context understanding |
# MAGIC
# MAGIC **Why Batching Matters (Chapter 9 Pattern):**
# MAGIC
# MAGIC Embedding generation is often a bottleneck in RAG pipelines. Batching provides:
# MAGIC 1. **Reduced API overhead** - Fewer HTTP requests
# MAGIC 2. **Better GPU utilization** - Models process batches more efficiently
# MAGIC 3. **Predictable throughput** - Easier to estimate processing time
# MAGIC
# MAGIC **Batch Size Trade-offs:**
# MAGIC - **Too small (1-8)**: High API overhead, underutilized compute
# MAGIC - **Too large (128+)**: Memory pressure, potential timeouts
# MAGIC - **Optimal (16-64)**: Balances throughput and reliability
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Initializes the `DatabricksEmbeddings` wrapper for the BGE model
# MAGIC 2. Defines a batched embedding function with progress tracking
# MAGIC 3. Processes all chunks and measures throughput
# MAGIC 4. Reports embedding statistics for validation

# COMMAND ----------

# ============================================================
# EMBEDDING GENERATION
# ============================================================

# Initialize Databricks embedding model
# DatabricksEmbeddings automatically handles authentication within the workspace
embedding_model = DatabricksEmbeddings(
    endpoint=EMBEDDING_MODEL_NAME,
    # No API key needed when running in Databricks workspace
)

def generate_embeddings_batch(texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """
    Generate embeddings in batches for better throughput.

    Batching reduces API overhead and improves GPU utilization on the
    serving endpoint. Progress is logged every 5 batches.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process per API call

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"🔄 Generating embeddings for {len(texts)} chunks in {total_batches} batches...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        # Time each batch for throughput analysis
        start_time = time.time()
        embeddings = embedding_model.embed_documents(batch)
        elapsed = time.time() - start_time

        all_embeddings.extend(embeddings)

        # Log progress periodically
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"   Batch {batch_num}/{total_batches} completed ({elapsed:.2f}s)")

    return all_embeddings

# Extract text content from all chunks
chunk_texts = [chunk["content"] for chunk in all_chunks]

# Generate embeddings with timing
start_time = time.time()
embeddings = generate_embeddings_batch(chunk_texts)
total_time = time.time() - start_time

# Report embedding statistics
print(f"\n✅ Embedding generation complete!")
print(f"⏱️  Total time: {total_time:.2f} seconds")
print(f"📊 Throughput: {len(chunk_texts) / total_time:.1f} chunks/second")
print(f"📐 Embedding dimension: {len(embeddings[0])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Chunks with Embeddings to Delta Table
# MAGIC
# MAGIC This cell persists the chunks and their embeddings to a Delta table in Unity Catalog.
# MAGIC
# MAGIC **Schema Design:**
# MAGIC | Column | Type | Purpose |
# MAGIC |--------|------|---------|
# MAGIC | `chunk_id` | STRING | Primary key for the Vector Search index |
# MAGIC | `doc_id` | STRING | Foreign key to source document |
# MAGIC | `title` | STRING | Displayed in search results |
# MAGIC | `category` | STRING | Enables filtered retrieval |
# MAGIC | `chunk_index` | INTEGER | Position within document |
# MAGIC | `content` | STRING | Text content for LLM context |
# MAGIC | `embedding` | ARRAY<FLOAT> | 1024-dim vector for similarity search |
# MAGIC | `indexed_at` | TIMESTAMP | Audit trail for data freshness |
# MAGIC
# MAGIC **Critical Configuration - Change Data Feed:**
# MAGIC
# MAGIC The `delta.enableChangeDataFeed = true` property is **required** for Delta Sync indices. It enables:
# MAGIC - Automatic detection of inserts, updates, and deletes
# MAGIC - Incremental synchronization (only changed rows are re-indexed)
# MAGIC - Near real-time index updates without full rebuilds
# MAGIC
# MAGIC **Why Delta Lake for Vector Storage?**
# MAGIC 1. **ACID transactions** - Consistent reads during updates
# MAGIC 2. **Time travel** - Rollback to previous versions if needed
# MAGIC 3. **Schema evolution** - Add columns without breaking the index
# MAGIC 4. **Unified governance** - Same access controls as other data

# COMMAND ----------

# ============================================================
# SAVE CHUNKS WITH EMBEDDINGS TO DELTA TABLE
# ============================================================

# Attach embeddings to chunk records
for i, chunk in enumerate(all_chunks):
    chunk["embedding"] = embeddings[i]

# Define explicit schema for the chunks table
# This ensures correct types, especially for the embedding array
chunks_schema = StructType([
    StructField("chunk_id", StringType(), False),      # Primary key (NOT NULL)
    StructField("doc_id", StringType(), False),        # Foreign key (NOT NULL)
    StructField("title", StringType(), True),          # Display field
    StructField("category", StringType(), True),       # Filter field
    StructField("chunk_index", IntegerType(), True),   # Position in document
    StructField("total_chunks", IntegerType(), True),  # Total chunks from doc
    StructField("content", StringType(), True),        # Text for LLM context
    StructField("char_count", IntegerType(), True),    # For analysis
    StructField("embedding", ArrayType(FloatType()), True)  # Vector for search
])

# Create DataFrame with explicit schema
chunks_df = spark.createDataFrame(all_chunks, schema=chunks_schema)

# Add indexing timestamp for audit trail
chunks_df = chunks_df.withColumn("indexed_at", current_timestamp())

# Write to Delta table in Unity Catalog
chunks_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(CHUNKS_TABLE_PATH)

# CRITICAL: Enable Change Data Feed for Vector Search synchronization
# Without this, Delta Sync indices cannot detect changes
spark.sql(f"ALTER TABLE {CHUNKS_TABLE_PATH} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"✅ Chunks with embeddings saved to {CHUNKS_TABLE_PATH}")
print(f"📊 Total rows: {spark.table(CHUNKS_TABLE_PATH).count()}")

# Display sample (excluding embedding column for readability)
display(spark.table(CHUNKS_TABLE_PATH).select("chunk_id", "doc_id", "title", "category", "char_count").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 4: Creating the Vector Search Index
# MAGIC
# MAGIC Vector Search is the core retrieval component of our RAG system. It enables fast similarity search over millions of vectors.
# MAGIC
# MAGIC ### Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                    Vector Search Endpoint                        │
# MAGIC │  ┌─────────────────────────────────────────────────────────┐    │
# MAGIC │  │                    Delta Sync Index                      │    │
# MAGIC │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │    │
# MAGIC │  │  │   Vectors   │    │  Metadata   │    │   Content   │  │    │
# MAGIC │  │  │  (1024-dim) │    │ (category)  │    │   (text)    │  │    │
# MAGIC │  │  └─────────────┘    └─────────────┘    └─────────────┘  │    │
# MAGIC │  └─────────────────────────────────────────────────────────┘    │
# MAGIC │                              ▲                                   │
# MAGIC │                              │ Auto-sync                         │
# MAGIC │                    ┌─────────┴─────────┐                        │
# MAGIC │                    │   Delta Table     │                        │
# MAGIC │                    │ (Change Data Feed)│                        │
# MAGIC │                    └───────────────────┘                        │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC | Concept | Description |
# MAGIC |---------|-------------|
# MAGIC | **Endpoint** | Managed compute resource that hosts indices (like a server) |
# MAGIC | **Delta Sync Index** | Index that auto-syncs with a Delta table |
# MAGIC | **Direct Access Index** | Index managed via API (no auto-sync) |
# MAGIC | **Primary Key** | Unique identifier for each vector (chunk_id) |
# MAGIC | **Embedding Column** | Column containing the vector data |
# MAGIC
# MAGIC ### Why Delta Sync Index?
# MAGIC
# MAGIC We use Delta Sync (not Direct Access) because:
# MAGIC 1. **Automatic updates** - No manual re-indexing when data changes
# MAGIC 2. **Consistency** - Index always reflects the latest table state
# MAGIC 3. **Simplicity** - Single source of truth in Delta table
# MAGIC 4. **Governance** - Inherits Unity Catalog permissions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Vector Search Endpoint
# MAGIC
# MAGIC The Vector Search endpoint is a managed compute resource that hosts your vector indices.
# MAGIC
# MAGIC **Endpoint Characteristics:**
# MAGIC - **Shared resource**: Multiple indices can run on one endpoint
# MAGIC - **Auto-scaling**: Scales based on query load
# MAGIC - **High availability**: Managed by Databricks
# MAGIC - **Provisioning time**: 5-10 minutes for new endpoints
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Checks if the endpoint already exists
# MAGIC 2. If not, creates a new STANDARD endpoint
# MAGIC 3. Waits for the endpoint to reach ONLINE state
# MAGIC 4. Reports the final status
# MAGIC
# MAGIC **Note:** If you already have an endpoint in your workspace, you can reuse it by updating `VECTOR_SEARCH_ENDPOINT_NAME` in the configuration.

# COMMAND ----------

# ============================================================
# CREATE VECTOR SEARCH ENDPOINT
# ============================================================

def create_vector_search_endpoint(endpoint_name: str) -> None:
    """Create a Vector Search endpoint if it doesn't exist."""
    try:
        # Check if endpoint exists
        endpoint = vector_search_client.get_endpoint(endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' already exists")
        print(f"   Status: {endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')}")
        return
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            print(f"🔄 Creating endpoint '{endpoint_name}'...")
        else:
            raise e

    # Create new endpoint
    vector_search_client.create_endpoint(
        name=endpoint_name,
        endpoint_type="STANDARD"
    )

    print(f"⏳ Endpoint creation initiated. This may take 5-10 minutes...")

    # Wait for endpoint to be ready
    max_wait_time = 600  # 10 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            endpoint = vector_search_client.get_endpoint(endpoint_name)
            state = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")

            if state == "ONLINE":
                print(f"✅ Endpoint '{endpoint_name}' is ready!")
                return
            else:
                print(f"   Current state: {state}...")
                time.sleep(30)
        except Exception as e:
            print(f"   Waiting for endpoint... ({str(e)[:50]})")
            time.sleep(30)

    print(f"⚠️ Endpoint creation timed out. Please check the Databricks UI.")

# Create the endpoint
create_vector_search_endpoint(VECTOR_SEARCH_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Delta Sync Vector Index
# MAGIC
# MAGIC We create a **Delta Sync Index** that automatically synchronizes with our Delta table.
# MAGIC
# MAGIC **Index Configuration Parameters:**
# MAGIC
# MAGIC | Parameter | Value | Purpose |
# MAGIC |-----------|-------|---------|
# MAGIC | `endpoint_name` | Your endpoint | Where to host the index |
# MAGIC | `index_name` | Full Unity Catalog path | Unique identifier for the index |
# MAGIC | `source_table_name` | Chunks table path | Delta table to sync from |
# MAGIC | `primary_key` | `chunk_id` | Unique identifier for each vector |
# MAGIC | `embedding_vector_column` | `embedding` | Column containing vectors |
# MAGIC | `embedding_dimension` | 1024 | Must match embedding model output |
# MAGIC | `pipeline_type` | `TRIGGERED` | Manual sync control (vs. CONTINUOUS) |
# MAGIC | `columns_to_sync` | Metadata columns | Columns available for filtering/display |
# MAGIC
# MAGIC **Pipeline Types:**
# MAGIC - **TRIGGERED**: Sync runs when you call `sync()` - better for batch updates
# MAGIC - **CONTINUOUS**: Syncs automatically every few minutes - better for streaming
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Checks if the index already exists (idempotent)
# MAGIC 2. Creates a Delta Sync index with the specified configuration
# MAGIC 3. Waits for the index to become ready (initial sync)
# MAGIC 4. Reports the number of indexed rows
# MAGIC
# MAGIC **Note:** Initial index creation includes the first sync, which may take several minutes depending on data size.

# COMMAND ----------

# ============================================================
# CREATE VECTOR SEARCH INDEX
# ============================================================

def create_vector_index(
    endpoint_name: str,
    index_name: str,
    source_table: str,
    primary_key: str,
    embedding_column: str,
    embedding_dimension: int
) -> None:
    """Create a Delta Sync Vector Search index."""

    try:
        # Check if index exists
        index = vector_search_client.get_index(endpoint_name, index_name)
        index_info = index.describe()
        print(f"✅ Index '{index_name}' already exists")
        print(f"   Status: {index_info.get('status', {}).get('ready', False)}")
        return
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e):
            print(f"🔄 Creating index '{index_name}'...")
        else:
            raise e

    # Create the index
    vector_search_client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=source_table,
        primary_key=primary_key,
        embedding_dimension=embedding_dimension,
        embedding_vector_column=embedding_column,
        pipeline_type="TRIGGERED",  # Use TRIGGERED for manual sync control
        columns_to_sync=["chunk_id", "doc_id", "title", "category", "content", "chunk_index"]
    )

    print(f"⏳ Index creation initiated. This may take a few minutes...")

    # Wait for index to be ready
    max_wait_time = 600  # 10 minutes
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            index = vector_search_client.get_index(endpoint_name, index_name)
            index_info = index.describe()
            status = index_info.get("status", {})
            ready = status.get("ready", False)

            if ready:
                print(f"✅ Index '{index_name}' is ready!")
                print(f"   Indexed rows: {status.get('num_rows', 'N/A')}")
                return
            else:
                state = status.get("detailed_state", "UNKNOWN")
                print(f"   Current state: {state}...")
                time.sleep(30)
        except Exception as e:
            print(f"   Waiting for index... ({str(e)[:50]})")
            time.sleep(30)

    print(f"⚠️ Index creation timed out. Please check the Databricks UI.")

# Create the vector index
create_vector_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=VECTOR_INDEX_PATH,
    source_table=CHUNKS_TABLE_PATH,
    primary_key="chunk_id",
    embedding_column="embedding",
    embedding_dimension=EMBEDDING_DIMENSION
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Index Status and Test Query
# MAGIC
# MAGIC Before proceeding, we verify the index is operational by:
# MAGIC 1. Checking the index status and metadata
# MAGIC 2. Running a test similarity search
# MAGIC 3. Validating the result format
# MAGIC
# MAGIC **Understanding Similarity Scores:**
# MAGIC - Scores range from 0 to 1 (higher = more similar)
# MAGIC - Scores above 0.7 typically indicate strong relevance
# MAGIC - Scores below 0.5 may indicate weak matches
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Gets a reference to the index object
# MAGIC 2. Calls `describe()` to retrieve index metadata
# MAGIC 3. Generates an embedding for a test query
# MAGIC 4. Executes a similarity search with `num_results=TOP_K_RESULTS`
# MAGIC 5. Displays results with scores and content previews
# MAGIC
# MAGIC **Troubleshooting:**
# MAGIC - If status is not "ready", wait for sync to complete
# MAGIC - If no results, verify the embedding dimension matches
# MAGIC - If scores are low, check that the query is relevant to your content

# COMMAND ----------

# ============================================================
# VERIFY INDEX AND TEST QUERY
# ============================================================

# Get a reference to the index object
index = vector_search_client.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=VECTOR_INDEX_PATH
)

# Retrieve and display index metadata
index_info = index.describe()
print("📊 Index Information:")
print(f"   Name: {VECTOR_INDEX_PATH}")
print(f"   Endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
print(f"   Status: {index_info.get('status', {})}")

# Test query to verify the index is working
test_query = "How do I authenticate users with the API?"
print(f"\n🔍 Test Query: '{test_query}'")

# Generate embedding for the query using the same model used for indexing
query_embedding = embedding_model.embed_query(test_query)

# Execute similarity search against the index
results = index.similarity_search(
    query_vector=query_embedding,
    num_results=TOP_K_RESULTS,
    columns=["chunk_id", "doc_id", "title", "category", "content"]
)

# Display results with similarity scores
print(f"\n📋 Top {TOP_K_RESULTS} Results:")
for i, result in enumerate(results.get("result", {}).get("data_array", [])):
    # Result format: [chunk_id, doc_id, title, category, content, score]
    print(f"\n{i+1}. {result[2]} (Score: {result[-1]:.4f})")
    print(f"   Category: {result[3]}")
    print(f"   Content preview: {result[4][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 5: Configuring the Retriever for Semantic Search
# MAGIC
# MAGIC The retriever is the bridge between user queries and the Vector Search index. It provides a clean, LangChain-compatible interface for the RAG pipeline.
# MAGIC
# MAGIC ### Retriever Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐    ┌──────────────────────┐    ┌──────────────┐
# MAGIC │  User Query  │───▶│  DatabricksVector    │───▶│   Vector     │
# MAGIC │   (text)     │    │  Search Retriever    │    │   Search     │
# MAGIC └──────────────┘    └──────────────────────┘    │   Index      │
# MAGIC                               │                  └──────────────┘
# MAGIC                               ▼
# MAGIC                     ┌──────────────────────┐
# MAGIC                     │  LangChain Documents │
# MAGIC                     │  (page_content +     │
# MAGIC                     │   metadata)          │
# MAGIC                     └──────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Key Configuration Options
# MAGIC
# MAGIC | Parameter | Description | Our Setting |
# MAGIC |-----------|-------------|-------------|
# MAGIC | `text_column` | Column containing document text | `content` |
# MAGIC | `columns` | Metadata columns to include | chunk_id, doc_id, title, category |
# MAGIC | `search_type` | Similarity algorithm | `similarity` (cosine) |
# MAGIC | `k` | Number of results to return | 5 (configurable) |
# MAGIC | `filters` | Metadata filters | Optional (e.g., category) |
# MAGIC
# MAGIC **Why LangChain Integration?**
# MAGIC - Standardized interface compatible with any LLM
# MAGIC - Easy to swap retrieval backends
# MAGIC - Built-in support for chains and agents
# MAGIC - Consistent document format (page_content + metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the LangChain Retriever
# MAGIC
# MAGIC This cell creates a `DatabricksVectorSearch` wrapper and converts it to a LangChain retriever.
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Creates a `DatabricksVectorSearch` object wrapping our index
# MAGIC 2. Specifies which column contains the text content
# MAGIC 3. Lists metadata columns to include in results
# MAGIC 4. Converts to a retriever with search parameters

# COMMAND ----------

# ============================================================
# CONFIGURE LANGCHAIN RETRIEVER
# ============================================================

# Create DatabricksVectorSearch wrapper
# This provides LangChain compatibility for our Vector Search index
vector_store = DatabricksVectorSearch(
    index=index,                    # The Vector Search index object
    embedding=embedding_model,      # Same embedding model used for indexing
    text_column="content",          # Column containing document text
    columns=["chunk_id", "doc_id", "title", "category", "chunk_index"]  # Metadata
)

# Convert to a LangChain retriever
# The retriever provides a simple invoke(query) interface
retriever = vector_store.as_retriever(
    search_type="similarity",       # Use cosine similarity
    search_kwargs={
        "k": TOP_K_RESULTS,         # Number of results to return
        # Optional: Add filter for metadata-based filtering
        # "filter": {"category": "technical"}
    }
)

print("✅ Retriever configured successfully!")
print(f"   Search type: similarity")
print(f"   Top-K results: {TOP_K_RESULTS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Retriever
# MAGIC
# MAGIC We test the retriever with diverse queries to validate:
# MAGIC 1. **Relevance** - Are the returned documents related to the query?
# MAGIC 2. **Diversity** - Do results come from different document types?
# MAGIC 3. **Metadata** - Is metadata correctly attached to results?
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Defines a set of test queries covering different topics
# MAGIC 2. Invokes the retriever for each query
# MAGIC 3. Displays the top 3 results with metadata
# MAGIC 4. Allows you to assess retrieval quality

# COMMAND ----------

# ============================================================
# TEST RETRIEVER WITH SAMPLE QUERIES
# ============================================================

# Test queries covering different document categories
test_queries = [
    "How do I troubleshoot Kubernetes pod crashes?",
    "What is the data retention policy for PII?",
    "What was the root cause of the payment outage?",
    "How do I deploy a machine learning model?",
    "What are the code review best practices?"
]

print("🔍 Testing Retriever with Sample Queries\n")
print("=" * 80)

for query in test_queries:
    print(f"\n📝 Query: {query}")
    print("-" * 60)

    # Invoke the retriever - returns List[Document]
    docs = retriever.invoke(query)

    # Display top 3 results
    for i, doc in enumerate(docs[:3]):
        print(f"\n   Result {i+1}: {doc.metadata.get('title', 'N/A')}")
        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
        print(f"   Preview: {doc.page_content[:100]}...")

    print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implement Filtered Retrieval
# MAGIC
# MAGIC Filtering narrows search results based on metadata. This is essential for:
# MAGIC - **Multi-tenant systems** - Restrict results to user's organization
# MAGIC - **Category-specific search** - Search only technical docs or policies
# MAGIC - **Time-based filtering** - Find recent documents only
# MAGIC - **Access control** - Enforce document-level permissions
# MAGIC
# MAGIC **Filter Syntax:**
# MAGIC ```python
# MAGIC filters = {"category": "technical"}           # Exact match
# MAGIC filters = {"category IN": ["technical", "wiki"]}  # Multiple values
# MAGIC ```
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Defines a helper function for filtered retrieval
# MAGIC 2. Demonstrates filtering by category
# MAGIC 3. Shows how filters affect result sets

# COMMAND ----------

# ============================================================
# FILTERED RETRIEVAL EXAMPLES
# ============================================================

def retrieve_with_filter(query: str, category_filter: str = None, top_k: int = 5) -> List:
    """Retrieve documents with optional category filtering."""

    search_kwargs = {"k": top_k}

    if category_filter:
        search_kwargs["filter"] = {"category": category_filter}

    filtered_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    return filtered_retriever.invoke(query)

# Test filtered retrieval
print("🔍 Filtered Retrieval Examples\n")

# Search only in technical documents
print("📁 Category: technical")
print("-" * 40)
tech_results = retrieve_with_filter(
    "How do I handle authentication errors?",
    category_filter="technical"
)
for doc in tech_results[:3]:
    print(f"   • {doc.metadata.get('title')}")

# Search only in operational reports
print("\n📁 Category: operational")
print("-" * 40)
ops_results = retrieve_with_filter(
    "What incidents occurred recently?",
    category_filter="operational"
)
for doc in ops_results[:3]:
    print(f"   • {doc.metadata.get('title')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 6: Serving LLM and Embedding Models with Mosaic AI
# MAGIC
# MAGIC This section brings together retrieval and generation to create a complete RAG system.
# MAGIC
# MAGIC ### RAG Pipeline Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
# MAGIC │  User Query  │───▶│  Retriever   │───▶│   Prompt     │───▶│     LLM      │
# MAGIC │              │    │  (Vector     │    │  Template    │    │  (Llama 3.3) │
# MAGIC │              │    │   Search)    │    │              │    │              │
# MAGIC └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
# MAGIC                            │                   │                    │
# MAGIC                            ▼                   ▼                    ▼
# MAGIC                     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
# MAGIC                     │  Retrieved   │    │   Context +  │    │   Generated  │
# MAGIC                     │  Documents   │    │   Question   │    │   Response   │
# MAGIC                     └──────────────┘    └──────────────┘    └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Mosaic AI Foundation Models
# MAGIC
# MAGIC Databricks provides pre-deployed Foundation Models with:
# MAGIC
# MAGIC | Feature | Benefit |
# MAGIC |---------|---------|
# MAGIC | **Pay-per-token** | No idle compute costs |
# MAGIC | **Auto-scaling** | Handles traffic spikes automatically |
# MAGIC | **Low latency** | Optimized inference infrastructure |
# MAGIC | **Enterprise security** | VPC, encryption, audit logs |
# MAGIC | **No management** | Databricks handles updates and maintenance |
# MAGIC
# MAGIC ### Available Models in Your Workspace
# MAGIC
# MAGIC Based on the endpoint discovery at the time of this lab from previous cell you have access to:
# MAGIC - **Llama 3.3 70B** - Excellent balance of quality and speed (configured)
# MAGIC - **Llama 3.1 405B** - Highest quality, slower
# MAGIC - **Claude Sonnet 4** - Strong reasoning capabilities
# MAGIC - **Gemma 3 12B** - Fastest, good for simple tasks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure LLM for Response Generation
# MAGIC
# MAGIC We configure the `ChatDatabricks` wrapper to use the Llama 3.3 70B model.
# MAGIC
# MAGIC **LLM Parameters:**
# MAGIC
# MAGIC | Parameter | Value | Purpose |
# MAGIC |-----------|-------|---------|
# MAGIC | `endpoint` | databricks-meta-llama-3-3-70b-instruct | Model to use |
# MAGIC | `temperature` | 0.1 | Low = more deterministic, focused responses |
# MAGIC | `max_tokens` | 1024 | Maximum response length |
# MAGIC
# MAGIC **Temperature Tuning (Chapter 9 Insight):**
# MAGIC - **0.0-0.3**: Factual, consistent responses (best for RAG)
# MAGIC - **0.4-0.7**: Balanced creativity and accuracy
# MAGIC - **0.8-1.0**: Creative, varied responses (not recommended for RAG)
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Initializes the ChatDatabricks LLM wrapper
# MAGIC 2. Sends a test query to verify the endpoint is working
# MAGIC 3. Displays the response to confirm connectivity

# COMMAND ----------

# ============================================================
# CONFIGURE LLM FOR RESPONSE GENERATION
# ============================================================

# Initialize the LLM with tuned parameters
llm = ChatDatabricks(
    endpoint=LLM_MODEL_NAME,    # Foundation Model endpoint
    temperature=0.1,            # Low temperature for focused, factual responses
    max_tokens=1024,            # Maximum tokens in the response
)

# Test the LLM with a simple query
test_response = llm.invoke("What is retrieval-augmented generation in one sentence?")
print("🤖 LLM Test Response:")
print(f"   {test_response.content}")
print(f"\n✅ LLM configured successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the Complete RAG Pipeline
# MAGIC
# MAGIC Now we combine the retriever and LLM into a complete RAG pipeline using LangChain Expression Language (LCEL).
# MAGIC
# MAGIC **Pipeline Components:**
# MAGIC
# MAGIC | Component | Purpose |
# MAGIC |-----------|---------|
# MAGIC | `retriever` | Fetches relevant documents from Vector Search |
# MAGIC | `format_docs` | Converts Document objects to a context string |
# MAGIC | `rag_prompt` | Template that combines context and question |
# MAGIC | `llm` | Generates the final response |
# MAGIC | `StrOutputParser` | Extracts the text content from the response |
# MAGIC
# MAGIC **Prompt Engineering Best Practices:**
# MAGIC 1. **Clear instructions** - Tell the model exactly what to do
# MAGIC 2. **Context first** - Place retrieved documents before the question
# MAGIC 3. **Explicit constraints** - "Use only the information from the context"
# MAGIC 4. **Fallback behavior** - "If the context doesn't contain enough information, say so"
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Imports LangChain core components
# MAGIC 2. Defines a RAG prompt template with clear instructions
# MAGIC 3. Creates a document formatting function
# MAGIC 4. Builds the chain using LCEL pipe syntax

# COMMAND ----------

# ============================================================
# BUILD RAG PIPELINE
# ============================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Define the RAG prompt template
# This template instructs the LLM to answer based only on provided context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question. If the context doesn't contain
enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context above."""

# Create the prompt template object
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs):
    """
    Format retrieved documents into a single context string.

    Each document is labeled with its title for attribution.
    This helps the LLM understand the source of information.
    """
    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(f"[Document {i+1}: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# Build the RAG chain using LangChain Expression Language (LCEL)
# The pipe (|) operator chains components together
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve docs, then format
        "question": RunnablePassthrough()    # Pass question through unchanged
    }
    | rag_prompt    # Combine into prompt
    | llm           # Generate response
    | StrOutputParser()  # Extract text content
)

print("✅ RAG pipeline built successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the RAG Pipeline
# MAGIC
# MAGIC Now we test the complete end-to-end RAG pipeline with diverse queries.
# MAGIC
# MAGIC **What to Look For:**
# MAGIC 1. **Accuracy** - Does the answer match the source documents?
# MAGIC 2. **Relevance** - Is the answer focused on the question?
# MAGIC 3. **Attribution** - Can you trace the answer to specific documents?
# MAGIC 4. **Latency** - Is the response time acceptable?
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Defines test queries covering different document categories
# MAGIC 2. Invokes the RAG chain for each query
# MAGIC 3. Measures and displays response time
# MAGIC 4. Shows the generated answer

# COMMAND ----------

# ============================================================
# TEST RAG PIPELINE
# ============================================================

# Test queries covering different document types and topics
rag_test_queries = [
    "What are the steps to authenticate a user using the API?",
    "How should I handle a Pod CrashLoopBackOff error in Kubernetes?",
    "What was the impact of the payment processing outage?",
    "What are the data retention policies for different types of data?",
    "How do I deploy a machine learning model to production?"
]

print("🚀 Testing RAG Pipeline\n")
print("=" * 80)

for query in rag_test_queries:
    print(f"\n❓ Question: {query}")
    print("-" * 60)

    # Invoke the RAG chain and measure latency
    start_time = time.time()
    response = rag_chain.invoke(query)
    elapsed = time.time() - start_time

    print(f"\n💬 Answer:\n{response}")
    print(f"\n⏱️  Response time: {elapsed:.2f} seconds")
    print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 7: Batching and Context Length Optimization
# MAGIC
# MAGIC Production RAG systems must handle high query volumes efficiently. This section covers key optimization techniques from Chapter 9.
# MAGIC
# MAGIC ### Optimization Strategies Overview
# MAGIC
# MAGIC | Strategy | Benefit | Trade-off |
# MAGIC |----------|---------|-----------|
# MAGIC | **Batch processing** | Higher throughput | Increased latency for individual queries |
# MAGIC | **Parallel execution** | Reduced total time | Higher resource consumption |
# MAGIC | **Context length tuning** | Lower token costs | Potential loss of relevant context |
# MAGIC | **Caching** | Faster repeated queries | Memory usage, staleness |
# MAGIC
# MAGIC ### Why These Optimizations Matter
# MAGIC
# MAGIC Consider a production scenario:
# MAGIC - **1000 queries/day** at 3 seconds each = 50 minutes of compute
# MAGIC - **With 4x parallelism** = 12.5 minutes of compute
# MAGIC - **With caching (50% hit rate)** = 6.25 minutes of compute
# MAGIC
# MAGIC These optimizations directly impact cost and user experience.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Query Processing
# MAGIC
# MAGIC Batch processing enables handling multiple queries efficiently using parallel execution.
# MAGIC
# MAGIC **Concurrency Considerations:**
# MAGIC - **max_workers=1**: Sequential processing, lowest resource usage
# MAGIC - **max_workers=4**: Good balance for most workloads
# MAGIC - **max_workers=8+**: High throughput, but may hit rate limits
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Defines a single-query processing function with timing
# MAGIC 2. Implements a batch processor using ThreadPoolExecutor
# MAGIC 3. Tracks progress and calculates throughput metrics
# MAGIC 4. Returns structured results for analysis

# COMMAND ----------

# ============================================================
# BATCH QUERY PROCESSING
# ============================================================

import concurrent.futures
from typing import Tuple

def process_query(query: str) -> Tuple[str, str, float]:
    """Process a single query and return the result with timing."""
    start_time = time.time()
    response = rag_chain.invoke(query)
    elapsed = time.time() - start_time
    return query, response, elapsed

def batch_process_queries(
    queries: List[str],
    max_workers: int = 4,
    show_progress: bool = True
) -> List[Dict]:
    """Process multiple queries in parallel."""
    results = []

    if show_progress:
        print(f"🔄 Processing {len(queries)} queries with {max_workers} workers...")

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(process_query, query): query
            for query in queries
        }

        for future in concurrent.futures.as_completed(future_to_query):
            query, response, elapsed = future.result()
            results.append({
                "query": query,
                "response": response,
                "latency": elapsed
            })

            if show_progress:
                print(f"   ✓ Completed: {query[:50]}... ({elapsed:.2f}s)")

    total_time = time.time() - start_time

    if show_progress:
        print(f"\n✅ Batch processing complete!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Throughput: {len(queries) / total_time:.2f} queries/second")

    return results

# Test batch processing
batch_queries = [
    "What is the authentication flow?",
    "How do I check Kubernetes pod logs?",
    "What is the data pipeline architecture?",
    "What are the on-call responsibilities?",
    "What was the Q4 system uptime?"
]

batch_results = batch_process_queries(batch_queries, max_workers=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Context Length Tuning
# MAGIC
# MAGIC Context length is a critical tuning parameter that affects:
# MAGIC 1. **Response quality** - More context = more information for the LLM
# MAGIC 2. **Token costs** - More context = higher API costs
# MAGIC 3. **Latency** - More context = longer processing time
# MAGIC 4. **Relevance dilution** - Too much context = irrelevant information
# MAGIC
# MAGIC **Tuning Strategy (Chapter 9 Insight):**
# MAGIC
# MAGIC | Scenario | Recommended Setting |
# MAGIC |----------|---------------------|
# MAGIC | Simple factual queries | top_k=3, 2000 chars |
# MAGIC | Complex multi-part questions | top_k=5-8, 4000-6000 chars |
# MAGIC | Summarization tasks | top_k=10+, 8000+ chars |
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Creates a function to build RAG chains with custom parameters
# MAGIC 2. Implements context truncation to enforce character limits
# MAGIC 3. Tests three configurations (small, medium, large)
# MAGIC 4. Compares response quality and latency

# COMMAND ----------

# ============================================================
# CONTEXT LENGTH TUNING EXPERIMENTS
# ============================================================

def create_tuned_rag_chain(top_k: int, max_context_chars: int = 4000):
    """Create a RAG chain with tuned context parameters."""

    # Create retriever with custom top-k
    tuned_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    def format_docs_with_limit(docs):
        """Format docs with character limit to control context length."""
        formatted = []
        total_chars = 0

        for i, doc in enumerate(docs):
            doc_text = f"[Document {i+1}: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"

            if total_chars + len(doc_text) > max_context_chars:
                # Truncate if exceeding limit
                remaining = max_context_chars - total_chars
                if remaining > 100:  # Only add if meaningful content remains
                    formatted.append(doc_text[:remaining] + "...")
                break

            formatted.append(doc_text)
            total_chars += len(doc_text)

        return "\n\n".join(formatted)

    # Build tuned chain
    tuned_chain = (
        {
            "context": tuned_retriever | format_docs_with_limit,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return tuned_chain

# Experiment with different configurations
configurations = [
    {"top_k": 3, "max_context_chars": 2000, "name": "Small Context"},
    {"top_k": 5, "max_context_chars": 4000, "name": "Medium Context"},
    {"top_k": 8, "max_context_chars": 6000, "name": "Large Context"},
]

test_query = "What are the best practices for code review and what should be included in the checklist?"

print("📊 Context Length Tuning Experiments\n")
print("=" * 80)

for config in configurations:
    print(f"\n🔧 Configuration: {config['name']}")
    print(f"   Top-K: {config['top_k']}, Max Context: {config['max_context_chars']} chars")
    print("-" * 60)

    tuned_chain = create_tuned_rag_chain(
        top_k=config["top_k"],
        max_context_chars=config["max_context_chars"]
    )

    start_time = time.time()
    response = tuned_chain.invoke(test_query)
    elapsed = time.time() - start_time

    print(f"\n   Response preview: {response[:200]}...")
    print(f"   Response length: {len(response)} chars")
    print(f"   Latency: {elapsed:.2f}s")
    print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 8: Performance Profiling and Bottleneck Identification
# MAGIC
# MAGIC Performance profiling is essential for optimizing RAG systems. This section implements comprehensive profiling to identify bottlenecks.
# MAGIC
# MAGIC ### RAG Pipeline Latency Breakdown
# MAGIC
# MAGIC A typical RAG query involves four stages, each with different latency characteristics:
# MAGIC
# MAGIC | Stage | Typical Latency | Optimization Levers |
# MAGIC |-------|-----------------|---------------------|
# MAGIC | **Query Embedding** | 50-200ms | Batch queries, cache embeddings |
# MAGIC | **Vector Search** | 10-100ms | Index tuning, reduce top-k |
# MAGIC | **Context Formatting** | 1-10ms | Optimize string operations |
# MAGIC | **LLM Generation** | 500-5000ms | Smaller model, shorter context |
# MAGIC
# MAGIC **Key Insight:** LLM generation is typically the bottleneck (60-90% of total latency). Focus optimization efforts there first.
# MAGIC
# MAGIC ### Profiling Methodology
# MAGIC
# MAGIC We measure:
# MAGIC 1. **Latency per stage** - Where is time spent?
# MAGIC 2. **P50/P95 percentiles** - What's the typical vs. worst-case latency?
# MAGIC 3. **Throughput** - How many queries per second?
# MAGIC 4. **Resource utilization** - Are we hitting rate limits?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Component-Level Profiling
# MAGIC
# MAGIC The `RAGProfiler` class instruments each stage of the pipeline to collect detailed metrics.
# MAGIC
# MAGIC **Metrics Collected:**
# MAGIC - `embedding_latency` - Time to convert query to vector
# MAGIC - `search_latency` - Time for Vector Search similarity query
# MAGIC - `formatting_latency` - Time to format documents into context
# MAGIC - `generation_latency` - Time for LLM to generate response
# MAGIC - `total_latency` - End-to-end query time
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Defines the `RAGProfiler` class with timing instrumentation
# MAGIC 2. Implements `profile_query()` to measure a single query
# MAGIC 3. Implements `get_summary()` to aggregate statistics
# MAGIC 4. Provides percentile calculations for SLA monitoring

# COMMAND ----------

# ============================================================
# COMPONENT-LEVEL PROFILING
# ============================================================

class RAGProfiler:
    """Profiler for measuring RAG pipeline component latencies."""

    def __init__(self, embedding_model, vector_index, llm, retriever):
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.llm = llm
        self.retriever = retriever
        self.metrics = []

    def profile_query(self, query: str, top_k: int = 5) -> Dict:
        """Profile a single query through all pipeline stages."""
        metrics = {"query": query, "timestamp": datetime.now().isoformat()}

        # Stage 1: Query Embedding
        start = time.time()
        query_embedding = self.embedding_model.embed_query(query)
        metrics["embedding_latency"] = time.time() - start

        # Stage 2: Vector Search
        start = time.time()
        search_results = self.vector_index.similarity_search(
            query_vector=query_embedding,
            num_results=top_k,
            columns=["chunk_id", "doc_id", "title", "category", "content"]
        )
        metrics["search_latency"] = time.time() - start
        metrics["results_count"] = len(search_results.get("result", {}).get("data_array", []))

        # Stage 3: Context Formatting
        start = time.time()
        docs = self.retriever.invoke(query)
        context = format_docs(docs)
        metrics["formatting_latency"] = time.time() - start
        metrics["context_length"] = len(context)

        # Stage 4: LLM Generation
        start = time.time()
        prompt = rag_prompt.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        metrics["generation_latency"] = time.time() - start
        metrics["response_length"] = len(response.content)

        # Total latency
        metrics["total_latency"] = (
            metrics["embedding_latency"] +
            metrics["search_latency"] +
            metrics["formatting_latency"] +
            metrics["generation_latency"]
        )

        self.metrics.append(metrics)
        return metrics

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics for all profiled queries."""
        if not self.metrics:
            return pd.DataFrame()

        df = pd.DataFrame(self.metrics)

        summary = {
            "Metric": ["Embedding", "Search", "Formatting", "Generation", "Total"],
            "Mean (s)": [
                df["embedding_latency"].mean(),
                df["search_latency"].mean(),
                df["formatting_latency"].mean(),
                df["generation_latency"].mean(),
                df["total_latency"].mean()
            ],
            "P50 (s)": [
                df["embedding_latency"].median(),
                df["search_latency"].median(),
                df["formatting_latency"].median(),
                df["generation_latency"].median(),
                df["total_latency"].median()
            ],
            "P95 (s)": [
                df["embedding_latency"].quantile(0.95),
                df["search_latency"].quantile(0.95),
                df["formatting_latency"].quantile(0.95),
                df["generation_latency"].quantile(0.95),
                df["total_latency"].quantile(0.95)
            ]
        }

        return pd.DataFrame(summary)

# Initialize profiler
profiler = RAGProfiler(
    embedding_model=embedding_model,
    vector_index=index,
    llm=llm,
    retriever=retriever
)

print("✅ RAG Profiler initialized!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Profiling Benchmark
# MAGIC
# MAGIC Let's profile multiple queries to gather performance statistics and identify bottlenecks.

# COMMAND ----------

# ============================================================
# RUN PROFILING BENCHMARK
# ============================================================

profiling_queries = [
    "What is the authentication API endpoint for login?",
    "How do I troubleshoot ImagePullBackOff errors?",
    "What is the medallion architecture?",
    "What are the code review response time expectations?",
    "What was the root cause of the payment outage?",
    "How do I register a model in Unity Catalog?",
    "What is the data retention policy for transaction data?",
    "What are the on-call escalation paths?",
    "What was the Q4 uptime percentage?",
    "How do I configure liveness probes in Kubernetes?"
]

print("🔬 Running Profiling Benchmark\n")
print(f"   Queries: {len(profiling_queries)}")
print("-" * 60)

for i, query in enumerate(profiling_queries):
    metrics = profiler.profile_query(query)
    print(f"\n{i+1}. {query[:50]}...")
    print(f"   Embedding: {metrics['embedding_latency']:.3f}s | "
          f"Search: {metrics['search_latency']:.3f}s | "
          f"Generation: {metrics['generation_latency']:.3f}s | "
          f"Total: {metrics['total_latency']:.2f}s")

# Display summary
print("\n" + "=" * 60)
print("📊 Performance Summary")
print("=" * 60)
summary_df = profiler.get_summary()
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify Bottlenecks
# MAGIC
# MAGIC This cell analyzes the profiling data to identify the primary bottleneck and provide targeted optimization recommendations.
# MAGIC
# MAGIC **Bottleneck Analysis Approach:**
# MAGIC 1. Calculate total time spent in each stage across all queries
# MAGIC 2. Compute percentage distribution
# MAGIC 3. Identify the stage consuming the most time
# MAGIC 4. Provide specific recommendations based on the bottleneck
# MAGIC
# MAGIC **Common Bottleneck Patterns:**
# MAGIC
# MAGIC | Bottleneck | Typical Cause | Quick Wins |
# MAGIC |------------|---------------|------------|
# MAGIC | LLM Generation | Large context, complex queries | Reduce context, use faster model |
# MAGIC | Vector Search | Large index, high top-k | Reduce top-k, add filters |
# MAGIC | Embedding | No caching, large batches | Cache common queries |
# MAGIC | Formatting | Complex logic | Simplify, pre-compute |
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Aggregates timing data from all profiled queries
# MAGIC 2. Calculates percentage of time in each stage
# MAGIC 3. Visualizes the distribution with a bar chart
# MAGIC 4. Identifies the primary bottleneck
# MAGIC 5. Provides actionable optimization recommendations

# COMMAND ----------

# ============================================================
# BOTTLENECK ANALYSIS
# ============================================================

metrics_df = pd.DataFrame(profiler.metrics)

# Calculate percentage of time spent in each stage
total_time = metrics_df["total_latency"].sum()
stage_times = {
    "Embedding": metrics_df["embedding_latency"].sum(),
    "Vector Search": metrics_df["search_latency"].sum(),
    "Formatting": metrics_df["formatting_latency"].sum(),
    "LLM Generation": metrics_df["generation_latency"].sum()
}

print("🔍 Bottleneck Analysis\n")
print("=" * 60)
print("\n📊 Time Distribution by Component:")
print("-" * 40)

bottleneck = None
max_pct = 0

for stage, time_spent in stage_times.items():
    pct = (time_spent / total_time) * 100
    bar = "█" * int(pct / 2)
    print(f"   {stage:15} {pct:5.1f}% {bar}")

    if pct > max_pct:
        max_pct = pct
        bottleneck = stage

print(f"\n⚠️  Primary Bottleneck: {bottleneck} ({max_pct:.1f}% of total time)")

# Recommendations based on bottleneck
print("\n💡 Optimization Recommendations:")
print("-" * 40)

if bottleneck == "LLM Generation":
    print("""
   • Consider using a smaller/faster model for simple queries
   • Reduce max_tokens if responses are being truncated anyway
   • Implement response caching for common queries
   • Use streaming for better perceived latency
""")
elif bottleneck == "Vector Search":
    print("""
   • Reduce the number of results (top-k)
   • Optimize index configuration (e.g., HNSW parameters)
   • Consider using approximate search for faster results
   • Pre-filter using metadata to reduce search space
""")
elif bottleneck == "Embedding":
    print("""
   • Cache embeddings for frequently asked queries
   • Use a smaller embedding model if accuracy permits
   • Batch multiple queries together
   • Consider using quantized embeddings
""")
else:
    print("""
   • Optimize document formatting logic
   • Reduce context length if possible
   • Pre-compute formatted chunks
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 9: Optimization Strategies from Chapter 9
# MAGIC
# MAGIC In this section, we apply the optimization strategies introduced in Chapter 9:
# MAGIC
# MAGIC 1. **Context-Length Tuning**: Optimize the amount of context passed to the LLM
# MAGIC 2. **Embedding Dimensionality Adjustments**: Trade-off between accuracy and speed
# MAGIC 3. **Batching Optimization**: Improve throughput with efficient batch processing
# MAGIC 4. **Index-Level Performance Improvements**: Optimize Vector Search configuration
# MAGIC 5. **Query Caching**: Reduce redundant computations for repeated queries
# MAGIC 6. **Hybrid Search**: Combine semantic and keyword search for better relevance
# MAGIC 7. **Adaptive Retrieval**: Dynamically adjust parameters based on query type
# MAGIC
# MAGIC These patterns address common RAG performance and quality issues identified through profiling.
# MAGIC
# MAGIC ### Strategy Overview
# MAGIC
# MAGIC | Strategy | Benefit | Best For |
# MAGIC |----------|---------|----------|
# MAGIC | **Context-Length Tuning** | Balance quality vs. latency/cost | All RAG applications |
# MAGIC | **Embedding Dimensionality** | Reduce storage and search time | Large-scale deployments |
# MAGIC | **Batching Optimization** | Higher throughput | High-volume workloads |
# MAGIC | **Index-Level Tuning** | Faster vector search | Latency-sensitive apps |
# MAGIC | **Caching** | 10-100x faster for repeated queries | FAQ-style applications |
# MAGIC | **Hybrid Search** | Better precision for technical terms | Code/API documentation |
# MAGIC | **Adaptive Retrieval** | Optimized per-query performance | Diverse query types |
# MAGIC
# MAGIC > **Note:** Context-length tuning and batching optimization were covered in Part 7. This section focuses on additional strategies.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 1: Embedding Dimensionality Adjustments
# MAGIC
# MAGIC Embedding dimensionality directly impacts storage, search speed, and semantic accuracy. Higher dimensions capture more nuance but increase computational cost.
# MAGIC
# MAGIC **Dimensionality Trade-offs:**
# MAGIC
# MAGIC | Dimension | Storage per Vector | Search Speed | Semantic Quality |
# MAGIC |-----------|-------------------|--------------|------------------|
# MAGIC | 384 | 1.5 KB | Fastest | Good for simple queries |
# MAGIC | 768 | 3 KB | Fast | Balanced performance |
# MAGIC | 1024 | 4 KB | Moderate | High semantic nuance |
# MAGIC | 1536+ | 6+ KB | Slower | Maximum accuracy |
# MAGIC
# MAGIC **When to Reduce Dimensionality:**
# MAGIC - Large-scale deployments (millions of vectors)
# MAGIC - Latency-critical applications
# MAGIC - Simple, factual queries
# MAGIC
# MAGIC **Techniques for Dimensionality Reduction:**
# MAGIC 1. **Choose a smaller model**: Use `bge-base` (768d) instead of `bge-large` (1024d)
# MAGIC 2. **PCA/SVD reduction**: Reduce dimensions post-embedding
# MAGIC 3. **Matryoshka embeddings**: Models trained to work at multiple dimensions
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Demonstrates the impact of dimensionality on search performance
# MAGIC 2. Shows how to compare different embedding models
# MAGIC 3. Provides guidance on choosing the right dimensionality

# COMMAND ----------

# ============================================================
# STRATEGY 1: EMBEDDING DIMENSIONALITY ANALYSIS
# ============================================================

# Compare embedding model options available in Databricks
embedding_options = [
    {"model": "databricks-bge-large-en", "dimensions": 1024, "max_tokens": 512},
    {"model": "databricks-gte-large-en", "dimensions": 1024, "max_tokens": 512},
]

print("📐 Embedding Model Comparison\n")
print("=" * 70)
print(f"{'Model':<30} {'Dimensions':<12} {'Max Tokens':<12} {'Use Case':<20}")
print("-" * 70)

for opt in embedding_options:
    use_case = "High accuracy" if opt["dimensions"] >= 1024 else "Balanced"
    print(f"{opt['model']:<30} {opt['dimensions']:<12} {opt['max_tokens']:<12} {use_case:<20}")

print("\n" + "=" * 70)
print("\n💡 Current Configuration:")
print(f"   Model: {EMBEDDING_MODEL_NAME}")
print(f"   Dimensions: {EMBEDDING_DIMENSION}")
print(f"\n📊 Storage Impact:")
print(f"   Per vector: {EMBEDDING_DIMENSION * 4 / 1024:.1f} KB (float32)")
print(f"   For 1M vectors: {EMBEDDING_DIMENSION * 4 * 1_000_000 / (1024**3):.1f} GB")

print("\n🔧 Optimization Tips:")
print("   • For faster search: Consider models with 384-768 dimensions")
print("   • For higher accuracy: Use 1024+ dimensions (current setting)")
print("   • For cost savings: Smaller dimensions reduce storage and compute")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 2: Index-Level Performance Improvements
# MAGIC
# MAGIC Vector Search index configuration significantly impacts query latency and throughput. This section covers key tuning parameters.
# MAGIC
# MAGIC **Index Configuration Options:**
# MAGIC
# MAGIC | Parameter | Options | Impact |
# MAGIC |-----------|---------|--------|
# MAGIC | **Pipeline Type** | TRIGGERED vs CONTINUOUS | Sync frequency vs. freshness |
# MAGIC | **Sync Compute** | Serverless vs Provisioned | Cost vs. control |
# MAGIC | **Index Type** | Delta Sync vs Direct | Automatic sync vs. manual |
# MAGIC
# MAGIC **Performance Tuning Strategies:**
# MAGIC
# MAGIC 1. **Reduce search scope with filters**: Pre-filter by metadata before vector search
# MAGIC 2. **Optimize top-k**: Lower k = faster search, but may miss relevant docs
# MAGIC 3. **Use TRIGGERED sync**: Better control over when index updates occur
# MAGIC 4. **Monitor index health**: Check sync status and row counts regularly
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Analyzes current index configuration
# MAGIC 2. Provides recommendations for performance improvement
# MAGIC 3. Demonstrates metadata filtering for faster searches

# COMMAND ----------

# ============================================================
# STRATEGY 2: INDEX-LEVEL PERFORMANCE ANALYSIS
# ============================================================

print("🔍 Index-Level Performance Analysis\n")
print("=" * 70)

# Get current index configuration
try:
    index_info = index.describe()

    print("📋 Current Index Configuration:")
    print("-" * 40)
    print(f"   Index Name: {VECTOR_INDEX_PATH}")
    print(f"   Endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
    print(f"   Status: {index_info.get('status', {}).get('ready', 'Unknown')}")
    print(f"   Indexed Rows: {index_info.get('status', {}).get('num_rows', 'N/A')}")

    # Analyze configuration
    print("\n🔧 Performance Recommendations:")
    print("-" * 40)

    # Check pipeline type
    pipeline_type = index_info.get('delta_sync_index_spec', {}).get('pipeline_type', 'UNKNOWN')
    print(f"\n   Pipeline Type: {pipeline_type}")
    if pipeline_type == "TRIGGERED":
        print("   ✅ Good: TRIGGERED allows controlled sync timing")
    else:
        print("   ⚠️  Consider TRIGGERED for better control over sync costs")

    # Check embedding dimension
    embed_dim = index_info.get('delta_sync_index_spec', {}).get('embedding_dimension', EMBEDDING_DIMENSION)
    print(f"\n   Embedding Dimension: {embed_dim}")
    if embed_dim > 768:
        print("   💡 Tip: Consider smaller dimensions for faster search if accuracy permits")
    else:
        print("   ✅ Good: Balanced dimension for performance")

except Exception as e:
    print(f"   Could not retrieve index info: {str(e)[:50]}")

# Demonstrate filtered search for performance
print("\n" + "=" * 70)
print("\n📊 Filtered Search Performance Comparison:")
print("-" * 40)

test_query_embedding = embedding_model.embed_query("authentication API")

# Unfiltered search
start = time.time()
unfiltered_results = index.similarity_search(
    query_vector=test_query_embedding,
    num_results=5,
    columns=["chunk_id", "title", "category", "content"]
)
unfiltered_time = time.time() - start

# Filtered search (by category)
start = time.time()
filtered_results = index.similarity_search(
    query_vector=test_query_embedding,
    num_results=5,
    columns=["chunk_id", "title", "category", "content"],
    filters={"category": "technical"}
)
filtered_time = time.time() - start

print(f"\n   Unfiltered search: {unfiltered_time:.3f}s")
print(f"   Filtered search:   {filtered_time:.3f}s")
print(f"   Speedup:           {unfiltered_time/filtered_time:.1f}x" if filtered_time > 0 else "   N/A")

print("\n💡 Key Insight: Metadata filters reduce search scope and improve latency")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 3: Query Caching
# MAGIC
# MAGIC Caching stores responses for previously seen queries, providing instant responses for repeated questions.
# MAGIC
# MAGIC **Cache Design Considerations:**
# MAGIC - **Key generation**: Normalize queries (lowercase, strip whitespace)
# MAGIC - **Cache size**: Balance memory usage vs. hit rate
# MAGIC - **Eviction policy**: LRU (Least Recently Used) is common
# MAGIC - **TTL (Time-to-Live)**: Invalidate stale responses
# MAGIC
# MAGIC **When to Use Caching:**
# MAGIC - High query repetition (FAQ-style applications)
# MAGIC - Stable underlying data (infrequent updates)
# MAGIC - Latency-sensitive applications
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Implements a `CachedRAGPipeline` class with LRU-like caching
# MAGIC 2. Uses MD5 hashing for cache key generation
# MAGIC 3. Tracks cache hits/misses for monitoring
# MAGIC 4. Demonstrates cache effectiveness with repeated queries

# COMMAND ----------

# ============================================================
# STRATEGY 3: QUERY CACHING
# ============================================================

from functools import lru_cache
import hashlib

class CachedRAGPipeline:
    """RAG pipeline with query caching for improved performance."""

    def __init__(self, rag_chain, cache_size: int = 100):
        self.rag_chain = rag_chain
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for the query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def invoke(self, query: str) -> str:
        """Invoke the RAG pipeline with caching."""
        cache_key = self._get_cache_key(query)

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1
        response = self.rag_chain.invoke(query)

        # Add to cache (simple LRU-like behavior)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = response
        return response

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }

# Create cached pipeline
cached_pipeline = CachedRAGPipeline(rag_chain, cache_size=50)

# Test caching
print("🔄 Testing Query Caching\n")

test_queries_with_repeats = [
    "What is the authentication flow?",
    "How do I deploy a model?",
    "What is the authentication flow?",  # Repeat
    "What are the code review practices?",
    "How do I deploy a model?",  # Repeat
    "What is the authentication flow?",  # Repeat
]

for query in test_queries_with_repeats:
    start = time.time()
    response = cached_pipeline.invoke(query)
    elapsed = time.time() - start
    print(f"   Query: {query[:40]}... | Time: {elapsed:.3f}s")

print(f"\n📊 Cache Statistics: {cached_pipeline.get_stats()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 4: Hybrid Search with Keyword Boosting
# MAGIC
# MAGIC Semantic search excels at understanding meaning, but can miss exact keyword matches. Hybrid search combines both approaches.
# MAGIC
# MAGIC **Why Hybrid Search?**
# MAGIC - **Semantic search** understands "authentication" ≈ "login" ≈ "sign in"
# MAGIC - **Keyword search** ensures exact matches for technical terms like "CrashLoopBackOff"
# MAGIC - **Combined** provides the best of both worlds
# MAGIC
# MAGIC **Weighting Strategy:**
# MAGIC - `semantic_weight=0.7` - Prioritize semantic understanding
# MAGIC - `keyword_weight=0.3` - Boost exact keyword matches
# MAGIC - Adjust based on your content type (more technical = higher keyword weight)
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Retrieves more results than needed (top_k * 2) for re-ranking
# MAGIC 2. Extracts keywords from the query (removing stop words)
# MAGIC 3. Calculates keyword match score for each result
# MAGIC 4. Combines semantic and keyword scores with configurable weights
# MAGIC 5. Re-ranks and returns the top-k results

# COMMAND ----------

# ============================================================
# STRATEGY 4: HYBRID SEARCH SIMULATION
# ============================================================

def hybrid_search(
    query: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: int = 5
) -> List[Dict]:
    """
    Simulate hybrid search by combining semantic and keyword scores.

    In production, you would use Databricks Vector Search's built-in
    hybrid search capabilities or combine with a keyword index.
    """
    # Get semantic search results
    query_embedding = embedding_model.embed_query(query)
    semantic_results = index.similarity_search(
        query_vector=query_embedding,
        num_results=top_k * 2,  # Get more results for re-ranking
        columns=["chunk_id", "doc_id", "title", "category", "content"]
    )

    # Extract keywords from query (simple approach)
    query_keywords = set(query.lower().split())
    stop_words = {"what", "how", "is", "the", "a", "an", "to", "do", "i", "for", "in", "of"}
    query_keywords = query_keywords - stop_words

    # Score and re-rank results
    scored_results = []
    for result in semantic_results.get("result", {}).get("data_array", []):
        content = result[4].lower()  # content column

        # Calculate keyword score
        keyword_matches = sum(1 for kw in query_keywords if kw in content)
        keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0

        # Combine scores
        semantic_score = result[-1]  # similarity score
        combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)

        scored_results.append({
            "chunk_id": result[0],
            "title": result[2],
            "category": result[3],
            "content": result[4],
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "combined_score": combined_score
        })

    # Sort by combined score and return top-k
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored_results[:top_k]

# Test hybrid search
print("🔍 Hybrid Search Results\n")
test_query = "kubectl pod CrashLoopBackOff troubleshooting"
print(f"Query: {test_query}\n")

hybrid_results = hybrid_search(test_query)

for i, result in enumerate(hybrid_results):
    print(f"{i+1}. {result['title']}")
    print(f"   Semantic: {result['semantic_score']:.4f} | "
          f"Keyword: {result['keyword_score']:.4f} | "
          f"Combined: {result['combined_score']:.4f}")
    print(f"   Preview: {result['content'][:100]}...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 5: Adaptive Retrieval
# MAGIC
# MAGIC Different queries benefit from different retrieval strategies. Adaptive retrieval analyzes the query and adjusts parameters accordingly.
# MAGIC
# MAGIC **Query Type Detection:**
# MAGIC
# MAGIC | Query Type | Indicators | Optimal Strategy |
# MAGIC |------------|------------|------------------|
# MAGIC | **Specific** | "error", "API", "endpoint" | Low top-k (3), high precision |
# MAGIC | **Broad** | "overview", "explain", "what is" | High top-k (8), more context |
# MAGIC | **Troubleshooting** | "fix", "issue", "problem" | Medium top-k (5), category filter |
# MAGIC
# MAGIC **Adaptive Parameters:**
# MAGIC - `top_k` - Number of results to retrieve
# MAGIC - `max_context_chars` - Context length limit
# MAGIC - `category_filter` - Restrict to relevant document types
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Analyzes the query to detect its type (specific, broad, troubleshooting)
# MAGIC 2. Selects optimal retrieval parameters based on query type
# MAGIC 3. Creates a customized RAG chain for the query
# MAGIC 4. Demonstrates how different queries get different treatment

# COMMAND ----------

# ============================================================
# STRATEGY 5: ADAPTIVE RETRIEVAL
# ============================================================

def analyze_query(query: str) -> Dict:
    """Analyze query to determine optimal retrieval strategy."""
    query_lower = query.lower()

    # Detect query type
    is_specific = any(word in query_lower for word in ["error", "code", "api", "endpoint", "command"])
    is_broad = any(word in query_lower for word in ["overview", "explain", "describe", "what is"])
    is_troubleshooting = any(word in query_lower for word in ["troubleshoot", "fix", "error", "issue", "problem"])

    # Determine parameters
    if is_troubleshooting:
        return {
            "type": "troubleshooting",
            "top_k": 7,
            "max_context": 5000,
            "temperature": 0.1
        }
    elif is_specific:
        return {
            "type": "specific",
            "top_k": 3,
            "max_context": 2000,
            "temperature": 0.0
        }
    elif is_broad:
        return {
            "type": "broad",
            "top_k": 5,
            "max_context": 4000,
            "temperature": 0.2
        }
    else:
        return {
            "type": "default",
            "top_k": 5,
            "max_context": 3000,
            "temperature": 0.1
        }

def adaptive_rag(query: str) -> str:
    """Execute RAG with adaptive parameters based on query analysis."""
    params = analyze_query(query)

    print(f"   Query type: {params['type']}")
    print(f"   Parameters: top_k={params['top_k']}, max_context={params['max_context']}")

    # Create chain with adaptive parameters
    adaptive_chain = create_tuned_rag_chain(
        top_k=params["top_k"],
        max_context_chars=params["max_context"]
    )

    return adaptive_chain.invoke(query)

# Test adaptive retrieval
print("🎯 Adaptive Retrieval Examples\n")
print("=" * 60)

adaptive_queries = [
    "What is the data pipeline architecture?",  # Broad
    "How do I fix ImagePullBackOff error?",  # Troubleshooting
    "What is the API endpoint for user login?",  # Specific
]

for query in adaptive_queries:
    print(f"\n❓ Query: {query}")
    print("-" * 40)
    response = adaptive_rag(query)
    print(f"\n💬 Response: {response[:200]}...")
    print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 6: Performance Monitoring Dashboard
# MAGIC
# MAGIC Production RAG systems require monitoring to ensure reliability and performance. This strategy implements a simple monitoring dashboard.
# MAGIC
# MAGIC **Key Metrics to Monitor:**
# MAGIC
# MAGIC | Metric | Purpose | Alert Threshold |
# MAGIC |--------|---------|-----------------|
# MAGIC | **Success Rate** | System reliability | < 99% |
# MAGIC | **P50 Latency** | Typical user experience | > 3s |
# MAGIC | **P95 Latency** | Worst-case experience | > 10s |
# MAGIC | **Error Count** | System health | > 0 |
# MAGIC | **Requests/min** | Load tracking | Varies |
# MAGIC
# MAGIC **Production Monitoring Considerations:**
# MAGIC - Integrate with Databricks Lakehouse Monitoring
# MAGIC - Set up alerts for SLA violations
# MAGIC - Track token usage for cost monitoring
# MAGIC - Log queries for debugging and improvement
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Implements a `RAGMonitor` class to track requests
# MAGIC 2. Logs latency, success/failure, and errors
# MAGIC 3. Calculates percentile latencies (P50, P95, P99)
# MAGIC 4. Generates a dashboard summary

# COMMAND ----------

# ============================================================
# STRATEGY 6: PERFORMANCE MONITORING
# ============================================================

class RAGMonitor:
    """Simple monitoring for RAG system performance."""

    def __init__(self):
        self.requests = []
        self.errors = []

    def log_request(self, query: str, latency: float, success: bool, error: str = None):
        """Log a request for monitoring."""
        self.requests.append({
            "timestamp": datetime.now(),
            "query": query,
            "latency": latency,
            "success": success,
            "error": error
        })
        if not success:
            self.errors.append({"timestamp": datetime.now(), "error": error})

    def get_dashboard(self) -> Dict:
        """Generate monitoring dashboard metrics."""
        if not self.requests:
            return {"status": "No data"}

        df = pd.DataFrame(self.requests)

        return {
            "total_requests": len(self.requests),
            "success_rate": f"{(df['success'].sum() / len(df) * 100):.1f}%",
            "avg_latency": f"{df['latency'].mean():.2f}s",
            "p50_latency": f"{df['latency'].median():.2f}s",
            "p95_latency": f"{df['latency'].quantile(0.95):.2f}s",
            "p99_latency": f"{df['latency'].quantile(0.99):.2f}s",
            "error_count": len(self.errors),
            "requests_per_minute": len(df) / max(1, (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60)
        }

# Initialize monitor
monitor = RAGMonitor()

# Simulate some requests
print("📊 Simulating RAG Requests for Monitoring\n")

simulation_queries = [
    "What is the authentication API?",
    "How do I deploy models?",
    "What are the data governance policies?",
    "Explain the medallion architecture",
    "What was the payment outage impact?"
]

for query in simulation_queries:
    try:
        start = time.time()
        response = rag_chain.invoke(query)
        latency = time.time() - start
        monitor.log_request(query, latency, success=True)
        print(f"   ✓ {query[:40]}... ({latency:.2f}s)")
    except Exception as e:
        monitor.log_request(query, 0, success=False, error=str(e))
        print(f"   ✗ {query[:40]}... (Error: {str(e)[:30]})")

# Display dashboard
print("\n" + "=" * 60)
print("📈 Performance Dashboard")
print("=" * 60)
dashboard = monitor.get_dashboard()
for metric, value in dashboard.items():
    print(f"   {metric}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Part 10: Cleanup (Optional)
# MAGIC
# MAGIC This section provides cleanup utilities to remove all resources created during the lab.
# MAGIC
# MAGIC ### Resources Created in This Lab
# MAGIC
# MAGIC | Resource Type | Name | Purpose |
# MAGIC |---------------|------|---------|
# MAGIC | **Vector Search Index** | `{catalog}.{schema}.chapter9_vector_index` | Stores embeddings for similarity search |
# MAGIC | **Delta Table (Chunks)** | `{catalog}.{schema}.chapter9_chunks` | Chunked documents with embeddings |
# MAGIC | **Delta Table (Source)** | `{catalog}.{schema}.chapter9_documents` | Original source documents |
# MAGIC | **Schema** | `{catalog}.{schema}` | Container for all lab objects |
# MAGIC
# MAGIC ### When to Clean Up
# MAGIC
# MAGIC - **Keep resources** if you plan to continue experimenting
# MAGIC - **Clean up** if you're done with the lab and want to free resources
# MAGIC - **Note**: The Vector Search endpoint is shared and should NOT be deleted if other indices use it
# MAGIC
# MAGIC **⚠️ Warning:** Cleanup is irreversible. All data will be permanently deleted.
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Deletes the Vector Search index (not the endpoint)
# MAGIC 2. Drops the chunks Delta table
# MAGIC 3. Drops the source documents Delta table
# MAGIC 4. Optionally drops the schema if empty

# COMMAND ----------

# ============================================================
# CLEANUP RESOURCES (OPTIONAL)
# ============================================================

def cleanup_resources(confirm: bool = False):
    """Clean up all resources created during the lab."""

    if not confirm:
        print("⚠️  Cleanup not confirmed. Set confirm=True to proceed.")
        return

    print("🧹 Cleaning up resources...\n")

    # Delete Vector Search index
    try:
        vector_search_client.delete_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=VECTOR_INDEX_PATH
        )
        print(f"   ✓ Deleted index: {VECTOR_INDEX_PATH}")
    except Exception as e:
        print(f"   ✗ Could not delete index: {str(e)[:50]}")

    # Delete tables
    try:
        spark.sql(f"DROP TABLE IF EXISTS {CHUNKS_TABLE_PATH}")
        print(f"   ✓ Deleted table: {CHUNKS_TABLE_PATH}")
    except Exception as e:
        print(f"   ✗ Could not delete table: {str(e)[:50]}")

    try:
        spark.sql(f"DROP TABLE IF EXISTS {SOURCE_TABLE_PATH}")
        print(f"   ✓ Deleted table: {SOURCE_TABLE_PATH}")
    except Exception as e:
        print(f"   ✗ Could not delete table: {str(e)[:50]}")

    # Optionally delete schema
    try:
        spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
        print(f"   ✓ Deleted schema: {CATALOG_NAME}.{SCHEMA_NAME}")
    except Exception as e:
        print(f"   ✗ Could not delete schema: {str(e)[:50]}")

    print("\n✅ Cleanup complete!")

# Uncomment the line below to run cleanup
# cleanup_resources(confirm=True)
print("ℹ️  To clean up resources, uncomment and run: cleanup_resources(confirm=True)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Lab Summary and Key Takeaways
# MAGIC
# MAGIC Congratulations! 🎉 You have successfully completed this hands-on lab on building a scalable Vector Search and Retrieval System.
# MAGIC
# MAGIC ### What You Accomplished
# MAGIC
# MAGIC 1. **Created an Embedding Pipeline**: Chunked documents and generated embeddings using a Databricks-hosted embedding model
# MAGIC 2. **Built a Vector Search Index**: Set up a Delta Sync index and populated it with documents
# MAGIC 3. **Configured a Retriever**: Built a LangChain retriever to power semantic search
# MAGIC 4. **Served LLM and Embedding Models**: Integrated Mosaic AI Model Serving for generation
# MAGIC 5. **Implemented Batching and Context Tuning**: Optimized throughput with parallel processing and context length adjustments
# MAGIC 6. **Profiled RAG Performance**: Identified bottlenecks using component-level profiling
# MAGIC 7. **Applied Chapter 9 Optimization Strategies**: Including context-length tuning, embedding dimensionality adjustments, batching optimization, and index-level performance improvements
# MAGIC
# MAGIC ### Key Optimization Strategies from Chapter 9
# MAGIC
# MAGIC | Strategy | What You Learned | Implementation |
# MAGIC |----------|------------------|----------------|
# MAGIC | **Context-Length Tuning** | Balance quality vs. latency/cost | Adjustable top-k and max context chars |
# MAGIC | **Embedding Dimensionality** | Trade-off between accuracy and speed | Model selection and dimension analysis |
# MAGIC | **Batching Optimization** | Higher throughput for volume workloads | ThreadPoolExecutor with configurable workers |
# MAGIC | **Index-Level Improvements** | Faster vector search with filters | Metadata filtering and index configuration |
# MAGIC | **Query Caching** | Instant responses for repeated queries | LRU cache with hit rate tracking |
# MAGIC | **Hybrid Search** | Better precision for technical terms | Combined semantic + keyword scoring |
# MAGIC
# MAGIC ### Bottleneck Diagnosis Patterns
# MAGIC
# MAGIC | Bottleneck | Symptoms | Root-Cause Adjustments |
# MAGIC |------------|----------|------------------------|
# MAGIC | Slow Vector Search | High search latency, timeouts | Reduce top-k, use metadata filters, optimize index |
# MAGIC | Model Execution Delays | High generation latency | Use smaller models, reduce context, enable streaming |
# MAGIC | Inefficient Batching | Low throughput, high costs | Increase batch size, use async processing |
# MAGIC | Data Retrieval Slowdowns | Slow embedding generation | Cache embeddings, use smaller embedding models |
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC 1. **Scale Testing**: Test with larger document collections (10K+ documents)
# MAGIC 2. **A/B Testing**: Compare different embedding models and chunk sizes
# MAGIC 3. **Production Deployment**: Set up monitoring, alerting, and auto-scaling
# MAGIC 4. **Fine-tuning**: Consider fine-tuning embedding models for domain-specific content
# MAGIC 5. **Evaluation**: Implement systematic evaluation using relevance metrics
# MAGIC
# MAGIC ### Resources
# MAGIC
# MAGIC - [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC - [Mosaic AI Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
# MAGIC - [LangChain Databricks Integration](https://python.langchain.com/docs/integrations/providers/databricks)
# MAGIC - [MLflow Model Registry](https://docs.databricks.com/en/mlflow/model-registry.html)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Thank you for completing this lab!**
# MAGIC
# MAGIC If you have questions or feedback, please reach out to your instructor or refer to the Chapter 9 materials for additional context.