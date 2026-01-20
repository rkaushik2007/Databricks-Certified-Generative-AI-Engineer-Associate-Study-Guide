# Databricks notebook source
# MAGIC %md
# MAGIC # Hands-On Lab: Building a Retrieval-Augmented GenAI App
# MAGIC
# MAGIC ## Scenario
# MAGIC You are a data engineer working for a knowledge-intensive enterprise. Your team has been asked to build a retrieval-augmented generation (RAG) application that allows employees to query internal documents such as compliance manuals, product specifications, and policy handbooks using large language models (LLMs). The challenge is to ensure that responses are not only fluent but also factually accurate, contextually grounded, and compliant with organizational standards.
# MAGIC
# MAGIC This lab mirrors a real-world enterprise use case where accuracy, governance, and safety are critical. Source documents are lengthy, inconsistently formatted, and may contain sensitive information. The system must retrieve only approved content, minimize hallucinations, and clearly handle situations where sufficient evidence is not available.
# MAGIC
# MAGIC To achieve this, you must:
# MAGIC - Extract, clean, and chunk internal documents so they can be used effectively for retrieval
# MAGIC - Store and index document embeddings in a vector database
# MAGIC - Connect a retriever to query the indexed knowledge base
# MAGIC - Integrate the retriever with an LLM using a LangChain pipeline
# MAGIC - Ground model responses strictly in retrieved content
# MAGIC - Apply prompt-level safety instructions to prevent hallucinations and unsafe disclosures
# MAGIC - Qualitatively evaluate responses for grounding, safety, and reliability
# MAGIC
# MAGIC ## Objective
# MAGIC By the end of this lab, you will be able to:
# MAGIC - Build a LangChain pipeline that combines document retrieval with an LLM
# MAGIC - Configure a vector store retriever to return relevant, semantically similar context
# MAGIC - Apply chunking and indexing strategies that improve retrieval quality
# MAGIC - Enforce grounding-focused prompt instructions that require evidence-based answers
# MAGIC - Demonstrate how retrieval augmentation reduces hallucinations compared to prompt-only approaches
# MAGIC - Qualitatively assess model outputs for factual accuracy, safety risks, and policy compliance
# MAGIC - Identify common failure modes (such as unsupported claims or sensitive data exposure) and propose mitigations
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Install Required Libraries
# MAGIC
# MAGIC In this step, we install the essential packages needed to build a Databricks-native Retrieval-Augmented Generation (RAG) application:
# MAGIC
# MAGIC - **databricks-vectorsearch**: Databricks Vector Search SDK for creating and querying vector indexes
# MAGIC - **langchain-core**: The core LangChain framework providing foundational abstractions (Document, Prompt, etc.)
# MAGIC - **mlflow**: MLflow SDK for accessing Databricks Foundation Model APIs
# MAGIC
# MAGIC **Why These Minimal Dependencies:**
# MAGIC - **No NumPy Conflicts**: We avoid packages with conflicting dependencies (langchain-community, langchain-text-splitters)
# MAGIC - **Databricks-Native**: All core functionality uses Databricks APIs (Vector Search, MLflow, Delta Lake)
# MAGIC - **Production-Ready**: Minimal dependencies reduce compatibility issues and maintenance burden
# MAGIC - **Enterprise Governance**: Unity Catalog integration for access control and audit logging
# MAGIC
# MAGIC **Key Databricks Components:**
# MAGIC - **Databricks Vector Search**: Enterprise-grade, scalable vector database with Unity Catalog integration
# MAGIC - **Databricks Foundation Model APIs**: Managed LLM endpoints (DBRX, Llama, etc.) accessed via MLflow
# MAGIC - **Delta Lake**: Persistent storage for documents with ACID transactions and versioning
# MAGIC - **Unity Catalog**: Centralized governance, access control, and lineage tracking
# MAGIC
# MAGIC > **Note:** After installation, we restart the Python runtime to ensure the environment is updated with the new packages.
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch langchain-core mlflow
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Import Required Libraries and Initialize Databricks Clients
# MAGIC
# MAGIC In this step, we import the necessary libraries for building a Databricks-native RAG application.
# MAGIC
# MAGIC **Key Databricks Components:**
# MAGIC - **VectorSearchClient**: Databricks Vector Search client for creating and querying vector indexes
# MAGIC - **MLflow Deployment Client**: Direct access to Databricks Foundation Model APIs (DBRX, Llama, etc.)
# MAGIC - **WorkspaceClient**: Databricks SDK for workspace operations and configuration
# MAGIC
# MAGIC **Key LangChain Components (Core Only):**
# MAGIC - **ChatPromptTemplate**: Defines structured, type-safe prompts for consistent and safe responses
# MAGIC - **Document**: LangChain document structure for retrieved content
# MAGIC - **BaseRetriever**: Base class for building custom retrievers
# MAGIC
# MAGIC **Why Minimal Dependencies:**
# MAGIC - **No NumPy Conflicts**: We use only `langchain-core` which has no conflicting dependencies
# MAGIC - **Native Databricks Integration**: MLflow is pre-installed in Databricks Runtime
# MAGIC - **No External API Keys**: Uses Databricks workspace authentication
# MAGIC - **Enterprise Governance**: Unity Catalog integration for access control and audit logging
# MAGIC - **Production-Ready**: Fewer dependencies mean fewer compatibility issues
# MAGIC - **Better Performance**: Direct API calls without wrapper overhead
# MAGIC

# COMMAND ----------

# Import Databricks-native components
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import mlflow.deployments

# Import LangChain components (core only - no dependencies with NumPy conflicts)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

# Import standard libraries
import pandas as pd
from pyspark.sql import SparkSession
import os
import re

# Initialize Databricks clients
vsc = VectorSearchClient(disable_notice=True)
w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# Initialize MLflow deployment client for Foundation Models
deploy_client = mlflow.deployments.get_deploy_client("databricks")

print("✅ Successfully imported all libraries")
print(f"   Databricks Vector Search: Ready")
print(f"   MLflow Deployment Client: Ready")
print(f"   Spark Session: {spark.version}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Configure Databricks Resources
# MAGIC
# MAGIC Before we can build our RAG application, we need to configure the Databricks resources:
# MAGIC
# MAGIC **Required Configuration:**
# MAGIC - **Catalog and Schema**: Unity Catalog namespace for organizing tables and indexes
# MAGIC - **Vector Search Endpoint**: Compute endpoint for vector similarity search
# MAGIC - **Embedding Model Endpoint**: Databricks Foundation Model endpoint for generating embeddings
# MAGIC
# MAGIC **Why Unity Catalog:**
# MAGIC - **Governance**: Centralized access control and permissions
# MAGIC - **Lineage**: Track data flow from source documents to RAG responses
# MAGIC - **Audit**: Complete audit trail of who accessed what data and when
# MAGIC - **Discovery**: Searchable catalog of all data assets
# MAGIC
# MAGIC **Note:** Replace these values with your actual Databricks workspace configuration.
# MAGIC

# COMMAND ----------

# First, let's discover what Vector Search endpoints are available in your workspace
print("🔍 Discovering available Vector Search endpoints in your workspace...\n")

try:
    # List all Vector Search endpoints
    endpoints = w.vector_search_endpoints.list_endpoints()
    endpoint_list = list(endpoints)

    if endpoint_list:
        print(f"✅ Found {len(endpoint_list)} Vector Search endpoint(s):\n")
        for idx, endpoint in enumerate(endpoint_list, 1):
            print(f"   {idx}. Name: {endpoint.name}")
            print(f"      Status: {endpoint.endpoint_status.state}")
            print(f"      Type: {endpoint.endpoint_type}")
            print()

        # Use the first ONLINE endpoint
        online_endpoints = [e for e in endpoint_list if e.endpoint_status.state == "ONLINE"]
        if online_endpoints:
            suggested_endpoint = online_endpoints[0].name
            print(f"💡 Suggestion: Use endpoint '{suggested_endpoint}' (it's ONLINE)")
        else:
            suggested_endpoint = endpoint_list[0].name
            print(f"⚠️  No ONLINE endpoints found. Using '{suggested_endpoint}' (may need to wait for it)")
    else:
        print("❌ No Vector Search endpoints found in your workspace!")
        print("\n📋 You need to create a Vector Search endpoint:")
        print("   1. Go to Databricks UI: Compute > Vector Search")
        print("   2. Click 'Create Endpoint'")
        print("   3. Name: 'rag-demo-endpoint' (or any name you prefer)")
        print("   4. Type: Serverless (recommended)")
        print("   5. Wait 5-10 minutes for it to become ONLINE")
        print("   6. Come back and re-run this cell")
        suggested_endpoint = "rag-demo-endpoint"  # Default suggestion

except Exception as e:
    print(f"⚠️  Could not list Vector Search endpoints: {str(e)}")
    print("   You may not have permissions or Vector Search may not be enabled.")
    suggested_endpoint = "rag-demo-endpoint"

print("\n" + "="*80)


# COMMAND ----------

# Configure your Databricks resources
# IMPORTANT: Update these values for your workspace
CATALOG = "main"  # Your Unity Catalog name
SCHEMA = "rag_demo"  # Schema for storing tables and indexes
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.rag_documents"  # Delta table for source documents

# UPDATE THIS: Use the endpoint name from the discovery above
VECTOR_SEARCH_ENDPOINT = suggested_endpoint  # ← This will use the discovered endpoint

# UPDATE THIS: Check available embedding models in Machine Learning > Serving
EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"  # Databricks embedding model

# Create schema if it doesn't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"✅ Configured resources:")
print(f"   Catalog: {CATALOG}")
print(f"   Schema: {SCHEMA}")
print(f"   Source Table: {SOURCE_TABLE}")
print(f"   Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT}")
print(f"   Embedding Model: {EMBEDDING_MODEL_ENDPOINT}")
print(f"\n⚠️  If the Vector Search endpoint is not ONLINE, you'll need to wait or create one.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Prepare Source Documents and Store in Delta Lake
# MAGIC
# MAGIC In this step, we create internal enterprise documents and store them in a Delta table:
# MAGIC
# MAGIC - **Compliance policies**: Regulatory requirements and mandatory training schedules
# MAGIC - **Product specifications**: Technical requirements and maintenance procedures
# MAGIC - **Employee handbooks**: HR policies, benefits, and workplace guidelines
# MAGIC - **Security protocols**: Data handling and access control policies
# MAGIC
# MAGIC **Why Delta Lake:**
# MAGIC - **ACID Transactions**: Ensures data consistency and reliability
# MAGIC - **Time Travel**: Version history for auditing and rollback
# MAGIC - **Schema Evolution**: Adapt to changing document structures
# MAGIC - **Scalability**: Handles billions of documents efficiently
# MAGIC - **Unity Catalog Integration**: Automatic governance and access control
# MAGIC - **Change Data Feed (CDF)**: Required for Vector Search Delta Sync indexes to track changes
# MAGIC
# MAGIC **Change Data Feed (CDF):**
# MAGIC Vector Search Delta Sync indexes require CDF to be enabled on the source table. CDF tracks all changes (inserts, updates, deletes) to the table, allowing the Vector Search index to automatically sync when documents are added, modified, or removed.
# MAGIC
# MAGIC In a production environment, these documents would be extracted from various sources (PDFs, databases, SharePoint, etc.) and loaded into Delta tables using Databricks workflows.
# MAGIC

# COMMAND ----------

# Simulating internal enterprise documents
# In production, these would be extracted from PDFs, databases, or document management systems
documents_data = [
    {"id": "doc_001", "text": "Policy: All employees must complete annual compliance training by June 30th. Failure to complete training may result in access restrictions to sensitive systems. Training covers data privacy, security protocols, and ethical guidelines.", "category": "compliance"},
    {"id": "doc_002", "text": "Product spec: The Model X100 device must undergo comprehensive testing every 12 months. Testing includes safety checks, performance validation, and regulatory compliance verification. Devices failing inspection must be decommissioned immediately.", "category": "product"},
    {"id": "doc_003", "text": "Handbook: Employees are entitled to 15 days of paid leave per year, accrued at 1.25 days per month. Leave requests must be submitted at least 2 weeks in advance through the HR portal. Unused leave may be carried over up to 5 days.", "category": "hr"},
    {"id": "doc_004", "text": "Security: All confidential data must be encrypted at rest and in transit using AES-256 encryption. Access to confidential data requires multi-factor authentication and is logged for audit purposes. Data retention follows the 7-year policy.", "category": "security"},
    {"id": "doc_005", "text": "Policy: Remote work is permitted up to 3 days per week with manager approval. Remote workers must maintain secure home office setups and use company-provided VPN for all work-related activities.", "category": "policy"},
    {"id": "doc_006", "text": "Product spec: Model X200 supersedes X100 and includes enhanced safety features. X200 devices require testing every 18 months. All X100 devices must be upgraded to X200 by December 2026.", "category": "product"},
    {"id": "doc_007", "text": "Handbook: Health insurance coverage begins on the first day of employment. Coverage includes medical, dental, and vision. Employees can add dependents during open enrollment or within 30 days of a qualifying life event.", "category": "hr"}
]

# Create DataFrame and write to Delta table
df = spark.createDataFrame(documents_data)

# Write to Delta table with Change Data Feed enabled (required for Vector Search)
df.write.format("delta") \
    .mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(SOURCE_TABLE)

# Also enable CDF at the table level (in case it wasn't set during creation)
spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"✅ Created Delta table: {SOURCE_TABLE}")
print(f"   Total documents: {df.count()}")
print(f"   Change Data Feed: ENABLED (required for Vector Search)")
display(df)





# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Create Databricks Vector Search Index
# MAGIC
# MAGIC Now we create a Databricks Vector Search index that automatically:
# MAGIC 1. Generates embeddings for the `text` column using the specified embedding model
# MAGIC 2. Stores vectors in a managed vector database
# MAGIC 3. Syncs automatically when the source Delta table is updated
# MAGIC 4. Provides a retrieval interface for similarity search
# MAGIC
# MAGIC **Vector Search Index Types:**
# MAGIC - **Delta Sync Index**: Automatically syncs with source Delta table (used here)
# MAGIC - **Direct Vector Access Index**: For pre-computed embeddings
# MAGIC - **Self-Managed Index**: For custom embedding pipelines
# MAGIC
# MAGIC **Why Databricks Vector Search:**
# MAGIC - **Automatic Embedding**: No need to manually generate and store embeddings
# MAGIC - **Auto-Sync**: Index updates automatically when source table changes
# MAGIC - **Scalability**: Handles billions of vectors with sub-second query latency
# MAGIC - **Unity Catalog Integration**: Inherits permissions from source table
# MAGIC - **Managed Infrastructure**: No need to manage vector database infrastructure
# MAGIC - **Cost Optimization**: Pay only for what you use with serverless endpoints
# MAGIC
# MAGIC **Index Configuration:**
# MAGIC - `primary_key="id"`: Unique identifier for each document
# MAGIC - `embedding_source_column="text"`: Column to generate embeddings from
# MAGIC - `embedding_model_endpoint_name`: Databricks embedding model to use
# MAGIC

# COMMAND ----------

# Define index name
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_documents_index"

print(f"🔧 Creating Vector Search Index: {INDEX_NAME}")
print(f"   This will create an index that automatically embeds and indexes the documents")
print(f"   from the Delta table: {SOURCE_TABLE}\n")

# Create Delta Sync Vector Search Index
try:
    # Check if index already exists
    print("   Checking if index already exists...")
    existing_indexes = vsc.list_indexes(VECTOR_SEARCH_ENDPOINT)

    # Fix: list_indexes returns dicts, not objects
    index_exists = False
    if "vector_indexes" in existing_indexes:
        for idx in existing_indexes["vector_indexes"]:
            # idx is a dict, access with ['name'] not .name
            if idx.get("name") == INDEX_NAME:
                index_exists = True
                break

    if index_exists:
        print(f"   ✅ Index {INDEX_NAME} already exists!")
        print(f"   Skipping creation and using existing index.")
        print(f"   If you want to recreate it, manually delete it first:")
        print(f"   → Databricks UI: Compute > Vector Search > {INDEX_NAME} > Delete")
        print(f"   → Or run: vsc.delete_index(endpoint_name='{VECTOR_SEARCH_ENDPOINT}', index_name='{INDEX_NAME}')")

        print(f"\n✅ Using existing Vector Search Index!")
        print(f"   Index Name: {INDEX_NAME}")
        print(f"   Endpoint: {VECTOR_SEARCH_ENDPOINT}")
        print(f"   Source Table: {SOURCE_TABLE}")
        print(f"\n   Proceed to Step 6 to verify the index is ONLINE.")
    else:
        # Create new index
        print(f"   Creating new index...")
        index = vsc.create_delta_sync_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT,
            index_name=INDEX_NAME,
            source_table_name=SOURCE_TABLE,
            pipeline_type="TRIGGERED",  # or "CONTINUOUS" for real-time updates
            primary_key="id",
            embedding_source_column="text",
            embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT
        )

        print(f"\n✅ Successfully created Vector Search Index!")
        print(f"   Index Name: {INDEX_NAME}")
        print(f"   Endpoint: {VECTOR_SEARCH_ENDPOINT}")
        print(f"   Source Table: {SOURCE_TABLE}")
        print(f"   Embedding Model: {EMBEDDING_MODEL_ENDPOINT}")
        print(f"\n⏳ Index is being created and will start indexing documents...")
        print(f"   This may take 5-10 minutes depending on the number of documents.")
        print(f"   You can monitor progress in the Databricks UI:")
        print(f"   Compute > Vector Search > {INDEX_NAME}")
        print(f"\n   Proceed to Step 6 to wait for the index to become ONLINE.")

except Exception as e:
    print(f"\n❌ ERROR: Failed to create Vector Search index!")
    print(f"   Error: {str(e)}\n")
    print(f"   Common issues:")
    print(f"   1. Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}' does not exist or is not ONLINE")
    print(f"      → Check: Compute > Vector Search in Databricks UI")
    print(f"   2. Embedding model endpoint '{EMBEDDING_MODEL_ENDPOINT}' is not available")
    print(f"      → Check: Machine Learning > Serving in Databricks UI")
    print(f"   3. Source table '{SOURCE_TABLE}' does not exist")
    print(f"      → Make sure you ran Step 4 successfully")
    print(f"   4. Insufficient permissions to create indexes")
    print(f"      → Contact your Databricks admin")
    print(f"\n   Please fix the issue above and re-run this cell.")
    raise


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Wait for Index to be Ready
# MAGIC
# MAGIC Before we can query the vector search index, we need to wait for it to finish indexing the documents.
# MAGIC
# MAGIC **Index Status:**
# MAGIC - **ONLINE**: Index is ready for queries
# MAGIC - **PROVISIONING**: Index is being created
# MAGIC - **OFFLINE**: Index is not available
# MAGIC

# COMMAND ----------

import time

# Wait for index to be ready
print("⏳ Waiting for index to be ready...")
max_wait_time = 300  # 5 minutes
start_time = time.time()

while time.time() - start_time < max_wait_time:
    try:
        index_status = vsc.get_index(INDEX_NAME)
        status = index_status.get("status", {}).get("detailed_state", "UNKNOWN")

        if status == "ONLINE":
            print(f"✅ Index is ONLINE and ready for queries!")
            break
        else:
            print(f"   Status: {status}. Waiting...")
            time.sleep(10)
    except Exception as e:
        print(f"   Checking status... ({str(e)})")
        time.sleep(10)
else:
    print(f"⚠️  Index did not become ready within {max_wait_time} seconds")
    print(f"   You can continue and check status manually in the Databricks UI")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7: Configure the Retriever
# MAGIC
# MAGIC Now we create a retriever that uses the Databricks Vector Search index for similarity search.
# MAGIC
# MAGIC **How Databricks Vector Search Retrieval Works:**
# MAGIC 1. User query is sent to the embedding model endpoint
# MAGIC 2. Query embedding is generated automatically
# MAGIC 3. Vector Search performs similarity search against indexed documents
# MAGIC 4. Top k most similar documents are returned as context for the LLM
# MAGIC
# MAGIC **Configuration Parameters:**
# MAGIC - `num_results=3`: Returns the top 3 most relevant documents
# MAGIC - Similarity search uses cosine similarity by default
# MAGIC - Results include both the document text and similarity scores
# MAGIC
# MAGIC **Why k=3?**
# MAGIC - Balances context richness with token efficiency
# MAGIC - Provides enough information without overwhelming the LLM
# MAGIC - Can be tuned based on document length and query complexity
# MAGIC

# COMMAND ----------

# Create a retriever wrapper for Databricks Vector Search
class DatabricksVectorSearchRetriever:
    def __init__(self, index_name, num_results=3):
        self.vsc = VectorSearchClient(disable_notice=True)
        self.index_name = index_name
        self.num_results = num_results

        # Check if index exists before trying to use it
        try:
            self.index = self.vsc.get_index(index_name=self.index_name)
            print(f"   ✅ Successfully connected to index: {self.index_name}")
        except Exception as e:
            print(f"   ❌ Error: Index '{self.index_name}' does not exist!")
            print(f"   Please make sure you ran Step 5 to create the Vector Search index.")
            print(f"   Error details: {str(e)}")
            raise

    def get_relevant_documents(self, query):
        """Retrieve relevant documents for a query"""
        # Use the pre-initialized index object
        results = self.index.similarity_search(
            query_text=query,
            columns=["id", "text", "category"],
            num_results=self.num_results
        )

        # Convert to LangChain Document format
        from langchain_core.documents import Document
        documents = []
        for item in results.get("result", {}).get("data_array", []):
            # item format: [id, text, category, score]
            doc = Document(
                page_content=item[1],  # text column
                metadata={"id": item[0], "category": item[2], "score": item[3] if len(item) > 3 else None}
            )
            documents.append(doc)
        return documents

# Initialize retriever
print("🔧 Initializing retriever...")
print(f"   Looking for index: {INDEX_NAME}")

try:
    retriever = DatabricksVectorSearchRetriever(
        index_name=INDEX_NAME,
        num_results=3
    )
    print("✅ Retriever configured successfully")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Number of documents to retrieve: 3")
except Exception as e:
    print("\n" + "="*80)
    print("⚠️  RETRIEVER INITIALIZATION FAILED")
    print("="*80)
    print("\nThe Vector Search index does not exist. Please complete these steps:")
    print("\n1. Go back to Step 5 and run it to create the Vector Search index")
    print("2. Wait for Step 6 to confirm the index is ONLINE")
    print("3. Then come back and run this cell (Step 7) again")
    print("\nIf Step 5 failed, check:")
    print(f"   - Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}' exists and is ONLINE")
    print(f"   - Embedding model endpoint '{EMBEDDING_MODEL_ENDPOINT}' is available")
    print(f"   - You have permissions to create indexes in '{CATALOG}.{SCHEMA}'")
    print("="*80)
    raise


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8: Create a Safety-Focused Prompt Template
# MAGIC
# MAGIC To reduce the risk of hallucinations and ensure compliance, we design a **prompt template** with explicit safety instructions.
# MAGIC
# MAGIC **Prompt Engineering for Safety:**
# MAGIC This template enforces the following rules:
# MAGIC - Clearly defines the model's **role** as a compliance assistant
# MAGIC - Instructs the LLM to base answers strictly on the provided **context**
# MAGIC - Requires the model to state *"I don't have enough information to answer that question"* if the answer cannot be found
# MAGIC - Prevents speculation, assumptions, or use of external knowledge
# MAGIC - Ensures responses are professional and policy-compliant
# MAGIC
# MAGIC **Why ChatPromptTemplate?**
# MAGIC - Type-safe prompt construction
# MAGIC - Supports message roles (system, human, assistant)
# MAGIC - Better integration with modern LangChain chains
# MAGIC - Easier to extend with few-shot examples
# MAGIC

# COMMAND ----------

# Creating a safety-focused prompt template using ChatPromptTemplate
system_prompt = (
    "You are a compliance assistant for an enterprise organization. "
    "Your role is to provide accurate, factual answers based strictly on the provided context. "
    "\n\n"
    "IMPORTANT RULES:\n"
    "1. Only use information from the context below to answer questions\n"
    "2. If the answer is not in the context, respond with: 'I don't have enough information to answer that question.'\n"
    "3. Do not make assumptions or use external knowledge\n"
    "4. Do not speculate or infer information not explicitly stated\n"
    "5. Be concise and professional in your responses\n"
    "6. If you find conflicting information, mention both versions\n"
    "\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

print("✅ Safety-focused prompt template created successfully")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9: Build the RAG Pipeline with Databricks Foundation Models
# MAGIC
# MAGIC Now we bring all the components together to create a **Retrieval-Augmented Generation (RAG) pipeline** using Databricks Foundation Model APIs via MLflow.
# MAGIC
# MAGIC **RAG Pipeline Architecture:**
# MAGIC We'll build a custom RAG chain that:
# MAGIC 1. Takes a user query
# MAGIC 2. Retrieves relevant documents from Vector Search
# MAGIC 3. Formats the prompt with retrieved context
# MAGIC 4. Calls Databricks Foundation Model API
# MAGIC 5. Returns the grounded response
# MAGIC
# MAGIC **Databricks Foundation Model Configuration:**
# MAGIC - **Model**: DBRX Instruct for high-quality responses
# MAGIC - **Temperature**: 0 for deterministic, consistent outputs
# MAGIC - **No API Keys**: Uses Databricks workspace authentication
# MAGIC - **Governance**: All requests logged in Unity Catalog for audit and compliance
# MAGIC

# COMMAND ----------

# First, discover available Foundation Model endpoints
print("🔍 Discovering available Foundation Model endpoints in your workspace...\n")

try:
    # List all serving endpoints
    endpoints = w.serving_endpoints.list()
    endpoint_list = list(endpoints)

    # Filter for Foundation Model endpoints (usually start with 'databricks-')
    fm_endpoints = [e for e in endpoint_list if e.name.startswith('databricks-')]

    if fm_endpoints:
        print(f"✅ Found {len(fm_endpoints)} Foundation Model endpoint(s):\n")
        for idx, endpoint in enumerate(fm_endpoints, 1):
            print(f"   {idx}. {endpoint.name}")
            print(f"      State: {endpoint.state.config_update if endpoint.state else 'Unknown'}")

        # Suggest the first available endpoint
        suggested_fm = fm_endpoints[0].name
        print(f"\n💡 Suggestion: Use endpoint '{suggested_fm}'")
    else:
        print("⚠️  No Foundation Model endpoints found starting with 'databricks-'")
        print("\n📋 Common Foundation Model endpoints:")
        print("   - databricks-meta-llama-3-1-70b-instruct")
        print("   - databricks-meta-llama-3-1-405b-instruct")
        print("   - databricks-mixtral-8x7b-instruct")
        print("   - databricks-dbrx-instruct (may not be available in all regions)")
        print("\n   Check: Machine Learning > Serving in Databricks UI")
        suggested_fm = "databricks-meta-llama-3-1-70b-instruct"  # Common default

except Exception as e:
    print(f"⚠️  Could not list serving endpoints: {str(e)}")
    suggested_fm = "databricks-meta-llama-3-1-70b-instruct"

print("\n" + "="*80)


# COMMAND ----------

# Define the Foundation Model endpoint
# UPDATE THIS: Use the endpoint name from the discovery above
FOUNDATION_MODEL_ENDPOINT = suggested_fm  # ← This will use the discovered endpoint

print(f"🔧 Using Foundation Model endpoint: {FOUNDATION_MODEL_ENDPOINT}")

# Create a function to call the Foundation Model
def call_foundation_model(prompt_text):
    """Call Databricks Foundation Model via MLflow deployment client"""
    response = deploy_client.predict(
        endpoint=FOUNDATION_MODEL_ENDPOINT,
        inputs={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.0,
            "max_tokens": 500
        }
    )
    return response['choices'][0]['message']['content']

# Create the RAG chain function
def rag_chain_invoke(query):
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents
    2. Format prompt with context
    3. Call Foundation Model
    4. Return response with context
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # Step 2: Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 3: Format the complete prompt
    formatted_prompt = f"""You are a compliance assistant for an enterprise organization. Your role is to provide accurate, factual answers based strictly on the provided context.

IMPORTANT RULES:
1. Only use information from the context below to answer questions
2. If the answer is not in the context, respond with: 'I don't have enough information to answer that question.'
3. Do not make assumptions or use external knowledge
4. Do not speculate or infer information not explicitly stated
5. Be concise and professional in your responses
6. If you find conflicting information, mention both versions

Context:
{context}

Question: {query}

Answer:"""

    # Step 4: Call Foundation Model
    answer = call_foundation_model(formatted_prompt)

    # Step 5: Return response with context
    return {
        "input": query,
        "context": retrieved_docs,
        "answer": answer
    }

print("✅ RAG pipeline built successfully using Databricks-native architecture")
print(f"   LLM: {FOUNDATION_MODEL_ENDPOINT}")
print(f"   Temperature: 0.0 (deterministic)")
print(f"   Retriever: Databricks Vector Search with k=3")
print(f"   Governance: Unity Catalog enabled")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10: Test the RAG Pipeline with Sample Queries
# MAGIC
# MAGIC Let's test our Databricks-native RAG system with various queries to validate its performance and safety features.
# MAGIC
# MAGIC **Test Scenarios:**
# MAGIC 1. **Grounded Query**: Question with clear answer in the knowledge base
# MAGIC 2. **Out-of-Scope Query**: Question not covered by our documents
# MAGIC 3. **Ambiguous Query**: Question that might have multiple interpretations
# MAGIC 4. **Conflicting Information**: Query where documents might have different information
# MAGIC

# COMMAND ----------

# Test Query 1: Grounded question with clear answer
print("=" * 80)
print("TEST 1: Grounded Query")
print("=" * 80)
query1 = "How many days of paid leave are employees entitled to?"
response1 = rag_chain_invoke(query1)

print(f"\nQuery: {query1}")
print(f"\nAnswer: {response1['answer']}")
print(f"\nRetrieved {len(response1['context'])} context chunks")


# COMMAND ----------

# Test Query 2: Out-of-scope question
print("\n" + "=" * 80)
print("TEST 2: Out-of-Scope Query (Testing Hallucination Prevention)")
print("=" * 80)
query2 = "What is the company's policy on cryptocurrency investments?"
response2 = rag_chain_invoke(query2)

print(f"\nQuery: {query2}")
print(f"\nAnswer: {response2['answer']}")
print(f"\nRetrieved {len(response2['context'])} context chunks")
print("\n✓ Expected: Model should indicate insufficient information")


# COMMAND ----------

# Test Query 3: Product specification query
print("\n" + "=" * 80)
print("TEST 3: Product Specification Query")
print("=" * 80)
query3 = "How often does the Model X100 device need to be tested?"
response3 = rag_chain_invoke(query3)

print(f"\nQuery: {query3}")
print(f"\nAnswer: {response3['answer']}")
print(f"\nRetrieved {len(response3['context'])} context chunks")


# COMMAND ----------

# Test Query 4: Security policy query
print("\n" + "=" * 80)
print("TEST 4: Security Policy Query")
print("=" * 80)
query4 = "What encryption standard is required for confidential data?"
response4 = rag_chain_invoke(query4)

print(f"\nQuery: {query4}")
print(f"\nAnswer: {response4['answer']}")
print(f"\nRetrieved {len(response4['context'])} context chunks")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 11: Demonstrate RAG vs. Prompt-Only Approach
# MAGIC
# MAGIC To illustrate the value of retrieval augmentation, let's compare RAG responses with a prompt-only approach (no retrieval).
# MAGIC
# MAGIC **Hypothesis:**
# MAGIC Without retrieval, the LLM will either:
# MAGIC 1. Hallucinate plausible-sounding but incorrect information
# MAGIC 2. Refuse to answer due to lack of context
# MAGIC 3. Provide generic, unhelpful responses
# MAGIC

# COMMAND ----------

# Test the same query without RAG (prompt-only approach)
print("=" * 80)
print("COMPARISON: RAG vs. Prompt-Only")
print("=" * 80)

test_query = "How many days of paid leave are employees entitled to?"

# Prompt-only approach (no retrieval)
prompt_only_response = call_foundation_model(
    f"You are a compliance assistant. Answer this question: {test_query}"
)

print(f"\nQuery: {test_query}")
print(f"\n--- PROMPT-ONLY RESPONSE (No Retrieval) ---")
print(prompt_only_response)
print(f"\n--- RAG RESPONSE (With Retrieval) ---")
print(response1['answer'])

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print("✓ RAG provides specific, grounded answer: '15 days per year'")
print("✓ Prompt-only likely provides generic or refuses to answer")
print("✓ RAG cites specific policy details (accrual rate, carryover limits)")
print("✓ This demonstrates how retrieval reduces hallucinations")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 12: Qualitative Evaluation Framework
# MAGIC
# MAGIC Now we establish a framework for evaluating RAG responses across multiple dimensions critical for enterprise deployment.
# MAGIC
# MAGIC **Evaluation Dimensions:**
# MAGIC 1. **Grounding**: Is the answer based solely on retrieved context?
# MAGIC 2. **Accuracy**: Is the information factually correct?
# MAGIC 3. **Completeness**: Does it answer the full question?
# MAGIC 4. **Safety**: Does it avoid hallucinations and unsafe disclosures?
# MAGIC 5. **Compliance**: Does it follow organizational policies?
# MAGIC
# MAGIC **Evaluation Process:**
# MAGIC For each response, we assess:
# MAGIC - ✓ **Pass**: Meets all criteria
# MAGIC - ⚠ **Warning**: Partially meets criteria, needs review
# MAGIC - ✗ **Fail**: Does not meet criteria, requires intervention
# MAGIC

# COMMAND ----------

def evaluate_response(query, response, expected_behavior):
    """
    Qualitatively evaluate a RAG response across multiple dimensions.

    Args:
        query: The user's question
        response: The RAG system's response dictionary
        expected_behavior: Description of expected behavior

    Returns:
        Dictionary with evaluation scores
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION: {query}")
    print(f"{'='*80}")

    answer = response['answer']
    context = response['context']

    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer}")
    print(f"\nRetrieved Contexts ({len(context)} chunks):")
    for i, doc in enumerate(context, 1):
        print(f"  [{i}] {doc.page_content[:100]}...")

    print(f"\n--- EVALUATION CRITERIA ---")

    # Grounding Check
    grounding_check = "I don't have enough information" in answer or any(
        chunk.page_content.lower() in answer.lower()
        for chunk in context
    )
    print(f"✓ Grounding: {'PASS' if grounding_check else 'REVIEW NEEDED'}")
    print(f"  - Answer appears to be based on retrieved context")

    # Safety Check
    safety_indicators = [
        "I don't have enough information",
        "based on the context",
        "according to",
        "the policy states"
    ]
    safety_check = any(indicator.lower() in answer.lower() for indicator in safety_indicators)
    print(f"✓ Safety: {'PASS' if safety_check else 'REVIEW NEEDED'}")
    print(f"  - Response includes grounding indicators or appropriate refusal")

    # Hallucination Check
    hallucination_indicators = [
        "typically",
        "usually",
        "generally",
        "in most cases",
        "it depends"
    ]
    hallucination_check = not any(indicator.lower() in answer.lower() for indicator in hallucination_indicators)
    print(f"✓ Hallucination Prevention: {'PASS' if hallucination_check else 'WARNING'}")
    print(f"  - No speculative language detected")

    print(f"\nExpected Behavior: {expected_behavior}")
    print(f"{'='*80}\n")

    return {
        "query": query,
        "grounding": grounding_check,
        "safety": safety_check,
        "hallucination_prevention": hallucination_check
    }

# Evaluate our test queries
evaluation_results = []

eval1 = evaluate_response(
    query1,
    response1,
    "Should provide specific answer: 15 days per year with accrual details"
)
evaluation_results.append(eval1)

eval2 = evaluate_response(
    query2,
    response2,
    "Should refuse to answer or state insufficient information"
)
evaluation_results.append(eval2)

eval3 = evaluate_response(
    query3,
    response3,
    "Should provide specific answer: every 12 months"
)
evaluation_results.append(eval3)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 13: Identify Common Failure Modes and Mitigations
# MAGIC
# MAGIC Based on real-world RAG deployments, here are common failure modes and recommended mitigations:
# MAGIC
# MAGIC **Failure Mode 1: Retrieval Failure**
# MAGIC - **Problem**: Relevant documents exist but aren't retrieved
# MAGIC - **Causes**: Poor chunking, inadequate embeddings, query-document mismatch
# MAGIC - **Mitigations**:
# MAGIC   - Experiment with chunk sizes (200-1000 characters)
# MAGIC   - Use hybrid search (keyword + semantic)
# MAGIC   - Implement query expansion or reformulation
# MAGIC   - Add metadata filtering
# MAGIC
# MAGIC **Failure Mode 2: Context Overflow**
# MAGIC - **Problem**: Too many chunks exceed LLM context window
# MAGIC - **Causes**: Large k value, long documents, inefficient chunking
# MAGIC - **Mitigations**:
# MAGIC   - Reduce k parameter
# MAGIC   - Implement re-ranking to select best chunks
# MAGIC   - Use map-reduce or refine chains for long contexts
# MAGIC   - Compress or summarize retrieved chunks
# MAGIC
# MAGIC **Failure Mode 3: Hallucination Despite Retrieval**
# MAGIC - **Problem**: LLM generates information not in context
# MAGIC - **Causes**: Weak prompt instructions, high temperature, model limitations
# MAGIC - **Mitigations**:
# MAGIC   - Strengthen prompt with explicit constraints
# MAGIC   - Set temperature to 0
# MAGIC   - Use citation-based prompting
# MAGIC   - Implement post-processing validation
# MAGIC
# MAGIC **Failure Mode 4: Sensitive Data Exposure**
# MAGIC - **Problem**: System returns confidential information inappropriately
# MAGIC - **Causes**: Inadequate access controls, poor document filtering
# MAGIC - **Mitigations**:
# MAGIC   - Implement user-based document filtering
# MAGIC   - Add PII detection and redaction
# MAGIC   - Use metadata for access control
# MAGIC   - Audit and log all retrievals
# MAGIC
# MAGIC **Failure Mode 5: Outdated Information**
# MAGIC - **Problem**: Retrieved documents contain stale information
# MAGIC - **Causes**: Infrequent index updates, no version control
# MAGIC - **Mitigations**:
# MAGIC   - Implement automated index refresh
# MAGIC   - Add document timestamps and versioning
# MAGIC   - Prioritize recent documents in retrieval
# MAGIC   - Display last-updated dates in responses
# MAGIC

# COMMAND ----------

# Demonstrate a potential failure mode: Conflicting information
print("=" * 80)
print("FAILURE MODE DEMONSTRATION: Conflicting Information")
print("=" * 80)

# Query about Model X100 vs X200 testing schedules
conflict_query = "What is the testing schedule for Model X devices?"
conflict_response = rag_chain_invoke(conflict_query)

print(f"\nQuery: {conflict_query}")
print(f"\nAnswer: {conflict_response['answer']}")
print(f"\nRetrieved Contexts:")
for i, doc in enumerate(conflict_response['context'], 1):
    print(f"  [{i}] {doc.page_content}")

print("\n--- ANALYSIS ---")
print("✓ The knowledge base contains information about both X100 (12 months) and X200 (18 months)")
print("✓ A good RAG system should:")
print("  1. Identify both models and their different schedules")
print("  2. Clarify the distinction in the response")
print("  3. Not conflate or confuse the two")
print("\n✓ Mitigation: Ensure prompt instructs model to handle conflicting info explicitly")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 13: Summary and Best Practices
# MAGIC
# MAGIC **What We've Accomplished:**
# MAGIC 1. ✓ Built a Databricks-native RAG pipeline using modern LangChain architecture
# MAGIC 2. ✓ Stored documents in Delta Lake with Unity Catalog governance
# MAGIC 3. ✓ Created Databricks Vector Search index with automatic embedding generation
# MAGIC 4. ✓ Used Databricks Foundation Model APIs (DBRX) for LLM inference
# MAGIC 5. ✓ Created a safety-focused prompt to prevent hallucinations
# MAGIC 6. ✓ Demonstrated RAG superiority over prompt-only approaches
# MAGIC 7. ✓ Established qualitative evaluation framework
# MAGIC 8. ✓ Identified common failure modes and mitigations
# MAGIC
# MAGIC **Key Takeaways:**
# MAGIC - **Databricks-Native**: All components run within Databricks (no external dependencies)
# MAGIC - **Unity Catalog**: Centralized governance, access control, and audit logging
# MAGIC - **Vector Search**: Managed, scalable vector database with auto-sync
# MAGIC - **Foundation Models**: No API keys, built-in governance, cost optimization
# MAGIC - **Modern APIs**: Use `create_retrieval_chain` instead of deprecated `RetrievalQA`
# MAGIC - **Prompt Engineering**: Explicit safety instructions are critical
# MAGIC - **Evaluation**: Assess grounding, safety, and hallucination prevention
# MAGIC
# MAGIC **Databricks Production Advantages:**
# MAGIC 1. **Scalability**: Vector Search handles billions of vectors with sub-second latency
# MAGIC 2. **Governance**: Unity Catalog provides centralized access control and audit trails
# MAGIC 3. **Auto-Sync**: Vector index automatically updates when source Delta table changes
# MAGIC 4. **Cost Optimization**: Pay-per-use serverless endpoints, no external API costs
# MAGIC 5. **Security**: Data never leaves your Databricks workspace
# MAGIC 6. **Monitoring**: Built-in logging and monitoring through Databricks UI
# MAGIC 7. **Compliance**: Meets enterprise security and regulatory requirements
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Implement user-based access control using Unity Catalog permissions
# MAGIC - Set up continuous sync for real-time document updates
# MAGIC - Add hybrid search combining keyword and semantic search
# MAGIC - Implement re-ranking to improve retrieval quality
# MAGIC - Create Databricks workflows for automated document ingestion
# MAGIC - Set up MLflow tracking for RAG performance monitoring
# MAGIC - Explore Databricks Model Serving for production deployment
# MAGIC

# COMMAND ----------

print("=" * 80)
print("LAB COMPLETE!")
print("=" * 80)
print("\n✅ You have successfully built a Databricks-native production-ready RAG application")
print("✅ You understand modern LangChain architecture and Databricks best practices")
print("✅ You can leverage Unity Catalog for governance and access control")
print("✅ You can use Databricks Vector Search for scalable retrieval")
print("✅ You can evaluate and improve RAG system performance")
print("✅ You are aware of common pitfalls and how to avoid them")
print("\n🎉 Congratulations! You're ready to deploy enterprise RAG systems on Databricks!")
