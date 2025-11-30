# Databricks notebook source
# MAGIC %md
# MAGIC # Hands-On Lab: Evaluating and Monitoring LLM Performance
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Scenario
# MAGIC
# MAGIC You are an **LLM operations engineer** responsible for managing a production question-answering system used by internal employees across your organization. The system relies on a large language model deployed through **Databricks Model Serving**. In recent weeks, users have reported:
# MAGIC
# MAGIC - **Inconsistent answer quality**
# MAGIC - **Occasional latency spikes**
# MAGIC - **Rising usage costs**
# MAGIC
# MAGIC Leadership has asked you to diagnose these issues, evaluate system performance, and implement improvements using Databricks monitoring and evaluation tools.
# MAGIC
# MAGIC ### Your Workflow in This Lab
# MAGIC
# MAGIC 1. **Running structured evaluations** using MLflow to measure latency, token usage, and groundedness
# MAGIC 2. **Querying inference tables** to identify patterns in real-world traffic, including token growth and slow prompts
# MAGIC 3. **Using monitoring dashboards and agent traces** to investigate multi-step workflows
# MAGIC 4. **Detecting anomalies and configuring alert conditions** for early issue detection
# MAGIC 5. **Applying cost-optimization techniques** such as prompt refinement and context control
# MAGIC 6. **Validating improvements** with repeatable evaluation runs
# MAGIC
# MAGIC This lab mirrors real-world LLM observability and optimization challenges, where reliable performance and controlled operational cost are essential to business continuity.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC - ✅ Design an end-to-end evaluation and monitoring workflow on Databricks
# MAGIC - ✅ Apply MLflow metrics to assess LLM quality and performance
# MAGIC - ✅ Use inference tables to diagnose latency and cost issues
# MAGIC - ✅ Monitor multi-step agent workflows to identify inefficiencies
# MAGIC - ✅ Configure anomaly alerts to detect unusual production behavior
# MAGIC - ✅ Apply optimization strategies to improve reliability and reduce cost
# MAGIC - ✅ Incorporate Chapter 8 best practices including calibrated metric interpretation, baseline-aware monitoring, structured alerting strategies, and periodic review cycles
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Lab Duration
# MAGIC
# MAGIC **Estimated Time:** 90-120 minutes
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Prerequisites and Environment Setup
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Install required libraries
# MAGIC 2. Configure connection to Databricks workspace
# MAGIC 3. Set up MLflow tracking
# MAGIC 4. Generate synthetic Q&A data for our evaluation scenarios
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.1: Install Required Libraries
# MAGIC
# MAGIC We need to install the necessary Python packages for working with Databricks, MLflow, and LLM evaluation. These packages provide the foundation for our monitoring and evaluation workflow.

# COMMAND ----------

# Install required packages for the lab
# - databricks-sdk: Official Databricks SDK for Python
# - mlflow: MLflow for experiment tracking and model evaluation
# - databricks-agents: For agent evaluation capabilities
# - openai: OpenAI client for model serving interactions

%pip install databricks-sdk mlflow>=2.14.0 databricks-agents openai pandas numpy matplotlib seaborn --quiet

# Restart Python to pick up new packages (required in Databricks notebooks)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.2: Configure Databricks Connection
# MAGIC
# MAGIC We'll set up the connection to our Databricks workspace using the SDK. This configuration enables us to:
# MAGIC - Create and manage Model Serving endpoints
# MAGIC - Access inference tables
# MAGIC - Configure monitoring and alerts
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📋 How to Find Your Workspace URL and ID
# MAGIC
# MAGIC 1. **Workspace URL**: Look at your browser's address bar when logged into Databricks. It follows this format:
# MAGIC    ```
# MAGIC    https://adb-<workspace-id>.<region>.azuredatabricks.net
# MAGIC    ```
# MAGIC    For example: `https://adb-3141834805281316.15.azuredatabricks.net`
# MAGIC
# MAGIC 2. **Workspace ID**: The numeric value after `adb-` in your URL (e.g., `3141834805281715`)
# MAGIC
# MAGIC 3. **Alternative Method**:
# MAGIC    - Click on your **username** in the top-right corner
# MAGIC    - Select **User Settings**
# MAGIC    - The Workspace ID is displayed under **Workspace Info**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔑 How to Generate a Personal Access Token (PAT)
# MAGIC
# MAGIC 1. **Navigate to User Settings**:
# MAGIC    - Click on your **username** in the top-right corner of the Databricks workspace
# MAGIC    - Select **User Settings** from the dropdown menu
# MAGIC
# MAGIC 2. **Access Developer Settings**:
# MAGIC    - In the left sidebar, click on **Developer**
# MAGIC    - You'll see the **Access tokens** section
# MAGIC
# MAGIC 3. **Generate New Token**:
# MAGIC    - Click the **Manage** button next to Access tokens
# MAGIC    - Click **Generate new token**
# MAGIC    - Enter a **Comment** (e.g., "Chapter 8 Lab Token")
# MAGIC    - Set **Lifetime (days)** - leave blank for no expiration, or set a value like `90`
# MAGIC    - Click **Generate**
# MAGIC
# MAGIC 4. **Copy and Save the Token**:
# MAGIC    - ⚠️ **IMPORTANT**: Copy the token immediately! It will only be shown once.
# MAGIC    - The token starts with `dapi` (e.g., `dapixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-2`)
# MAGIC    - Store it securely - you'll need it for the configuration below
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🔒 Security Note:** In a production environment, store credentials securely using Databricks Secrets instead of hardcoding tokens in notebooks.

# COMMAND ----------

# Import required libraries for Databricks connection
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# =============================================================================
# DATABRICKS WORKSPACE CONFIGURATION
# =============================================================================
# These credentials connect us to the Azure Databricks workspace
# In production, use Databricks Secrets instead of hardcoding tokens
# Please replace your own values in place of xxx based in the instructions provided above
DATABRICKS_HOST = "xxx"
DATABRICKS_TOKEN = "xxx"
CLUSTER_ID = "xxx"

# Set environment variables for SDK and MLflow
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# Initialize the Workspace Client
# This client provides access to all Databricks APIs
w = WorkspaceClient(
    host=DATABRICKS_HOST,
    token=DATABRICKS_TOKEN
)

# Verify connection by listing current user
current_user = w.current_user.me()
print(f"✅ Connected to Databricks as: {current_user.user_name}")
print(f"📍 Workspace: {DATABRICKS_HOST}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.3: Configure MLflow Tracking
# MAGIC
# MAGIC MLflow is our primary tool for tracking experiments, logging metrics, and evaluating LLM performance. We'll configure it to use the Databricks-hosted MLflow tracking server, which provides:
# MAGIC
# MAGIC - **Centralized experiment tracking** across team members
# MAGIC - **Model versioning** through the Model Registry
# MAGIC - **Built-in LLM evaluation metrics** for quality assessment

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
# Configure MLflow to use Databricks as the tracking server
# This enables centralized experiment management and model registry

mlflow.set_tracking_uri("databricks")

# Create an experiment for our LLM evaluation work
# Experiments group related runs together for easy comparison
EXPERIMENT_NAME = "/Users/" + current_user.user_name + "/llm_evaluation_monitoring_lab"

# Create or get the experiment
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"✅ Created new experiment: {EXPERIMENT_NAME}")
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print(f"✅ Using existing experiment: {EXPERIMENT_NAME}")

# Set the active experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"📊 Experiment ID: {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.4: Generate Synthetic Q&A Dataset
# MAGIC
# MAGIC To simulate our production question-answering system, we'll create a synthetic dataset that includes:
# MAGIC
# MAGIC - **Questions** of varying complexity (simple, moderate, complex)
# MAGIC - **Ground truth answers** for evaluation
# MAGIC - **Context documents** to test groundedness
# MAGIC - **Metadata** for categorization and analysis
# MAGIC
# MAGIC This dataset will be used throughout the lab to:
# MAGIC 1. Test our Model Serving endpoint
# MAGIC 2. Evaluate response quality
# MAGIC 3. Measure latency and token usage
# MAGIC 4. Simulate production traffic patterns

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================
# We create realistic Q&A data that mimics an internal knowledge base system
# This includes various question types, complexities, and domains

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define question categories and their characteristics
CATEGORIES = {
    "hr_policy": {
        "questions": [
            "What is the company's remote work policy?",
            "How many vacation days do employees get per year?",
            "What is the process for requesting parental leave?",
            "How do I submit an expense report?",
            "What are the health insurance options available?"
        ],
        "contexts": [
            "The company allows employees to work remotely up to 3 days per week. Remote work requests must be approved by the direct manager. Employees must maintain core hours of 10 AM to 3 PM in their local timezone.",
            "Full-time employees receive 20 vacation days per year, accrued monthly. Unused vacation days can be carried over up to a maximum of 5 days. New employees are eligible for vacation after 90 days.",
            "Parental leave policy provides 16 weeks of paid leave for primary caregivers and 8 weeks for secondary caregivers. Leave must be requested at least 30 days in advance through the HR portal.",
            "Expense reports must be submitted within 30 days of the expense date. Use the Concur system to upload receipts and categorize expenses. Manager approval is required for expenses over $100.",
            "The company offers three health insurance tiers: Basic, Standard, and Premium. All plans include dental and vision coverage. Open enrollment occurs annually in November."
        ],
        "complexity": "simple"
    },
    "technical_docs": {
        "questions": [
            "How do I configure the API authentication for our microservices?",
            "What is the recommended approach for database connection pooling?",
            "How should I handle error logging in production applications?",
            "What are the security requirements for storing customer data?",
            "How do I set up CI/CD pipelines for new projects?"
        ],
        "contexts": [
            "API authentication uses OAuth 2.0 with JWT tokens. Services must register with the central auth server and obtain client credentials. Token expiration is set to 1 hour with refresh token support.",
            "Database connection pooling should use HikariCP with a minimum pool size of 5 and maximum of 20 connections. Connection timeout is set to 30 seconds. Enable connection validation on borrow.",
            "Production logging must use structured JSON format with correlation IDs. Log levels: ERROR for exceptions, WARN for recoverable issues, INFO for business events. Use ELK stack for aggregation.",
            "Customer data must be encrypted at rest using AES-256 and in transit using TLS 1.3. PII requires additional masking in logs. Data retention policy is 7 years for financial data.",
            "CI/CD pipelines use GitHub Actions with standardized templates. All projects must include unit tests (80% coverage), security scanning, and automated deployment to staging before production."
        ],
        "complexity": "moderate"
    },
    "strategic_planning": {
        "questions": [
            "What is our company's five-year growth strategy and how does it align with market trends?",
            "How should we approach the integration of AI capabilities across all product lines?",
            "What are the key risk factors in our current market expansion plan?",
            "How do we balance innovation investment with maintaining core business profitability?",
            "What organizational changes are needed to support our digital transformation initiative?"
        ],
        "contexts": [
            "The five-year strategy focuses on three pillars: market expansion into APAC region, product diversification through AI integration, and operational efficiency through automation. Target growth is 25% CAGR.",
            "AI integration roadmap includes: Phase 1 - customer service automation, Phase 2 - predictive analytics for sales, Phase 3 - AI-powered product features. Budget allocation is $50M over 3 years.",
            "Key risks include: regulatory changes in target markets, currency fluctuation exposure, talent acquisition challenges, and competitive pressure from well-funded startups. Mitigation strategies are documented.",
            "Innovation budget is set at 15% of revenue with quarterly review cycles. Core business must maintain 20% profit margin. New ventures have 18-month runway to demonstrate product-market fit.",
            "Digital transformation requires: flattening organizational hierarchy, creating cross-functional pods, investing in employee upskilling, and establishing a dedicated transformation office with C-suite sponsorship."
        ],
        "complexity": "complex"
    }
}

print("✅ Question categories defined")
print(f"📋 Categories: {list(CATEGORIES.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating the Evaluation Dataset
# MAGIC
# MAGIC **What we're doing:** Building a structured evaluation dataset that pairs each question with its "ground truth" (correct) answer. This is essential for measuring how well the LLM performs.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. **`generate_ground_truth_answer()` function**: Extracts the expected answer from the context based on complexity:
# MAGIC    - *Simple questions*: First sentence of context is the answer
# MAGIC    - *Moderate questions*: First two sentences combined
# MAGIC    - *Complex questions*: Full context needed for complete answer
# MAGIC
# MAGIC 2. **Dataset structure**: Each record contains:
# MAGIC    - `id`: Unique identifier (e.g., "product_info_1")
# MAGIC    - `question`: The user's question
# MAGIC    - `context`: Background information the LLM should use
# MAGIC    - `ground_truth`: The correct answer we expect
# MAGIC    - `category` and `complexity`: For stratified analysis
# MAGIC    - `expected_tokens`: Estimated token count for cost prediction
# MAGIC
# MAGIC **Why this matters:** Without ground truth answers, we can't objectively measure if the LLM is giving correct responses. This dataset becomes our "answer key" for evaluation.

# COMMAND ----------

# =============================================================================
# CREATE EVALUATION DATASET
# =============================================================================
# Build a comprehensive dataset for LLM evaluation with ground truth answers

def generate_ground_truth_answer(question, context, complexity):
    """
    Generate a ground truth answer based on the context.
    In a real scenario, these would be human-curated answers.
    """
    # Extract key information from context as the ground truth
    sentences = context.split(". ")
    if complexity == "simple":
        return sentences[0] + "." if sentences else context
    elif complexity == "moderate":
        return ". ".join(sentences[:2]) + "." if len(sentences) >= 2 else context
    else:  # complex
        return context

# Build the evaluation dataset
eval_data = []

for category, data in CATEGORIES.items():
    for i, (question, context) in enumerate(zip(data["questions"], data["contexts"])):
        ground_truth = generate_ground_truth_answer(question, context, data["complexity"])

        eval_data.append({
            "id": f"{category}_{i+1}",
            "question": question,
            "context": context,
            "ground_truth": ground_truth,
            "category": category,
            "complexity": data["complexity"],
            "expected_tokens": len(context.split()) * 2  # Rough estimate
        })

# Create DataFrame
eval_df = pd.DataFrame(eval_data)

print(f"✅ Created evaluation dataset with {len(eval_df)} samples")
print(f"\n📊 Dataset Summary:")
print(eval_df.groupby(["category", "complexity"]).size().unstack(fill_value=0))
display(eval_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generating Simulated Production Traffic
# MAGIC
# MAGIC **What we're doing:** Creating synthetic inference logs that mimic what a real production LLM endpoint would generate over 7 days of operation.
# MAGIC
# MAGIC **How the `generate_traffic_data()` function works:**
# MAGIC
# MAGIC 1. **Time distribution**: Generates timestamps over 7 days with realistic patterns:
# MAGIC    - More requests during business hours (9 AM - 6 PM)
# MAGIC    - Fewer requests on weekends and nights
# MAGIC
# MAGIC 2. **Latency simulation** using normal distribution:
# MAGIC    ```python
# MAGIC    base_latency = {"simple": 200, "moderate": 400, "complex": 800}
# MAGIC    latency = np.random.normal(base_latency, std_dev)
# MAGIC    ```
# MAGIC    - Simple queries: ~200ms average
# MAGIC    - Complex queries: ~800ms average
# MAGIC    - 5% of requests get artificial "spikes" (3-5x normal latency)
# MAGIC
# MAGIC 3. **Token usage**: Calculated based on complexity with random variation
# MAGIC
# MAGIC 4. **Error injection**: 2% of requests marked as "error" or "timeout" to simulate real failures
# MAGIC
# MAGIC **Why this matters:** Real production data takes weeks to accumulate. Synthetic data lets us immediately practice monitoring, anomaly detection, and alerting without waiting.

# COMMAND ----------

# =============================================================================
# GENERATE SIMULATED PRODUCTION TRAFFIC DATA
# =============================================================================
# Create synthetic inference logs to simulate production traffic patterns
# This data will be used for monitoring and anomaly detection exercises

def generate_traffic_data(num_records=500):
    """
    Generate synthetic production traffic data with realistic patterns:
    - Normal latency with occasional spikes
    - Varying token usage based on question complexity
    - Time-based patterns (higher traffic during business hours)
    """
    traffic_data = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(num_records):
        # Select random question from our dataset
        sample = random.choice(eval_data)

        # Generate timestamp with business hour bias
        hour_offset = random.gauss(14, 4)  # Peak around 2 PM
        hour_offset = max(8, min(20, hour_offset))  # Clamp to business hours
        timestamp = base_time + timedelta(
            days=random.randint(0, 6),
            hours=hour_offset,
            minutes=random.randint(0, 59)
        )

        # Generate latency based on complexity with occasional spikes
        base_latency = {"simple": 200, "moderate": 400, "complex": 800}
        latency = base_latency[sample["complexity"]]
        latency += random.gauss(0, latency * 0.2)  # Add noise

        # Inject latency spikes (5% of requests)
        if random.random() < 0.05:
            latency *= random.uniform(3, 8)  # Spike multiplier

        # Generate token counts
        input_tokens = len(sample["question"].split()) + len(sample["context"].split())
        output_tokens = int(sample["expected_tokens"] * random.uniform(0.8, 1.2))

        # Inject token anomalies (3% of requests)
        if random.random() < 0.03:
            output_tokens *= random.randint(3, 5)  # Unexpectedly long responses

        # Generate quality score (simulated)
        base_quality = {"simple": 0.9, "moderate": 0.8, "complex": 0.7}
        quality_score = base_quality[sample["complexity"]] + random.gauss(0, 0.1)
        quality_score = max(0, min(1, quality_score))  # Clamp to [0, 1]

        traffic_data.append({
            "request_id": f"req_{i:06d}",
            "timestamp": timestamp,
            "question": sample["question"],
            "category": sample["category"],
            "complexity": sample["complexity"],
            "latency_ms": round(latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "quality_score": round(quality_score, 3),
            "status": "success" if random.random() > 0.02 else "error"
        })

    return pd.DataFrame(traffic_data)

# Generate traffic data
traffic_df = generate_traffic_data(500)

print(f"✅ Generated {len(traffic_df)} synthetic traffic records")
print(f"\n📊 Traffic Summary:")
print(f"   Date Range: {traffic_df['timestamp'].min()} to {traffic_df['timestamp'].max()}")
print(f"   Avg Latency: {traffic_df['latency_ms'].mean():.2f} ms")
print(f"   Avg Tokens: {traffic_df['total_tokens'].mean():.1f}")
print(f"   Error Rate: {(traffic_df['status'] == 'error').mean()*100:.1f}%")
display(traffic_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving Datasets to Delta Tables
# MAGIC
# MAGIC **What we're doing:** Persisting our synthetic datasets to Delta Lake tables in the Unity Catalog so they survive cluster restarts and can be queried by other notebooks.
# MAGIC
# MAGIC **How it works:**
# MAGIC
# MAGIC 1. **Create schema**: `spark.sql("CREATE SCHEMA IF NOT EXISTS main.llm_monitoring_lab")`
# MAGIC    - Creates a namespace to organize our tables
# MAGIC
# MAGIC 2. **Convert and save**:
# MAGIC    ```python
# MAGIC    spark.createDataFrame(pandas_df)  # Convert pandas → Spark DataFrame
# MAGIC    .write.mode("overwrite")          # Replace if exists
# MAGIC    .saveAsTable("catalog.schema.table")  # Save as managed Delta table
# MAGIC    ```
# MAGIC
# MAGIC 3. **Tables created**:
# MAGIC    - `evaluation_dataset`: Our Q&A pairs with ground truth answers
# MAGIC    - `traffic_data`: Synthetic production traffic logs
# MAGIC
# MAGIC **Why Delta Lake?**
# MAGIC - **ACID transactions**: Safe concurrent reads/writes
# MAGIC - **Time travel**: Query previous versions with `VERSION AS OF`
# MAGIC - **Schema enforcement**: Prevents bad data from corrupting tables
# MAGIC - **Optimized queries**: Auto-compaction and Z-ordering for fast reads

# COMMAND ----------

# =============================================================================
# SAVE DATASETS TO DATABRICKS
# =============================================================================
# Store our synthetic data in Delta tables for use throughout the lab

# Define catalog and schema (using default catalog)
CATALOG = "main"
SCHEMA = "llm_monitoring_lab"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Convert pandas DataFrames to Spark DataFrames and save as Delta tables
eval_spark_df = spark.createDataFrame(eval_df)
eval_spark_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.evaluation_dataset")

traffic_spark_df = spark.createDataFrame(traffic_df)
traffic_spark_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.traffic_data")

print(f"✅ Saved evaluation dataset to {CATALOG}.{SCHEMA}.evaluation_dataset")
print(f"✅ Saved traffic data to {CATALOG}.{SCHEMA}.traffic_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 2: Creating and Deploying the Model Serving Endpoint
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Register a foundation model for serving
# MAGIC 2. Create a Model Serving endpoint
# MAGIC 3. Configure inference tables for logging
# MAGIC 4. Test the endpoint with sample queries
# MAGIC
# MAGIC This endpoint will serve as our production Q&A system that we'll monitor and optimize throughout the lab.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2.1: Configure Foundation Model API Access
# MAGIC
# MAGIC Databricks provides **Foundation Model APIs** - pre-deployed, pay-per-token endpoints for popular LLMs. These are:
# MAGIC
# MAGIC - **Immediately available** - no deployment required
# MAGIC - **Auto-scaling** - handles any traffic volume
# MAGIC - **Pay-per-token** - cost-effective for variable workloads
# MAGIC - **OpenAI-compatible** - easy integration with existing code
# MAGIC
# MAGIC We'll use the **Meta Llama 3.1 8B Instruct** model (`databricks-meta-llama-3-1-8b-instruct`), which is well-suited for Q&A tasks.
# MAGIC
# MAGIC For inference table logging, we'll create a **custom proxy endpoint** that wraps the foundation model and enables request/response logging.

# COMMAND ----------

import time

# =============================================================================
# MODEL SERVING ENDPOINT CONFIGURATION
# =============================================================================
# Databricks Foundation Model APIs are pre-deployed and ready to use
# No custom endpoint creation needed - we query them directly

FOUNDATION_MODEL = "databricks-meta-llama-3-1-8b-instruct"

# First, let's verify the Foundation Model is available
print("🔍 Checking available Foundation Model APIs...")

# List foundation model endpoints
try:
    all_endpoints = list(w.serving_endpoints.list())
    foundation_endpoints = [e for e in all_endpoints if e.name.startswith("databricks-")]
    print(f"   Found {len(foundation_endpoints)} Foundation Model endpoints")

    # Check if our model is available
    available_models = [e.name for e in foundation_endpoints]
    if FOUNDATION_MODEL in available_models:
        print(f"   ✅ {FOUNDATION_MODEL} is available")
    else:
        print(f"   ⚠️ {FOUNDATION_MODEL} not found in list. Available models:")
        for model in available_models[:5]:
            print(f"      - {model}")
        print(f"   Note: The model may still work - Foundation Models are always available")
except Exception as e:
    print(f"   Note: Could not list endpoints ({e})")
    print(f"   Proceeding with Foundation Model - these are always available")

# For this lab, we'll use the Foundation Model directly
# The endpoint name for queries will be the foundation model name
ENDPOINT_NAME = FOUNDATION_MODEL

print(f"\n✅ Using Foundation Model: {ENDPOINT_NAME}")
print(f"   Endpoint URL: {DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verifying the Foundation Model Endpoint
# MAGIC
# MAGIC **What we're doing:** Confirming that the Databricks Foundation Model API is accessible before we start sending queries.
# MAGIC
# MAGIC **How it works:**
# MAGIC
# MAGIC 1. **SDK endpoint check**: `w.serving_endpoints.get(name=ENDPOINT_NAME)`
# MAGIC    - Retrieves metadata about the endpoint
# MAGIC    - Confirms the endpoint exists and is in "READY" state
# MAGIC
# MAGIC 2. **What we verify**:
# MAGIC    - Endpoint name matches our configuration
# MAGIC    - State is "READY" (not "PENDING" or "FAILED")
# MAGIC    - URL is correctly formed for API calls
# MAGIC
# MAGIC **Foundation Models vs Custom Endpoints:**
# MAGIC
# MAGIC | Aspect | Foundation Model | Custom Endpoint |
# MAGIC |--------|-----------------|-----------------|
# MAGIC | Provisioning | Always ready | Minutes to hours |
# MAGIC | Scaling | Automatic | Manual configuration |
# MAGIC | Billing | Pay-per-token | Pay for compute time |
# MAGIC | Customization | None | Fine-tuning possible |
# MAGIC
# MAGIC **Why this matters:** Verifying the endpoint upfront prevents confusing errors later. If the endpoint isn't accessible, we want to know immediately rather than after running expensive evaluation jobs.

# COMMAND ----------

# =============================================================================
# VERIFY FOUNDATION MODEL ENDPOINT
# =============================================================================
# Foundation Model APIs are always ready - no provisioning needed
# Let's verify the endpoint is accessible

try:
    endpoint_info = w.serving_endpoints.get(name=ENDPOINT_NAME)
    print(f"✅ Foundation Model Endpoint Verified")
    print(f"\n📊 Endpoint Details:")
    print(f"   Name: {endpoint_info.name}")
    print(f"   State: {endpoint_info.state.ready}")
    print(f"   URL: {DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations")

    # Check endpoint configuration
    if endpoint_info.config and endpoint_info.config.served_entities:
        for entity in endpoint_info.config.served_entities:
            print(f"\n   Served Entity:")
            print(f"      Name: {entity.entity_name if hasattr(entity, 'entity_name') else 'N/A'}")
except Exception as e:
    print(f"⚠️ Could not get endpoint details: {str(e)}")
    print(f"   This is expected for Foundation Model APIs - they work without explicit configuration")
    print(f"\n✅ Proceeding with endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2.2: Test the Model Serving Endpoint
# MAGIC
# MAGIC Now we'll test our endpoint with sample queries to verify it's working correctly. We'll use the OpenAI-compatible API that Databricks provides for foundation models.

# COMMAND ----------

from openai import OpenAI

# =============================================================================
# TEST MODEL SERVING ENDPOINT
# =============================================================================
# Use OpenAI-compatible client to query our endpoint

# Disable MLflow's automatic tracing for OpenAI calls to avoid duplicate trace ID warnings
# This happens because MLflow tries to auto-trace LLM calls, but re-running cells
# can cause trace ID conflicts
try:
    mlflow.openai.autolog(disable=True)
except AttributeError:
    # Older MLflow versions may not have this - that's OK
    pass

# Initialize OpenAI client pointing to Databricks
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints"
)

def query_qa_endpoint(question, context, max_tokens=500):
    """
    Query the Q&A endpoint with a question and context.
    Returns the response and timing information.
    """
    # Construct the prompt for Q&A
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    Be concise and accurate. If the answer is not in the context, say so."""

    user_prompt = f"""Context: {context}

Question: {question}

Answer:"""

    start_time = time.time()

    response = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.1  # Low temperature for factual responses
    )

    latency_ms = (time.time() - start_time) * 1000

    return {
        "answer": response.choices[0].message.content,
        "latency_ms": latency_ms,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

# Test with a sample question
test_sample = eval_data[0]
print(f"📝 Testing with question: {test_sample['question']}")
print(f"📄 Context: {test_sample['context'][:100]}...")

result = query_qa_endpoint(test_sample["question"], test_sample["context"])

print(f"\n✅ Response received:")
print(f"   Answer: {result['answer']}")
print(f"   Latency: {result['latency_ms']:.2f} ms")
print(f"   Tokens: {result['total_tokens']} (input: {result['input_tokens']}, output: {result['output_tokens']})")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Establishing Baseline Performance Metrics
# MAGIC
# MAGIC **What we're doing:** Running multiple queries to collect baseline performance data that will inform our monitoring thresholds.
# MAGIC
# MAGIC **How it works:**
# MAGIC
# MAGIC 1. **Batch execution loop**:
# MAGIC    ```python
# MAGIC    for sample in eval_data[:10]:
# MAGIC        result = query_qa_endpoint(sample["question"], sample["context"])
# MAGIC        batch_results.append(result)
# MAGIC        time.sleep(1)  # Rate limiting to avoid throttling
# MAGIC    ```
# MAGIC
# MAGIC 2. **Metrics collected per query**:
# MAGIC    - `latency_ms`: End-to-end response time
# MAGIC    - `input_tokens`: Tokens in prompt (question + context)
# MAGIC    - `output_tokens`: Tokens in generated answer
# MAGIC    - `category` and `complexity`: For stratified analysis
# MAGIC
# MAGIC 3. **Baseline statistics calculated**:
# MAGIC    - Mean, P50 (median), P95, P99 latency
# MAGIC    - Average token usage by complexity
# MAGIC    - Success rate
# MAGIC
# MAGIC **Why baselines matter (Chapter 8 best practice):**
# MAGIC - **Threshold setting**: "Alert when latency > P95 baseline × 1.5"
# MAGIC - **Drift detection**: Compare current metrics to baseline
# MAGIC - **Capacity planning**: Predict costs based on token patterns
# MAGIC - **SLA definition**: Set realistic performance guarantees
# MAGIC
# MAGIC Without baselines, you're guessing at what "normal" looks like.

# COMMAND ----------

# =============================================================================
# BATCH TEST FOR BASELINE METRICS
# =============================================================================
# Run multiple queries to establish baseline performance metrics

print("🔄 Running batch test to establish baseline metrics...")
print("   This will take a few minutes...\n")

batch_results = []

# Test with a subset of our evaluation data
for i, sample in enumerate(eval_data[:10]):
    try:
        result = query_qa_endpoint(sample["question"], sample["context"])
        result["id"] = sample["id"]
        result["category"] = sample["category"]
        result["complexity"] = sample["complexity"]
        result["ground_truth"] = sample["ground_truth"]
        batch_results.append(result)
        print(f"   ✓ Completed {i+1}/10: {sample['id']} ({result['latency_ms']:.0f}ms)")
    except Exception as e:
        print(f"   ✗ Failed {i+1}/10: {sample['id']} - {str(e)}")

    # Small delay to avoid rate limiting
    time.sleep(1)

# Create results DataFrame
batch_df = pd.DataFrame(batch_results)

print(f"\n📊 Baseline Performance Summary:")
print(f"   Average Latency: {batch_df['latency_ms'].mean():.2f} ms")
print(f"   P95 Latency: {batch_df['latency_ms'].quantile(0.95):.2f} ms")
print(f"   Average Tokens: {batch_df['total_tokens'].mean():.1f}")
print(f"\n   By Complexity:")
print(batch_df.groupby('complexity')[['latency_ms', 'total_tokens']].mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 3: Structured Evaluation with MLflow
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Set up MLflow evaluation for LLM responses
# MAGIC 2. Measure quality metrics (groundedness, relevance, coherence)
# MAGIC 3. Track latency and token usage metrics
# MAGIC 4. Compare baseline vs. optimized performance
# MAGIC
# MAGIC MLflow provides built-in LLM evaluation capabilities that help us systematically assess model quality.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3.1: Configure MLflow LLM Evaluation
# MAGIC
# MAGIC **What we're doing:** Setting up custom evaluation scorers using MLflow's new GenAI API (MLflow 3.4+).
# MAGIC
# MAGIC **How the new API works:**
# MAGIC
# MAGIC 1. **`@scorer` decorator**: Creates a custom evaluation function
# MAGIC    ```python
# MAGIC    @scorer
# MAGIC    def groundedness(inputs: dict, outputs: str) -> int:
# MAGIC        # Use LLM-as-judge to score the output
# MAGIC        return score  # 1-5
# MAGIC    ```
# MAGIC
# MAGIC 2. **Function signature**: Scorers receive standardized parameters:
# MAGIC    - `inputs`: Dictionary with question, context, etc.
# MAGIC    - `outputs`: The model's generated response
# MAGIC    - `expectations`: Ground truth for comparison (optional)
# MAGIC
# MAGIC 3. **LLM-as-Judge pattern**: We use another LLM (Llama 70B) to evaluate responses
# MAGIC
# MAGIC **Why the new API?**
# MAGIC - The old `make_genai_metric` is deprecated since MLflow 3.4.0
# MAGIC - New API is more flexible and Pythonic
# MAGIC - Better integration with MLflow's tracing and monitoring

# COMMAND ----------

from mlflow.genai.scorers import scorer
from typing import Literal

# =============================================================================
# MLFLOW GENAI EVALUATION SETUP (New API - MLflow 3.4+)
# =============================================================================
# Configure evaluation scorers for our Q&A system using the new GenAI API
# This replaces the deprecated make_genai_metric approach

# Use the same foundation model for judging (8B model available in workspace)
JUDGE_MODEL = FOUNDATION_MODEL  # databricks-meta-llama-3-1-8b-instruct

# Define a custom groundedness scorer using the @scorer decorator
# This measures how well the answer is grounded in the provided context
@scorer
def groundedness(inputs: dict, outputs: str) -> int:
    """
    Evaluate if the answer is grounded in the provided context.
    Returns a score from 1-5.
    """
    from openai import OpenAI

    judge_client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints"
    )

    prompt = f"""You are evaluating whether an answer is grounded in the provided context.

Context: {inputs.get('context', '')}
Question: {inputs.get('question', '')}
Answer: {outputs}

Score the groundedness from 1-5:
1 = Answer contains claims not supported by context
2 = Answer partially supported but has unsupported claims
3 = Answer mostly supported with minor gaps
4 = Answer well supported by context
5 = Answer fully grounded in context

Respond with ONLY a single number (1-5)."""

    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    try:
        score = int(response.choices[0].message.content.strip()[0])
        return max(1, min(5, score))  # Ensure score is between 1-5
    except:
        return 3  # Default to middle score if parsing fails

# Define answer correctness scorer
@scorer
def answer_correctness(inputs: dict, outputs: str, expectations: dict) -> int:
    """
    Evaluate if the answer correctly addresses the question based on ground truth.
    Returns a score from 1-5.
    """
    from openai import OpenAI

    judge_client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints"
    )

    prompt = f"""Compare the generated answer with the ground truth answer.

Question: {inputs.get('question', '')}
Ground Truth: {expectations.get('ground_truth', '')}
Generated Answer: {outputs}

Score from 1-5:
1 = Completely incorrect or contradicts ground truth
2 = Partially correct but missing key information
3 = Mostly correct with some inaccuracies
4 = Correct with minor differences in wording
5 = Fully correct and complete

Respond with ONLY a single number (1-5)."""

    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    try:
        score = int(response.choices[0].message.content.strip()[0])
        return max(1, min(5, score))  # Ensure score is between 1-5
    except:
        return 3  # Default to middle score if parsing fails

print("✅ Custom evaluation scorers configured (MLflow 3.4+ GenAI API):")
print(f"   - Judge model: {JUDGE_MODEL}")
print("   - groundedness: Measures factual support from context")
print("   - answer_correctness: Compares with ground truth answers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3.2: Run Comprehensive Evaluation
# MAGIC
# MAGIC Now we'll run a full evaluation using MLflow. This will:
# MAGIC 1. Query our endpoint for each evaluation sample
# MAGIC 2. Calculate quality metrics using LLM-as-judge
# MAGIC 3. Log all results to MLflow for tracking
# MAGIC 4. Generate a comprehensive evaluation report

# COMMAND ----------

# =============================================================================
# RUN MLFLOW EVALUATION
# =============================================================================
# Execute comprehensive evaluation and log results

# Prepare evaluation data with model outputs
eval_results = []

print("🔄 Running comprehensive evaluation...")
print("   Querying endpoint and collecting responses...\n")

for i, sample in enumerate(eval_data):
    try:
        # Query the endpoint
        result = query_qa_endpoint(sample["question"], sample["context"])

        eval_results.append({
            "id": sample["id"],
            "question": sample["question"],
            "context": sample["context"],
            "ground_truth": sample["ground_truth"],
            "output": result["answer"],  # MLflow expects 'output' column
            "category": sample["category"],
            "complexity": sample["complexity"],
            "latency_ms": result["latency_ms"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"]
        })

        print(f"   ✓ {i+1}/{len(eval_data)}: {sample['id']}")

    except Exception as e:
        print(f"   ✗ {i+1}/{len(eval_data)}: {sample['id']} - {str(e)}")

    time.sleep(0.5)  # Rate limiting

# Create evaluation DataFrame
eval_results_df = pd.DataFrame(eval_results)
print(f"\n✅ Collected {len(eval_results_df)} evaluation samples")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logging Evaluation Results to MLflow
# MAGIC
# MAGIC **What we're doing:** Running our custom scorers on each sample and logging results to MLflow.
# MAGIC
# MAGIC **How the new GenAI API works:**
# MAGIC
# MAGIC 1. **Prepare data in new format**:
# MAGIC    ```python
# MAGIC    eval_data_for_genai.append({
# MAGIC        "inputs": {"question": ..., "context": ...},
# MAGIC        "outputs": row["output"],
# MAGIC        "expectations": {"ground_truth": ...}
# MAGIC    })
# MAGIC    ```
# MAGIC
# MAGIC 2. **Call scorers directly**:
# MAGIC    ```python
# MAGIC    g_score = groundedness(inputs=item["inputs"], outputs=item["outputs"])
# MAGIC    c_score = answer_correctness(inputs=..., outputs=..., expectations=...)
# MAGIC    ```
# MAGIC    - Each scorer calls the judge LLM (Llama 70B)
# MAGIC    - Returns a score from 1-5
# MAGIC
# MAGIC 3. **Log aggregate metrics**:
# MAGIC    ```python
# MAGIC    mlflow.log_metric("groundedness_mean", avg_groundedness)
# MAGIC    mlflow.log_metric("answer_correctness_mean", avg_correctness)
# MAGIC    ```
# MAGIC
# MAGIC **Why this approach?** The new MLflow 3.4+ GenAI API is more flexible and Pythonic. We call scorers directly, giving us full control over error handling and rate limiting.

# COMMAND ----------

# =============================================================================
# LOG EVALUATION TO MLFLOW
# =============================================================================
# Run MLflow evaluation with our custom scorers (New GenAI API)

with mlflow.start_run(run_name="baseline_evaluation") as run:

    # Log evaluation parameters
    mlflow.log_param("endpoint_name", ENDPOINT_NAME)
    mlflow.log_param("model_name", FOUNDATION_MODEL)
    mlflow.log_param("num_samples", len(eval_results_df))
    mlflow.log_param("evaluation_date", datetime.now().isoformat())

    # Prepare data in the new format for mlflow.genai.evaluate()
    # The new API expects: inputs (dict), outputs (str), expectations (dict)
    eval_data_for_genai = []
    for _, row in eval_results_df.iterrows():
        eval_data_for_genai.append({
            "inputs": {
                "question": row["question"],
                "context": row["context"]
            },
            "outputs": row["output"],
            "expectations": {
                "ground_truth": row["ground_truth"]
            }
        })

    # Run evaluation with custom scorers
    print("🔄 Running MLflow evaluation with LLM-as-judge scorers...")
    print("   (Using new MLflow 3.4+ GenAI API)")

    # Calculate scores manually and log them
    groundedness_scores = []
    correctness_scores = []

    for i, item in enumerate(eval_data_for_genai):
        try:
            g_score = groundedness(inputs=item["inputs"], outputs=item["outputs"])
            c_score = answer_correctness(
                inputs=item["inputs"],
                outputs=item["outputs"],
                expectations=item["expectations"]
            )
            groundedness_scores.append(g_score)
            correctness_scores.append(c_score)
            if (i + 1) % 5 == 0:
                print(f"   Evaluated {i + 1}/{len(eval_data_for_genai)} samples...")
        except Exception as e:
            print(f"   Warning: Error evaluating sample {i}: {e}")
            groundedness_scores.append(3)
            correctness_scores.append(3)
        time.sleep(0.5)  # Rate limiting for judge model

    # Calculate aggregate metrics
    avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
    avg_correctness = sum(correctness_scores) / len(correctness_scores)

    # Log LLM-as-judge metrics
    mlflow.log_metric("groundedness_mean", avg_groundedness)
    mlflow.log_metric("answer_correctness_mean", avg_correctness)

    # Log additional performance metrics
    mlflow.log_metric("avg_latency_ms", eval_results_df["latency_ms"].mean())
    mlflow.log_metric("p95_latency_ms", eval_results_df["latency_ms"].quantile(0.95))
    mlflow.log_metric("avg_total_tokens", eval_results_df["total_tokens"].mean())
    mlflow.log_metric("total_input_tokens", eval_results_df["input_tokens"].sum())
    mlflow.log_metric("total_output_tokens", eval_results_df["output_tokens"].sum())

    # Add scores to dataframe and save
    eval_results_df["groundedness_score"] = groundedness_scores
    eval_results_df["correctness_score"] = correctness_scores

    # Log the evaluation dataset as artifact
    eval_results_df.to_csv("/tmp/evaluation_results.csv", index=False)
    mlflow.log_artifact("/tmp/evaluation_results.csv")

    print(f"\n✅ Evaluation logged to MLflow")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Experiment: {EXPERIMENT_NAME}")

    # Display metrics summary
    print(f"\n📊 Evaluation Metrics Summary:")
    print(f"   groundedness_mean: {avg_groundedness:.2f}")
    print(f"   answer_correctness_mean: {avg_correctness:.2f}")
    print(f"   avg_latency_ms: {eval_results_df['latency_ms'].mean():.2f}")
    print(f"   p95_latency_ms: {eval_results_df['latency_ms'].quantile(0.95):.2f}")
    print(f"   avg_total_tokens: {eval_results_df['total_tokens'].mean():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 4: Analyzing Inference Tables
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Query inference tables to analyze production traffic
# MAGIC 2. Identify latency patterns and anomalies
# MAGIC 3. Analyze token usage trends
# MAGIC 4. Detect problematic query patterns
# MAGIC
# MAGIC Inference tables automatically capture all requests to Model Serving endpoints, providing rich data for monitoring and debugging.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4.1: Query Inference Tables
# MAGIC
# MAGIC Databricks automatically logs all inference requests to Delta tables when auto-capture is enabled. These tables contain:
# MAGIC
# MAGIC - **Request payloads**: Input prompts and parameters
# MAGIC - **Response data**: Model outputs and metadata
# MAGIC - **Timing information**: Latency measurements
# MAGIC - **Token counts**: Usage metrics for cost analysis

# COMMAND ----------

# =============================================================================
# QUERY INFERENCE TABLES
# =============================================================================
# Analyze production traffic patterns from inference logs

# For this lab, we'll use our synthetic traffic data
# In production, you would query the actual inference tables:
# inference_table = f"{CATALOG}.{SCHEMA}.qa_endpoint_payload"

# Load our synthetic traffic data
traffic_analysis_df = spark.table(f"{CATALOG}.{SCHEMA}.traffic_data").toPandas()

print(f"📊 Analyzing {len(traffic_analysis_df)} inference records")
print(f"   Time Range: {traffic_analysis_df['timestamp'].min()} to {traffic_analysis_df['timestamp'].max()}")

# Basic statistics
print(f"\n📈 Traffic Statistics:")
print(f"   Total Requests: {len(traffic_analysis_df)}")
print(f"   Success Rate: {(traffic_analysis_df['status'] == 'success').mean()*100:.1f}%")
print(f"   Avg Latency: {traffic_analysis_df['latency_ms'].mean():.2f} ms")
print(f"   P50 Latency: {traffic_analysis_df['latency_ms'].median():.2f} ms")
print(f"   P95 Latency: {traffic_analysis_df['latency_ms'].quantile(0.95):.2f} ms")
print(f"   P99 Latency: {traffic_analysis_df['latency_ms'].quantile(0.99):.2f} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4.2: Identify Latency Patterns
# MAGIC
# MAGIC We'll analyze latency patterns to identify:
# MAGIC - Time-based variations (peak hours vs. off-peak)
# MAGIC - Complexity-based differences
# MAGIC - Anomalous slow requests

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# LATENCY PATTERN ANALYSIS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Latency distribution by complexity
ax1 = axes[0, 0]
for complexity in ['simple', 'moderate', 'complex']:
    data = traffic_analysis_df[traffic_analysis_df['complexity'] == complexity]['latency_ms']
    ax1.hist(data, bins=30, alpha=0.5, label=complexity)
ax1.set_xlabel('Latency (ms)')
ax1.set_ylabel('Frequency')
ax1.set_title('Latency Distribution by Complexity')
ax1.legend()

# 2. Latency over time
ax2 = axes[0, 1]
traffic_analysis_df['hour'] = pd.to_datetime(traffic_analysis_df['timestamp']).dt.hour
hourly_latency = traffic_analysis_df.groupby('hour')['latency_ms'].mean()
ax2.plot(hourly_latency.index, hourly_latency.values, marker='o')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Average Latency (ms)')
ax2.set_title('Average Latency by Hour')
ax2.grid(True, alpha=0.3)

# 3. Token usage vs latency
ax3 = axes[1, 0]
ax3.scatter(traffic_analysis_df['total_tokens'], traffic_analysis_df['latency_ms'],
            alpha=0.5, c=traffic_analysis_df['complexity'].map({'simple': 0, 'moderate': 1, 'complex': 2}))
ax3.set_xlabel('Total Tokens')
ax3.set_ylabel('Latency (ms)')
ax3.set_title('Token Usage vs Latency')

# 4. Latency percentiles by category
ax4 = axes[1, 1]
category_stats = traffic_analysis_df.groupby('category')['latency_ms'].agg(['mean', 'median', lambda x: x.quantile(0.95)])
category_stats.columns = ['Mean', 'Median', 'P95']
category_stats.plot(kind='bar', ax=ax4)
ax4.set_xlabel('Category')
ax4.set_ylabel('Latency (ms)')
ax4.set_title('Latency Statistics by Category')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/tmp/latency_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Latency analysis charts generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4.3: Identify Slow and Anomalous Queries
# MAGIC
# MAGIC We'll identify queries that exceed our latency thresholds and investigate their characteristics.

# COMMAND ----------

# =============================================================================
# ANOMALY DETECTION IN LATENCY
# =============================================================================

# Define latency thresholds based on complexity
LATENCY_THRESHOLDS = {
    "simple": 500,      # 500ms for simple queries
    "moderate": 1000,   # 1s for moderate queries
    "complex": 2000     # 2s for complex queries
}

# Identify slow queries
def is_slow_query(row):
    threshold = LATENCY_THRESHOLDS.get(row['complexity'], 1000)
    return row['latency_ms'] > threshold

traffic_analysis_df['is_slow'] = traffic_analysis_df.apply(is_slow_query, axis=1)

# Analyze slow queries
slow_queries = traffic_analysis_df[traffic_analysis_df['is_slow']]

print(f"🐢 Slow Query Analysis:")
print(f"   Total Slow Queries: {len(slow_queries)} ({len(slow_queries)/len(traffic_analysis_df)*100:.1f}%)")
print(f"\n   By Complexity:")
print(slow_queries.groupby('complexity').size())
print(f"\n   By Category:")
print(slow_queries.groupby('category').size())

# Identify extreme outliers (> 3 standard deviations)
mean_latency = traffic_analysis_df['latency_ms'].mean()
std_latency = traffic_analysis_df['latency_ms'].std()
outlier_threshold = mean_latency + 3 * std_latency

outliers = traffic_analysis_df[traffic_analysis_df['latency_ms'] > outlier_threshold]
print(f"\n⚠️ Extreme Outliers (>{outlier_threshold:.0f}ms): {len(outliers)}")

if len(outliers) > 0:
    print("\n   Sample outlier queries:")
    for _, row in outliers.head(3).iterrows():
        print(f"   - {row['request_id']}: {row['latency_ms']:.0f}ms ({row['complexity']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 5: Monitoring Dashboards and Agent Traces
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Create monitoring dashboards for key metrics
# MAGIC 2. Implement agent tracing for multi-step workflows
# MAGIC 3. Analyze trace data to identify bottlenecks
# MAGIC 4. Set up baseline-aware monitoring
# MAGIC
# MAGIC Effective monitoring requires both real-time dashboards and detailed traces for debugging.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.1: Create Monitoring Dashboard Metrics
# MAGIC
# MAGIC We'll define key metrics for our monitoring dashboard following Chapter 8 best practices:
# MAGIC
# MAGIC - **Latency metrics**: P50, P95, P99 response times
# MAGIC - **Quality metrics**: Groundedness, correctness scores
# MAGIC - **Cost metrics**: Token usage, request volume
# MAGIC - **Error metrics**: Error rates, timeout rates

# COMMAND ----------

# =============================================================================
# MONITORING DASHBOARD METRICS
# =============================================================================
# Define and calculate key monitoring metrics

def calculate_monitoring_metrics(df):
    """
    Calculate comprehensive monitoring metrics from traffic data.
    These metrics form the basis of our monitoring dashboard.
    """
    metrics = {
        # Latency Metrics
        "latency_p50": df['latency_ms'].median(),
        "latency_p95": df['latency_ms'].quantile(0.95),
        "latency_p99": df['latency_ms'].quantile(0.99),
        "latency_mean": df['latency_ms'].mean(),
        "latency_std": df['latency_ms'].std(),

        # Volume Metrics
        "total_requests": len(df),
        "requests_per_hour": len(df) / 168,  # 7 days * 24 hours
        "success_rate": (df['status'] == 'success').mean(),
        "error_rate": (df['status'] == 'error').mean(),

        # Token Metrics
        "avg_input_tokens": df['input_tokens'].mean(),
        "avg_output_tokens": df['output_tokens'].mean(),
        "avg_total_tokens": df['total_tokens'].mean(),
        "total_tokens_consumed": df['total_tokens'].sum(),

        # Quality Metrics (from our synthetic data)
        "avg_quality_score": df['quality_score'].mean(),
        "quality_below_threshold": (df['quality_score'] < 0.7).mean(),

        # Complexity Distribution
        "pct_simple": (df['complexity'] == 'simple').mean(),
        "pct_moderate": (df['complexity'] == 'moderate').mean(),
        "pct_complex": (df['complexity'] == 'complex').mean(),
    }

    return metrics

# Calculate current metrics
current_metrics = calculate_monitoring_metrics(traffic_analysis_df)

print("📊 Current Monitoring Metrics:")
print("\n   Latency Metrics:")
print(f"      P50: {current_metrics['latency_p50']:.2f} ms")
print(f"      P95: {current_metrics['latency_p95']:.2f} ms")
print(f"      P99: {current_metrics['latency_p99']:.2f} ms")

print("\n   Volume Metrics:")
print(f"      Total Requests: {current_metrics['total_requests']}")
print(f"      Success Rate: {current_metrics['success_rate']*100:.1f}%")

print("\n   Token Metrics:")
print(f"      Avg Total Tokens: {current_metrics['avg_total_tokens']:.1f}")
print(f"      Total Consumed: {current_metrics['total_tokens_consumed']:,}")

print("\n   Quality Metrics:")
print(f"      Avg Quality Score: {current_metrics['avg_quality_score']:.3f}")
print(f"      Below Threshold: {current_metrics['quality_below_threshold']*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.2: Implement Agent Tracing
# MAGIC
# MAGIC For multi-step agent workflows, we need detailed tracing to understand:
# MAGIC - Which steps take the longest
# MAGIC - Where errors occur
# MAGIC - How context flows between steps
# MAGIC
# MAGIC We'll use MLflow's tracing capabilities to instrument our Q&A workflow.

# COMMAND ----------

import uuid

# =============================================================================
# AGENT TRACING IMPLEMENTATION
# =============================================================================
# Implement manual tracing for multi-step Q&A workflow
# We'll track timing and metrics for each step without relying on decorators

class TracedQAAgent:
    """
    A traced Q&A agent that logs detailed execution information.
    This simulates a multi-step agent workflow with:
    1. Query preprocessing
    2. Context retrieval
    3. LLM inference
    4. Response post-processing
    """

    def __init__(self, endpoint_name, client):
        self.endpoint_name = endpoint_name
        self.client = client
        self.trace_log = []

    def preprocess_query(self, question):
        """Step 1: Preprocess the user query"""
        start = time.time()

        # Simulate preprocessing (normalization, spell check, etc.)
        processed = question.strip().lower()
        processed = ' '.join(processed.split())  # Normalize whitespace

        latency = (time.time() - start) * 1000
        return {
            "original": question,
            "processed": processed,
            "latency_ms": latency
        }

    def retrieve_context(self, query, context):
        """Step 2: Retrieve relevant context"""
        start = time.time()

        # In production, this would query a vector database
        # For this lab, we use the provided context
        relevant_context = context

        latency = (time.time() - start) * 1000
        return {
            "context": relevant_context,
            "context_tokens": len(relevant_context.split()),
            "latency_ms": latency
        }

    def generate_response(self, question, context):
        """Step 3: Generate response using LLM"""
        start = time.time()

        system_prompt = """You are a helpful assistant. Answer based on the context provided."""
        user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=self.endpoint_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )

        latency = (time.time() - start) * 1000
        return {
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "latency_ms": latency
        }

    def postprocess_response(self, response):
        """Step 4: Post-process the response"""
        start = time.time()

        # Simulate post-processing (formatting, safety checks, etc.)
        processed = response.strip()

        latency = (time.time() - start) * 1000
        return {
            "final_response": processed,
            "latency_ms": latency
        }

    def run(self, question, context):
        """Execute the full Q&A pipeline with manual tracing"""
        trace_id = str(uuid.uuid4())[:8]
        trace_record = {"trace_id": trace_id, "steps": []}

        # Step 1: Preprocess
        preprocessed = self.preprocess_query(question)
        trace_record["steps"].append({"step": "preprocess", "latency_ms": preprocessed["latency_ms"]})

        # Step 2: Retrieve context
        retrieved = self.retrieve_context(preprocessed["processed"], context)
        trace_record["steps"].append({"step": "retrieve", "latency_ms": retrieved["latency_ms"]})

        # Step 3: Generate response
        generated = self.generate_response(question, retrieved["context"])
        trace_record["steps"].append({"step": "generate", "latency_ms": generated["latency_ms"]})

        # Step 4: Post-process
        final = self.postprocess_response(generated["response"])
        trace_record["steps"].append({"step": "postprocess", "latency_ms": final["latency_ms"]})

        # Calculate total latency
        total_latency = sum(step["latency_ms"] for step in trace_record["steps"])
        trace_record["total_latency_ms"] = total_latency

        # Store trace
        self.trace_log.append(trace_record)

        return {
            "trace_id": trace_id,
            "answer": final["final_response"],
            "total_latency_ms": total_latency,
            "step_latencies": {
                "preprocess": preprocessed["latency_ms"],
                "retrieve": retrieved["latency_ms"],
                "generate": generated["latency_ms"],
                "postprocess": final["latency_ms"]
            },
            "tokens": generated["tokens"]
        }

# Initialize traced agent
traced_agent = TracedQAAgent(ENDPOINT_NAME, client)
print("✅ Traced Q&A Agent initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Running Traced Queries
# MAGIC
# MAGIC **What we're doing:** Executing queries through our `TracedQAAgent` to collect detailed timing data for each step of the pipeline.
# MAGIC
# MAGIC **How the tracing works:**
# MAGIC
# MAGIC 1. **Trace ID generation**: Each query gets a unique ID for correlation:
# MAGIC    ```python
# MAGIC    trace_id = str(uuid.uuid4())[:8]  # e.g., "a1b2c3d4"
# MAGIC    ```
# MAGIC
# MAGIC 2. **Step-by-step timing**: Each pipeline step is timed:
# MAGIC    ```python
# MAGIC    start = time.time()
# MAGIC    result = self.preprocess_query(question)
# MAGIC    latency_ms = (time.time() - start) * 1000
# MAGIC    trace_record["steps"].append({"step": "preprocess", "latency_ms": latency_ms})
# MAGIC    ```
# MAGIC
# MAGIC 3. **Pipeline steps traced**:
# MAGIC    | Step | What it does | Typical time |
# MAGIC    |------|-------------|--------------|
# MAGIC    | preprocess | Clean/normalize question | 1-5ms |
# MAGIC    | retrieve | Find relevant context | 5-20ms |
# MAGIC    | generate | LLM API call | 200-2000ms |
# MAGIC    | postprocess | Format response | 1-5ms |
# MAGIC
# MAGIC 4. **Trace log storage**: All traces stored in `traced_agent.trace_log` for analysis
# MAGIC
# MAGIC **Why trace?** The LLM call typically dominates latency (>95%), but tracing helps identify when other steps become bottlenecks (e.g., slow retrieval from a vector database).

# COMMAND ----------

# =============================================================================
# RUN TRACED QUERIES
# =============================================================================
# Execute queries with manual tracing

print("🔄 Running traced queries...")

traced_results = []

# Run a few traced queries
for sample in eval_data[:5]:
    try:
        result = traced_agent.run(sample["question"], sample["context"])

        traced_results.append({
            "id": sample["id"],
            "complexity": sample["complexity"],
            "trace_id": result["trace_id"],
            **result["step_latencies"],
            "total_latency": result["total_latency_ms"],
            "tokens": result["tokens"]
        })

        print(f"   ✓ {sample['id']} [trace:{result['trace_id']}]: {result['total_latency_ms']:.0f}ms total")

    except Exception as e:
        print(f"   ✗ {sample['id']}: {str(e)}")

    time.sleep(1)

# Analyze step latencies
traced_df = pd.DataFrame(traced_results)

print(f"\n📊 Step Latency Analysis:")
print(f"   Preprocess:  {traced_df['preprocess'].mean():.2f}ms avg")
print(f"   Retrieve:    {traced_df['retrieve'].mean():.2f}ms avg")
print(f"   Generate:    {traced_df['generate'].mean():.2f}ms avg")
print(f"   Postprocess: {traced_df['postprocess'].mean():.2f}ms avg")
print(f"\n   LLM generation accounts for {traced_df['generate'].mean()/traced_df['total_latency'].mean()*100:.1f}% of total latency")

# Display trace log from agent
print(f"\n📋 Trace Log ({len(traced_agent.trace_log)} traces recorded):")
for trace in traced_agent.trace_log[:3]:
    print(f"   Trace {trace['trace_id']}: {trace['total_latency_ms']:.0f}ms")
    for step in trace['steps']:
        print(f"      └─ {step['step']}: {step['latency_ms']:.2f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 6: Anomaly Detection and Alerting
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Implement statistical anomaly detection
# MAGIC 2. Configure alert thresholds based on baselines
# MAGIC 3. Create alerting rules for production monitoring
# MAGIC 4. Apply calibrated alerting strategies from Chapter 8
# MAGIC
# MAGIC Effective alerting requires careful calibration to avoid alert fatigue while catching real issues.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6.1: Statistical Anomaly Detection
# MAGIC
# MAGIC We'll implement anomaly detection using statistical methods:
# MAGIC - **Z-score detection**: Identify values beyond N standard deviations
# MAGIC - **IQR method**: Detect outliers using interquartile range
# MAGIC - **Rolling window analysis**: Detect trend changes over time

# COMMAND ----------

# =============================================================================
# ANOMALY DETECTION IMPLEMENTATION
# =============================================================================

class AnomalyDetector:
    """
    Statistical anomaly detection for LLM monitoring metrics.
    Implements multiple detection methods for robust alerting.
    """

    def __init__(self, baseline_df):
        """Initialize with baseline data for threshold calculation."""
        self.baseline = baseline_df
        self.thresholds = self._calculate_thresholds()

    def _calculate_thresholds(self):
        """Calculate detection thresholds from baseline data."""
        latency = self.baseline['latency_ms']
        tokens = self.baseline['total_tokens']

        return {
            'latency': {
                'mean': latency.mean(),
                'std': latency.std(),
                'p95': latency.quantile(0.95),
                'p99': latency.quantile(0.99),
                'iqr_low': latency.quantile(0.25) - 1.5 * (latency.quantile(0.75) - latency.quantile(0.25)),
                'iqr_high': latency.quantile(0.75) + 1.5 * (latency.quantile(0.75) - latency.quantile(0.25))
            },
            'tokens': {
                'mean': tokens.mean(),
                'std': tokens.std(),
                'p95': tokens.quantile(0.95),
                'p99': tokens.quantile(0.99)
            },
            'error_rate': {
                'baseline': (self.baseline['status'] == 'error').mean(),
                'threshold': max(0.05, (self.baseline['status'] == 'error').mean() * 2)
            }
        }

    def detect_latency_anomaly(self, value, method='zscore', threshold=3):
        """Detect if a latency value is anomalous."""
        if method == 'zscore':
            z = (value - self.thresholds['latency']['mean']) / self.thresholds['latency']['std']
            return abs(z) > threshold, z
        elif method == 'percentile':
            return value > self.thresholds['latency']['p99'], value / self.thresholds['latency']['p99']
        elif method == 'iqr':
            return value > self.thresholds['latency']['iqr_high'], value / self.thresholds['latency']['iqr_high']

    def detect_token_anomaly(self, value, threshold=2):
        """Detect if token usage is anomalous."""
        z = (value - self.thresholds['tokens']['mean']) / self.thresholds['tokens']['std']
        return abs(z) > threshold, z

    def detect_error_rate_anomaly(self, current_error_rate):
        """Detect if error rate exceeds threshold."""
        return current_error_rate > self.thresholds['error_rate']['threshold'], current_error_rate

    def analyze_batch(self, df):
        """Analyze a batch of data for anomalies."""
        anomalies = {
            'latency_anomalies': [],
            'token_anomalies': [],
            'error_rate_anomaly': False
        }

        for idx, row in df.iterrows():
            is_latency_anomaly, score = self.detect_latency_anomaly(row['latency_ms'])
            if is_latency_anomaly:
                anomalies['latency_anomalies'].append({
                    'request_id': row['request_id'],
                    'latency_ms': row['latency_ms'],
                    'z_score': score
                })

            is_token_anomaly, score = self.detect_token_anomaly(row['total_tokens'])
            if is_token_anomaly:
                anomalies['token_anomalies'].append({
                    'request_id': row['request_id'],
                    'total_tokens': row['total_tokens'],
                    'z_score': score
                })

        # Check overall error rate
        current_error_rate = (df['status'] == 'error').mean()
        anomalies['error_rate_anomaly'], _ = self.detect_error_rate_anomaly(current_error_rate)
        anomalies['current_error_rate'] = current_error_rate

        return anomalies

# Initialize detector with baseline data
detector = AnomalyDetector(traffic_analysis_df)

print("✅ Anomaly Detector initialized with baseline thresholds:")
print(f"\n   Latency Thresholds:")
print(f"      Mean: {detector.thresholds['latency']['mean']:.2f} ms")
print(f"      Std: {detector.thresholds['latency']['std']:.2f} ms")
print(f"      P95: {detector.thresholds['latency']['p95']:.2f} ms")
print(f"      P99: {detector.thresholds['latency']['p99']:.2f} ms")
print(f"\n   Token Thresholds:")
print(f"      Mean: {detector.thresholds['tokens']['mean']:.1f}")
print(f"      P95: {detector.thresholds['tokens']['p95']:.1f}")
print(f"\n   Error Rate Threshold: {detector.thresholds['error_rate']['threshold']*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Running Anomaly Detection
# MAGIC
# MAGIC **What we're doing:** Scanning our traffic data to find requests that deviate significantly from normal patterns.
# MAGIC
# MAGIC **How the `analyze_batch()` method works:**
# MAGIC
# MAGIC 1. **Iterate through each request**:
# MAGIC    ```python
# MAGIC    for idx, row in df.iterrows():
# MAGIC        is_anomaly, z_score = self.detect_latency_anomaly(row['latency_ms'])
# MAGIC    ```
# MAGIC
# MAGIC 2. **Z-score calculation** for each metric:
# MAGIC    ```python
# MAGIC    z_score = (value - mean) / std_dev
# MAGIC    # If z_score > 3, the value is 3+ standard deviations from mean
# MAGIC    ```
# MAGIC
# MAGIC 3. **Anomaly types detected**:
# MAGIC
# MAGIC    | Type | Detection Method | Threshold |
# MAGIC    |------|-----------------|-----------|
# MAGIC    | Latency | Z-score > 3 | ~99.7% of normal data is below this |
# MAGIC    | Tokens | Z-score > 2 | ~95% of normal data is below this |
# MAGIC    | Error Rate | Rate > 5% | Fixed threshold based on SLA |
# MAGIC
# MAGIC 4. **Output structure**:
# MAGIC    ```python
# MAGIC    {
# MAGIC        'latency_anomalies': [{'request_id': 'x', 'latency_ms': 5000, 'z_score': 4.2}],
# MAGIC        'token_anomalies': [...],
# MAGIC        'error_rate_anomaly': True/False,
# MAGIC        'current_error_rate': 0.03
# MAGIC    }
# MAGIC    ```
# MAGIC
# MAGIC **Why Z-scores?** They're distribution-agnostic and automatically adapt to your baseline. A Z-score of 3 means "this is extremely unusual" regardless of whether your mean latency is 100ms or 1000ms.

# COMMAND ----------

# =============================================================================
# RUN ANOMALY DETECTION
# =============================================================================

# Analyze our traffic data for anomalies
anomalies = detector.analyze_batch(traffic_analysis_df)

print("🔍 Anomaly Detection Results:")
print(f"\n   Latency Anomalies: {len(anomalies['latency_anomalies'])}")
if anomalies['latency_anomalies']:
    print("   Top 5 latency anomalies:")
    for a in sorted(anomalies['latency_anomalies'], key=lambda x: x['z_score'], reverse=True)[:5]:
        print(f"      {a['request_id']}: {a['latency_ms']:.0f}ms (z={a['z_score']:.2f})")

print(f"\n   Token Anomalies: {len(anomalies['token_anomalies'])}")
if anomalies['token_anomalies']:
    print("   Top 5 token anomalies:")
    for a in sorted(anomalies['token_anomalies'], key=lambda x: x['z_score'], reverse=True)[:5]:
        print(f"      {a['request_id']}: {a['total_tokens']} tokens (z={a['z_score']:.2f})")

print(f"\n   Error Rate Anomaly: {'⚠️ YES' if anomalies['error_rate_anomaly'] else '✅ NO'}")
print(f"   Current Error Rate: {anomalies['current_error_rate']*100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6.2: Configure Alert Rules
# MAGIC
# MAGIC Following Chapter 8 best practices for calibrated alerting:
# MAGIC - **Severity levels**: Critical, Warning, Info
# MAGIC - **Cooldown periods**: Prevent alert storms
# MAGIC - **Aggregation windows**: Reduce noise from transient spikes

# COMMAND ----------

# =============================================================================
# ALERT CONFIGURATION
# =============================================================================

class AlertManager:
    """
    Alert management system with calibrated thresholds and cooldowns.
    Implements Chapter 8 best practices for production alerting.
    """

    def __init__(self, detector):
        self.detector = detector
        self.alert_history = []
        self.cooldown_periods = {
            'critical': 300,   # 5 minutes
            'warning': 900,    # 15 minutes
            'info': 3600       # 1 hour
        }

        # Define alert rules
        self.rules = {
            'latency_critical': {
                'condition': lambda m: m['latency_p99'] > detector.thresholds['latency']['p99'] * 2,
                'severity': 'critical',
                'message': 'P99 latency exceeds 2x baseline'
            },
            'latency_warning': {
                'condition': lambda m: m['latency_p95'] > detector.thresholds['latency']['p95'] * 1.5,
                'severity': 'warning',
                'message': 'P95 latency exceeds 1.5x baseline'
            },
            'error_rate_critical': {
                'condition': lambda m: m['error_rate'] > 0.10,
                'severity': 'critical',
                'message': 'Error rate exceeds 10%'
            },
            'error_rate_warning': {
                'condition': lambda m: m['error_rate'] > 0.05,
                'severity': 'warning',
                'message': 'Error rate exceeds 5%'
            },
            'token_spike': {
                'condition': lambda m: m['avg_total_tokens'] > detector.thresholds['tokens']['p95'],
                'severity': 'warning',
                'message': 'Average token usage exceeds P95 baseline'
            },
            'quality_degradation': {
                'condition': lambda m: m.get('avg_quality_score', 1) < 0.7,
                'severity': 'warning',
                'message': 'Average quality score below threshold'
            }
        }

    def evaluate_rules(self, metrics):
        """Evaluate all alert rules against current metrics."""
        triggered_alerts = []

        for rule_name, rule in self.rules.items():
            try:
                if rule['condition'](metrics):
                    triggered_alerts.append({
                        'rule': rule_name,
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                print(f"Error evaluating rule {rule_name}: {e}")

        return triggered_alerts

    def should_alert(self, alert):
        """Check if alert should fire based on cooldown."""
        cooldown = self.cooldown_periods[alert['severity']]

        # Check recent alerts of same type
        for hist in self.alert_history:
            if hist['rule'] == alert['rule']:
                time_diff = (alert['timestamp'] - hist['timestamp']).total_seconds()
                if time_diff < cooldown:
                    return False

        return True

    def process_alerts(self, metrics):
        """Process metrics and generate alerts."""
        triggered = self.evaluate_rules(metrics)
        fired_alerts = []

        for alert in triggered:
            if self.should_alert(alert):
                fired_alerts.append(alert)
                self.alert_history.append(alert)

        return fired_alerts

# Initialize alert manager
alert_manager = AlertManager(detector)

# Evaluate current metrics
alerts = alert_manager.process_alerts(current_metrics)

print("🚨 Alert Evaluation Results:")
if alerts:
    for alert in alerts:
        severity_icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}
        print(f"   {severity_icon[alert['severity']]} [{alert['severity'].upper()}] {alert['message']}")
else:
    print("   ✅ No alerts triggered - all metrics within normal range")

print(f"\n📋 Configured Alert Rules:")
for rule_name, rule in alert_manager.rules.items():
    print(f"   - {rule_name} ({rule['severity']}): {rule['message']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 7: Cost Optimization Strategies
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Analyze token usage patterns for cost drivers
# MAGIC 2. Implement prompt optimization techniques
# MAGIC 3. Apply context control strategies
# MAGIC 4. Validate improvements with evaluation runs
# MAGIC
# MAGIC Cost optimization is critical for sustainable LLM operations at scale.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7.1: Analyze Cost Drivers
# MAGIC
# MAGIC We'll identify the main contributors to token usage and cost:
# MAGIC - Long context documents
# MAGIC - Verbose prompts
# MAGIC - Unnecessarily detailed responses

# COMMAND ----------

# =============================================================================
# COST ANALYSIS
# =============================================================================

# Analyze token usage patterns
print("💰 Cost Analysis:")

# Token usage by complexity
token_by_complexity = traffic_analysis_df.groupby('complexity').agg({
    'input_tokens': 'mean',
    'output_tokens': 'mean',
    'total_tokens': ['mean', 'sum']
}).round(2)

print("\n   Token Usage by Complexity:")
print(token_by_complexity)

# Token usage by category
token_by_category = traffic_analysis_df.groupby('category').agg({
    'total_tokens': ['mean', 'sum', 'count']
}).round(2)

print("\n   Token Usage by Category:")
print(token_by_category)

# Identify high-cost queries
high_cost_threshold = traffic_analysis_df['total_tokens'].quantile(0.9)
high_cost_queries = traffic_analysis_df[traffic_analysis_df['total_tokens'] > high_cost_threshold]

print(f"\n   High-Cost Queries (>{high_cost_threshold:.0f} tokens): {len(high_cost_queries)}")
print(f"   These represent {len(high_cost_queries)/len(traffic_analysis_df)*100:.1f}% of requests")
print(f"   But consume {high_cost_queries['total_tokens'].sum()/traffic_analysis_df['total_tokens'].sum()*100:.1f}% of tokens")

# Estimate costs (using approximate pricing)
COST_PER_1K_INPUT_TOKENS = 0.0015  # Example pricing
COST_PER_1K_OUTPUT_TOKENS = 0.002

total_input_tokens = traffic_analysis_df['input_tokens'].sum()
total_output_tokens = traffic_analysis_df['output_tokens'].sum()

estimated_cost = (total_input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS +
                  total_output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)

print(f"\n   Estimated Cost for {len(traffic_analysis_df)} requests: ${estimated_cost:.2f}")
print(f"   Average Cost per Request: ${estimated_cost/len(traffic_analysis_df)*1000:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7.2: Implement Prompt Optimization
# MAGIC
# MAGIC We'll implement several prompt optimization techniques:
# MAGIC 1. **Context truncation**: Limit context to relevant portions
# MAGIC 2. **Prompt compression**: Remove redundant instructions
# MAGIC 3. **Response length control**: Set appropriate max_tokens

# COMMAND ----------

# =============================================================================
# PROMPT OPTIMIZATION TECHNIQUES
# =============================================================================

class PromptOptimizer:
    """
    Implements prompt optimization strategies for cost reduction.
    """

    def __init__(self, max_context_tokens=200, max_response_tokens=150):
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens

    def truncate_context(self, context, max_tokens=None):
        """Truncate context to maximum token limit."""
        max_tokens = max_tokens or self.max_context_tokens
        words = context.split()

        # Approximate: 1 token ≈ 0.75 words
        max_words = int(max_tokens * 0.75)

        if len(words) <= max_words:
            return context, False

        truncated = ' '.join(words[:max_words]) + '...'
        return truncated, True

    def compress_prompt(self, system_prompt, user_prompt):
        """Compress prompts by removing redundancy."""
        # Remove excessive whitespace
        system_prompt = ' '.join(system_prompt.split())
        user_prompt = ' '.join(user_prompt.split())

        # Use shorter system prompt
        compressed_system = "Answer questions based on the context. Be concise."

        return compressed_system, user_prompt

    def optimize_query(self, question, context):
        """Apply all optimization techniques."""
        # Truncate context
        truncated_context, was_truncated = self.truncate_context(context)

        # Create optimized prompts
        system_prompt = "Answer based on context. Be concise and accurate."
        user_prompt = f"Context: {truncated_context}\n\nQ: {question}\n\nA:"

        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'context_truncated': was_truncated,
            'max_tokens': self.max_response_tokens
        }

# Initialize optimizer
optimizer = PromptOptimizer(max_context_tokens=150, max_response_tokens=100)

# Test optimization on a sample
sample = eval_data[0]
original_context_tokens = len(sample['context'].split())
optimized = optimizer.optimize_query(sample['question'], sample['context'])

print("🔧 Prompt Optimization Example:")
print(f"\n   Original context tokens: ~{original_context_tokens}")
print(f"   Optimized context tokens: ~{len(optimized['user_prompt'].split())}")
print(f"   Context truncated: {optimized['context_truncated']}")
print(f"   Max response tokens: {optimized['max_tokens']}")
print(f"\n   Optimized system prompt: '{optimized['system_prompt']}'")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparing Optimized vs Baseline Performance
# MAGIC
# MAGIC **What we're doing:** Running the same queries with optimized prompts and comparing token usage, latency, and answer quality against our baseline.
# MAGIC
# MAGIC **How the comparison works:**
# MAGIC
# MAGIC 1. **Run both versions** on the same questions:
# MAGIC    ```python
# MAGIC    baseline_result = query_qa_endpoint(question, context)  # Full prompts
# MAGIC    optimized_result = query_optimized(question, context, optimizer)  # Compressed
# MAGIC    ```
# MAGIC
# MAGIC 2. **Metrics compared**:
# MAGIC    | Metric | Baseline | Optimized | Goal |
# MAGIC    |--------|----------|-----------|------|
# MAGIC    | Input tokens | ~300 | ~150 | 50% reduction |
# MAGIC    | Output tokens | ~200 | ~100 | 50% reduction |
# MAGIC    | Latency | ~500ms | ~350ms | Lower is better |
# MAGIC    | Quality | 0.85 | 0.80+ | Minimal degradation |
# MAGIC
# MAGIC 3. **Cost calculation**:
# MAGIC    ```python
# MAGIC    # Typical pricing: $0.002 per 1K tokens
# MAGIC    baseline_cost = baseline_tokens * 0.002 / 1000
# MAGIC    optimized_cost = optimized_tokens * 0.002 / 1000
# MAGIC    savings = (baseline_cost - optimized_cost) / baseline_cost * 100
# MAGIC    ```
# MAGIC
# MAGIC **The trade-off:** Aggressive optimization saves money but may reduce answer quality. The goal is finding the sweet spot where you save 30-50% on tokens while maintaining >95% of baseline quality.

# COMMAND ----------

# =============================================================================
# COMPARE OPTIMIZED VS BASELINE PERFORMANCE
# =============================================================================

def query_optimized(question, context, optimizer):
    """Query endpoint with optimized prompts."""
    opt = optimizer.optimize_query(question, context)

    start_time = time.time()

    response = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[
            {"role": "system", "content": opt['system_prompt']},
            {"role": "user", "content": opt['user_prompt']}
        ],
        max_tokens=opt['max_tokens'],
        temperature=0.1
    )

    latency_ms = (time.time() - start_time) * 1000

    return {
        "answer": response.choices[0].message.content,
        "latency_ms": latency_ms,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

print("🔄 Comparing baseline vs optimized performance...")

comparison_results = []

for sample in eval_data[:5]:
    try:
        # Baseline query
        baseline = query_qa_endpoint(sample["question"], sample["context"])

        # Optimized query
        optimized_result = query_optimized(sample["question"], sample["context"], optimizer)

        comparison_results.append({
            "id": sample["id"],
            "baseline_tokens": baseline["total_tokens"],
            "optimized_tokens": optimized_result["total_tokens"],
            "token_reduction": baseline["total_tokens"] - optimized_result["total_tokens"],
            "baseline_latency": baseline["latency_ms"],
            "optimized_latency": optimized_result["latency_ms"]
        })

        print(f"   ✓ {sample['id']}: {baseline['total_tokens']} → {optimized_result['total_tokens']} tokens")

    except Exception as e:
        print(f"   ✗ {sample['id']}: {str(e)}")

    time.sleep(1)

# Analyze results
comparison_df = pd.DataFrame(comparison_results)

print(f"\n📊 Optimization Results:")
print(f"   Average Token Reduction: {comparison_df['token_reduction'].mean():.1f} tokens ({comparison_df['token_reduction'].mean()/comparison_df['baseline_tokens'].mean()*100:.1f}%)")
print(f"   Average Latency Change: {(comparison_df['optimized_latency'].mean() - comparison_df['baseline_latency'].mean()):.1f}ms")

# Estimate cost savings
baseline_cost = comparison_df['baseline_tokens'].sum() / 1000 * 0.002
optimized_cost = comparison_df['optimized_tokens'].sum() / 1000 * 0.002
savings = baseline_cost - optimized_cost

print(f"\n   Estimated Cost Savings: ${savings:.4f} for {len(comparison_df)} queries")
print(f"   Projected Monthly Savings (10K queries/day): ${savings/len(comparison_df)*10000*30:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part 8: Validation and Periodic Review Cycles
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Validate optimizations with repeatable evaluation runs
# MAGIC 2. Establish baseline metrics for ongoing comparison
# MAGIC 3. Implement periodic review workflows
# MAGIC 4. Create evaluation versioning for reproducibility
# MAGIC
# MAGIC Following Chapter 8 best practices, we'll ensure our improvements are measurable and sustainable.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8.1: Validate Optimizations
# MAGIC
# MAGIC We'll run a comprehensive evaluation to validate that our optimizations maintain quality while reducing costs.

# COMMAND ----------

# =============================================================================
# VALIDATION EVALUATION RUN
# =============================================================================

print("🔄 Running validation evaluation...")

validation_results = []

for sample in eval_data:
    try:
        # Run optimized query
        result = query_optimized(sample["question"], sample["context"], optimizer)

        validation_results.append({
            "id": sample["id"],
            "question": sample["question"],
            "context": sample["context"],
            "ground_truth": sample["ground_truth"],
            "output": result["answer"],
            "category": sample["category"],
            "complexity": sample["complexity"],
            "latency_ms": result["latency_ms"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"]
        })

    except Exception as e:
        print(f"   ✗ {sample['id']}: {str(e)}")

    time.sleep(0.5)

validation_df = pd.DataFrame(validation_results)

print(f"\n✅ Validation complete: {len(validation_df)} samples evaluated")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logging Validation Results to MLflow
# MAGIC
# MAGIC **What we're doing:** Creating a new MLflow run for our optimized model to enable side-by-side comparison with the baseline.
# MAGIC
# MAGIC **How the comparison works in MLflow:**
# MAGIC
# MAGIC 1. **Separate runs for each configuration**:
# MAGIC    - `baseline_evaluation`: Original prompts, no optimization
# MAGIC    - `optimized_validation`: Compressed prompts, token limits
# MAGIC
# MAGIC 2. **Parameters logged** (what we changed):
# MAGIC    ```python
# MAGIC    mlflow.log_param("optimization_type", "prompt_compression")
# MAGIC    mlflow.log_param("max_context_tokens", 150)  # Was unlimited
# MAGIC    mlflow.log_param("max_response_tokens", 100)  # Was unlimited
# MAGIC    ```
# MAGIC
# MAGIC 3. **Metrics logged using new GenAI API**:
# MAGIC    ```python
# MAGIC    # Run scorers on each sample
# MAGIC    g_score = groundedness(inputs=inputs, outputs=outputs)
# MAGIC    c_score = answer_correctness(inputs=inputs, outputs=outputs, expectations=expectations)
# MAGIC
# MAGIC    # Log aggregate metrics
# MAGIC    mlflow.log_metric("groundedness_mean", avg_groundedness)
# MAGIC    mlflow.log_metric("answer_correctness_mean", avg_correctness)
# MAGIC    ```
# MAGIC
# MAGIC 4. **MLflow UI comparison**:
# MAGIC    - Navigate to Experiments → Select both runs → Compare
# MAGIC    - See metrics side-by-side in charts
# MAGIC    - Identify if quality dropped with optimization
# MAGIC
# MAGIC **Decision framework:**
# MAGIC - If quality drops <5% and tokens drop >30% → Accept optimization
# MAGIC - If quality drops >10% → Reject, try less aggressive settings
# MAGIC - If quality unchanged → Great! Deploy optimized version

# COMMAND ----------

# =============================================================================
# LOG VALIDATION RUN TO MLFLOW
# =============================================================================

with mlflow.start_run(run_name="optimized_validation") as run:

    # Log parameters
    mlflow.log_param("endpoint_name", ENDPOINT_NAME)
    mlflow.log_param("optimization_type", "prompt_compression")
    mlflow.log_param("max_context_tokens", optimizer.max_context_tokens)
    mlflow.log_param("max_response_tokens", optimizer.max_response_tokens)
    mlflow.log_param("num_samples", len(validation_df))

    # Run evaluation using new GenAI API scorers
    print("🔄 Running MLflow evaluation on optimized results...")
    print("   (Using new MLflow 3.4+ GenAI API)")

    # Prepare data and run scorers
    val_groundedness_scores = []
    val_correctness_scores = []

    for i, row in validation_df.iterrows():
        try:
            inputs = {"question": row["question"], "context": row["context"]}
            outputs = row["output"]
            expectations = {"ground_truth": row["ground_truth"]}

            g_score = groundedness(inputs=inputs, outputs=outputs)
            c_score = answer_correctness(inputs=inputs, outputs=outputs, expectations=expectations)

            val_groundedness_scores.append(g_score)
            val_correctness_scores.append(c_score)
        except Exception as e:
            print(f"   Warning: Error evaluating sample {i}: {e}")
            val_groundedness_scores.append(3)
            val_correctness_scores.append(3)
        time.sleep(0.5)  # Rate limiting

    # Calculate and log aggregate metrics
    avg_groundedness = sum(val_groundedness_scores) / len(val_groundedness_scores)
    avg_correctness = sum(val_correctness_scores) / len(val_correctness_scores)

    mlflow.log_metric("groundedness_mean", avg_groundedness)
    mlflow.log_metric("answer_correctness_mean", avg_correctness)

    # Log performance metrics
    mlflow.log_metric("avg_latency_ms", validation_df["latency_ms"].mean())
    mlflow.log_metric("p95_latency_ms", validation_df["latency_ms"].quantile(0.95))
    mlflow.log_metric("avg_total_tokens", validation_df["total_tokens"].mean())
    mlflow.log_metric("total_tokens_consumed", validation_df["total_tokens"].sum())

    # Add scores to dataframe
    validation_df["groundedness_score"] = val_groundedness_scores
    validation_df["correctness_score"] = val_correctness_scores

    # Log artifacts
    validation_df.to_csv("/tmp/validation_results.csv", index=False)
    mlflow.log_artifact("/tmp/validation_results.csv")

    print(f"\n✅ Validation logged to MLflow")
    print(f"   Run ID: {run.info.run_id}")

    # Compare with baseline
    print(f"\n📊 Validation Metrics:")
    print(f"   groundedness_mean: {avg_groundedness:.2f}")
    print(f"   answer_correctness_mean: {avg_correctness:.2f}")
    print(f"   avg_latency_ms: {validation_df['latency_ms'].mean():.2f}")
    print(f"   avg_total_tokens: {validation_df['total_tokens'].mean():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8.2: Establish Review Cycle Workflow
# MAGIC
# MAGIC We'll create a structured review workflow that can be run periodically to:
# MAGIC - Compare current performance against baselines
# MAGIC - Detect drift in quality or performance
# MAGIC - Generate actionable reports

# COMMAND ----------

# =============================================================================
# PERIODIC REVIEW WORKFLOW
# =============================================================================

class PeriodicReviewWorkflow:
    """
    Implements a structured review workflow for LLM monitoring.
    Following Chapter 8 best practices for periodic evaluation.
    """

    def __init__(self, baseline_metrics, alert_manager):
        self.baseline = baseline_metrics
        self.alert_manager = alert_manager
        self.review_history = []

    def run_review(self, current_df, review_name="periodic_review"):
        """Execute a complete review cycle."""
        review_timestamp = datetime.now()

        # Calculate current metrics
        current_metrics = calculate_monitoring_metrics(current_df)

        # Compare with baseline
        comparison = self._compare_metrics(current_metrics)

        # Check for alerts
        alerts = self.alert_manager.evaluate_rules(current_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(comparison, alerts)

        # Create review report
        report = {
            "review_name": review_name,
            "timestamp": review_timestamp,
            "current_metrics": current_metrics,
            "baseline_comparison": comparison,
            "alerts": alerts,
            "recommendations": recommendations,
            "status": "HEALTHY" if not alerts else "NEEDS_ATTENTION"
        }

        self.review_history.append(report)

        return report

    def _compare_metrics(self, current):
        """Compare current metrics against baseline."""
        comparison = {}

        for key in ['latency_p50', 'latency_p95', 'avg_total_tokens', 'error_rate']:
            if key in self.baseline and key in current:
                baseline_val = self.baseline[key]
                current_val = current[key]

                if baseline_val > 0:
                    pct_change = (current_val - baseline_val) / baseline_val * 100
                else:
                    pct_change = 0

                comparison[key] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change_pct": pct_change,
                    "status": "OK" if abs(pct_change) < 20 else "DEGRADED" if pct_change > 0 else "IMPROVED"
                }

        return comparison

    def _generate_recommendations(self, comparison, alerts):
        """Generate actionable recommendations."""
        recommendations = []

        # Check latency
        if comparison.get('latency_p95', {}).get('status') == 'DEGRADED':
            recommendations.append({
                "priority": "HIGH",
                "area": "Latency",
                "action": "Investigate latency increase. Consider scaling up or optimizing prompts."
            })

        # Check token usage
        if comparison.get('avg_total_tokens', {}).get('status') == 'DEGRADED':
            recommendations.append({
                "priority": "MEDIUM",
                "area": "Cost",
                "action": "Token usage increased. Review prompt templates and context lengths."
            })

        # Check error rate
        if comparison.get('error_rate', {}).get('status') == 'DEGRADED':
            recommendations.append({
                "priority": "HIGH",
                "area": "Reliability",
                "action": "Error rate increased. Check endpoint health and input validation."
            })

        # Add alert-based recommendations
        for alert in alerts:
            if alert['severity'] == 'critical':
                recommendations.append({
                    "priority": "CRITICAL",
                    "area": alert['rule'],
                    "action": f"Address immediately: {alert['message']}"
                })

        return recommendations

# Initialize review workflow
review_workflow = PeriodicReviewWorkflow(current_metrics, alert_manager)

# Run a review
review_report = review_workflow.run_review(traffic_analysis_df, "lab_final_review")

print("📋 Periodic Review Report")
print("=" * 50)
print(f"\n   Review: {review_report['review_name']}")
print(f"   Timestamp: {review_report['timestamp']}")
print(f"   Status: {review_report['status']}")

print(f"\n   Baseline Comparison:")
for metric, data in review_report['baseline_comparison'].items():
    print(f"      {metric}: {data['current']:.2f} ({data['change_pct']:+.1f}%) - {data['status']}")

if review_report['recommendations']:
    print(f"\n   Recommendations:")
    for rec in review_report['recommendations']:
        print(f"      [{rec['priority']}] {rec['area']}: {rec['action']}")
else:
    print(f"\n   ✅ No recommendations - system performing within expected parameters")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generating the Review Report
# MAGIC
# MAGIC **What we're doing:** Creating a human-readable report that summarizes the review findings for stakeholders who may not be technical.
# MAGIC
# MAGIC **Report structure:**
# MAGIC
# MAGIC 1. **Executive Summary**: High-level health status
# MAGIC    - Overall health: HEALTHY / NEEDS ATTENTION / CRITICAL
# MAGIC    - Review period and sample size
# MAGIC    - One-line summary of key findings
# MAGIC
# MAGIC 2. **Key Metrics Table**: Current vs baseline comparison
# MAGIC    ```
# MAGIC    ✅ latency_p50: 245ms (baseline: 240ms, change: +2.1%)
# MAGIC    ⚠️ latency_p95: 890ms (baseline: 720ms, change: +23.6%)
# MAGIC    ✅ error_rate: 1.2% (baseline: 1.5%, change: -20.0%)
# MAGIC    ```
# MAGIC
# MAGIC 3. **Active Alerts**: Any triggered alert rules
# MAGIC    - Severity level (CRITICAL/WARNING/INFO)
# MAGIC    - Rule that triggered
# MAGIC    - Specific message
# MAGIC
# MAGIC 4. **Recommendations**: Prioritized action items
# MAGIC    - CRITICAL: Address immediately
# MAGIC    - HIGH: Address within 24 hours
# MAGIC    - MEDIUM: Address within 1 week
# MAGIC
# MAGIC **Why structured reports matter (Chapter 8 best practice):**
# MAGIC - Executives need summaries, not raw data
# MAGIC - Consistent format enables trend tracking over time
# MAGIC - Recommendations drive action, not just awareness

# COMMAND ----------

# =============================================================================
# GENERATE REVIEW REPORT
# =============================================================================

print("📋 Generating Review Report...")
print("=" * 60)
print(f"PERIODIC REVIEW REPORT")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

print(f"\n📊 EXECUTIVE SUMMARY")
print(f"   Review Period: Last 7 days")
print(f"   Total Requests Analyzed: {len(traffic_analysis_df)}")
print(f"   Overall Health: {'✅ HEALTHY' if len(review_report['alerts']) == 0 else '⚠️ NEEDS ATTENTION'}")

print(f"\n📈 KEY METRICS")
for metric, data in review_report['baseline_comparison'].items():
    status_icon = "✅" if data['status'] == 'ok' else "⚠️" if data['status'] == 'warning' else "🔴"
    print(f"   {status_icon} {metric}: {data['current']:.2f} (baseline: {data['baseline']:.2f}, change: {data['change_pct']:+.1f}%)")

print(f"\n🚨 ALERTS ({len(review_report['alerts'])})")
if review_report['alerts']:
    for alert in review_report['alerts']:
        print(f"   [{alert['severity'].upper()}] {alert['rule_name']}: {alert['message']}")
else:
    print("   No active alerts")

print(f"\n💡 RECOMMENDATIONS ({len(review_report['recommendations'])})")
if review_report['recommendations']:
    for i, rec in enumerate(review_report['recommendations'], 1):
        print(f"   {i}. [{rec['priority'].upper()}] {rec['area']}: {rec['action']}")
else:
    print("   No recommendations - system performing optimally")

print("\n" + "=" * 60)
print("END OF REPORT")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8.3: Create Evaluation Dataset Versioning
# MAGIC
# MAGIC **What we're doing:** Saving a timestamped version of our evaluation dataset so we can reproduce results and track changes over time.
# MAGIC
# MAGIC **How versioning works:**
# MAGIC
# MAGIC 1. **Generate version string** from timestamp:
# MAGIC    ```python
# MAGIC    version = datetime.now().strftime("%Y%m%d_%H%M%S")
# MAGIC    # Result: "20241130_143022"
# MAGIC    ```
# MAGIC
# MAGIC 2. **Save versioned Delta table**:
# MAGIC    ```python
# MAGIC    versioned_table_name = f"{CATALOG}.{SCHEMA}.evaluation_dataset_v{version}"
# MAGIC    spark_df.write.saveAsTable(versioned_table_name)
# MAGIC    # Creates: main.llm_monitoring_lab.evaluation_dataset_v20241130_143022
# MAGIC    ```
# MAGIC
# MAGIC 3. **Log to MLflow** for discoverability:
# MAGIC    ```python
# MAGIC    mlflow.log_param("dataset_version", version)
# MAGIC    mlflow.log_param("dataset_table", versioned_table_name)
# MAGIC    mlflow.log_artifact("eval_dataset.csv")  # Downloadable copy
# MAGIC    ```
# MAGIC
# MAGIC **Why version datasets?**
# MAGIC - **Reproducibility**: "What data did we use for the March evaluation?"
# MAGIC - **Debugging**: "Did the dataset change, or did the model regress?"
# MAGIC - **Compliance**: Audit trails for regulated industries
# MAGIC - **A/B testing**: Compare model performance on identical datasets
# MAGIC
# MAGIC **Best practice:** Always version both the dataset AND the evaluation results together. Link them via MLflow run IDs.

# COMMAND ----------

# =============================================================================
# EVALUATION DATASET VERSIONING
# =============================================================================

# Save versioned evaluation dataset
version = datetime.now().strftime("%Y%m%d_%H%M%S")
versioned_table_name = f"{CATALOG}.{SCHEMA}.evaluation_dataset_v{version}"

# Save to Delta table with version
eval_spark_df = spark.createDataFrame(eval_df)
eval_spark_df.write.mode("overwrite").saveAsTable(versioned_table_name)

print(f"✅ Evaluation dataset versioned: {versioned_table_name}")

# Log version to MLflow
with mlflow.start_run(run_name=f"dataset_version_{version}"):
    mlflow.log_param("dataset_version", version)
    mlflow.log_param("dataset_table", versioned_table_name)
    mlflow.log_param("num_samples", len(eval_df))
    mlflow.log_param("categories", list(eval_df['category'].unique()))

    # Log dataset as artifact
    eval_df.to_csv(f"/tmp/eval_dataset_{version}.csv", index=False)
    mlflow.log_artifact(f"/tmp/eval_dataset_{version}.csv")

    print(f"   Logged to MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Lab Summary and Key Takeaways
# MAGIC
# MAGIC ## What You Accomplished
# MAGIC
# MAGIC In this hands-on lab, you successfully:
# MAGIC
# MAGIC ### 1. ✅ Designed an End-to-End Evaluation and Monitoring Workflow
# MAGIC - Set up MLflow experiment tracking
# MAGIC - Created synthetic evaluation datasets
# MAGIC - Established baseline metrics
# MAGIC
# MAGIC ### 2. ✅ Applied MLflow Metrics to Assess LLM Quality and Performance
# MAGIC - Implemented custom evaluation metrics (groundedness, answer correctness)
# MAGIC - Ran comprehensive evaluations with LLM-as-judge
# MAGIC - Logged all results for comparison and tracking
# MAGIC
# MAGIC ### 3. ✅ Used Inference Tables to Diagnose Latency and Cost Issues
# MAGIC - Analyzed traffic patterns and latency distributions
# MAGIC - Identified slow queries and anomalies
# MAGIC - Correlated token usage with performance
# MAGIC
# MAGIC ### 4. ✅ Monitored Multi-Step Agent Workflows
# MAGIC - Implemented tracing for Q&A pipeline steps
# MAGIC - Identified bottlenecks in the workflow
# MAGIC - Measured step-by-step latency contributions
# MAGIC
# MAGIC ### 5. ✅ Configured Anomaly Alerts for Production Monitoring
# MAGIC - Implemented statistical anomaly detection
# MAGIC - Created calibrated alert rules with severity levels
# MAGIC - Applied cooldown periods to prevent alert fatigue
# MAGIC
# MAGIC ### 6. ✅ Applied Optimization Strategies
# MAGIC - Analyzed cost drivers in token usage
# MAGIC - Implemented prompt compression techniques
# MAGIC - Validated improvements with evaluation runs
# MAGIC
# MAGIC ### 7. ✅ Incorporated Chapter 8 Best Practices
# MAGIC - Calibrated metric interpretation
# MAGIC - Baseline-aware monitoring
# MAGIC - Structured alerting strategies
# MAGIC - Periodic review cycles
# MAGIC - Evaluation dataset versioning
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Metrics Summary

# COMMAND ----------

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("=" * 60)
print("           LAB COMPLETION SUMMARY")
print("=" * 60)

print(f"\n📊 Evaluation Metrics:")
print(f"   Samples Evaluated: {len(eval_df)}")
print(f"   Categories: {list(eval_df['category'].unique())}")

print(f"\n⚡ Performance Baseline:")
print(f"   P50 Latency: {current_metrics['latency_p50']:.2f} ms")
print(f"   P95 Latency: {current_metrics['latency_p95']:.2f} ms")
print(f"   Avg Tokens: {current_metrics['avg_total_tokens']:.1f}")

print(f"\n💰 Cost Optimization:")
if len(comparison_df) > 0:
    print(f"   Token Reduction: {comparison_df['token_reduction'].mean():.1f} tokens/query")
    print(f"   Projected Monthly Savings: ${savings/len(comparison_df)*10000*30:.2f}")

print(f"\n🔍 Anomaly Detection:")
print(f"   Latency Anomalies Detected: {len(anomalies['latency_anomalies'])}")
print(f"   Token Anomalies Detected: {len(anomalies['token_anomalies'])}")

print(f"\n📋 Review Status: {review_report['status']}")

print(f"\n✅ Lab completed successfully!")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC After completing this lab, consider:
# MAGIC
# MAGIC 1. **Deploy to Production**: Apply these monitoring patterns to your production LLM systems
# MAGIC 2. **Customize Metrics**: Add domain-specific evaluation metrics for your use case
# MAGIC 3. **Automate Reviews**: Schedule periodic review workflows using Databricks Jobs
# MAGIC 4. **Expand Alerting**: Integrate alerts with your team's notification systems (Slack, PagerDuty)
# MAGIC 5. **Continuous Improvement**: Use evaluation results to guide model fine-tuning or prompt engineering
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC - [Databricks MLflow Documentation](https://docs.databricks.com/mlflow/index.html)
# MAGIC - [Model Serving Best Practices](https://docs.databricks.com/machine-learning/model-serving/index.html)
# MAGIC - [LLM Evaluation with MLflow](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
# MAGIC - [Databricks Monitoring](https://docs.databricks.com/lakehouse-monitoring/index.html)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Congratulations on completing the Hands-On Lab!** 🎉

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC Run the following cell to clean up resources created during this lab.

# COMMAND ----------

# =============================================================================
# CLEANUP RESOURCES
# =============================================================================
# Uncomment and run to delete resources created during this lab

# Delete Model Serving endpoint
# w.serving_endpoints.delete(name=ENDPOINT_NAME)
# print(f"✅ Deleted endpoint: {ENDPOINT_NAME}")

# Delete Delta tables
# spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
# print(f"✅ Deleted schema: {CATALOG}.{SCHEMA}")

# Delete MLflow experiment
# Note: MLflow experiments cannot be permanently deleted, only archived
# mlflow.delete_experiment(experiment_id)
# print(f"✅ Archived experiment: {EXPERIMENT_NAME}")

print("⚠️ Cleanup code is commented out. Uncomment to delete resources.")