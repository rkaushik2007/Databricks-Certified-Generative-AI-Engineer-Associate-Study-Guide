# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 Hands-On Lab: Multi-Agent Workflow with LangChain in Databricks
# MAGIC
# MAGIC **Using Databricks Foundation Models (FREE) or OpenAI**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📍 Scenario
# MAGIC
# MAGIC You are working for an **auto insurance company** that processes thousands of claims submitted as free-form text. The business goal is to transform these unstructured claim descriptions into **consistent, machine-actionable decisions** while maintaining traceability and control over each step of the reasoning process.
# MAGIC
# MAGIC Instead of relying on a single monolithic prompt, you will design a **multi-stage generative AI workflow** composed of specialized, task-aligned components. Each stage is responsible for a clearly defined task and produces structured output that can be validated and consumed downstream.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### The workflow is composed of the following logical stages:
# MAGIC
# MAGIC 1. **Extraction Stage**
# MAGIC    A structured prompt extracts key fields—such as claim type, incident description, and severity—from a raw claim record stored in a Databricks table.
# MAGIC
# MAGIC 2. **Policy Validation Stage**
# MAGIC    A validation step determines whether the associated policy is active and eligible for coverage based on structured inputs.
# MAGIC
# MAGIC 3. **Assessment Stage**
# MAGIC    An LLM-driven reasoning step evaluates the extracted claim data and policy status to determine whether the claim should be auto-approved or flagged for manual review.
# MAGIC
# MAGIC 4. **Resolution Stage**
# MAGIC    A final structured response consolidates the outputs of previous stages into a machine-readable decision record suitable for downstream systems.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This design mirrors **real-world enterprise pipelines** where prompt-task alignment, structured outputs, tool ordering, and modular composition are essential for reliability, auditability, and scalability.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🎯 Objective
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC - **Apply prompt-task alignment** to ensure each stage of a workflow performs the correct LLM task (extraction, classification, or transformation).
# MAGIC
# MAGIC - **Design structured prompts** that produce consistent, machine-readable outputs suitable for automated pipelines.
# MAGIC
# MAGIC - **Translate a business use case** into a multi-stage AI pipeline with clearly defined inputs and outputs.
# MAGIC
# MAGIC - **Define and order reasoning steps** to ensure downstream components receive the correct data at the correct time.
# MAGIC
# MAGIC - **Use LangChain abstractions** to compose prompts, models, and reasoning stages without relying on monolithic prompts.
# MAGIC
# MAGIC - **Execute the workflow in Databricks**, using PySpark for data preparation and OpenAI models for structured reasoning.
# MAGIC
# MAGIC - **Observe how modular design** improves interpretability, debuggability, and control in generative AI systems.
# MAGIC
# MAGIC You will simulate the complete workflow inside a Databricks notebook, focusing on **design correctness** rather than optimization, to reinforce the architectural principles introduced in Chapter 2.
# MAGIC
# MAGIC ---
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📋 Prerequisites
# MAGIC
# MAGIC Before running this lab, ensure you have:
# MAGIC
# MAGIC #### **1. Databricks Environment**
# MAGIC - A Databricks Workspace (Community Edition or paid tier)
# MAGIC - Access to Foundation Model Serving Endpoints (available in most Databricks workspaces)
# MAGIC
# MAGIC #### **2. Model Endpoint Configuration**
# MAGIC
# MAGIC This notebook uses **Databricks Foundation Models** which are FREE and don't require API keys.
# MAGIC
# MAGIC **To configure your model endpoint:**
# MAGIC
# MAGIC 1. In your Databricks workspace, navigate to: **Serving** → **Serving Endpoints**
# MAGIC 2. Look for available Foundation Model endpoints, such as:
# MAGIC    - `databricks-meta-llama-3-3-70b-instruct` (recommended)
# MAGIC    - `databricks-meta-llama-3-1-70b-instruct`
# MAGIC    - `databricks-dbrx-instruct`
# MAGIC    - `databricks-mixtral-8x7b-instruct`
# MAGIC
# MAGIC 3. **Copy the endpoint name** (just the name, NOT the full URL)
# MAGIC
# MAGIC **The notebook is pre-configured with:** `databricks-meta-llama-3-3-70b-instruct`
# MAGIC
# MAGIC - ✅ If this model is available in your workspace: No changes needed!
# MAGIC - ⚙️ If you need a different model: Update the endpoint name in **Step 5** below
# MAGIC
# MAGIC **Important:** Use only the endpoint **name** in your code:
# MAGIC ```python
# MAGIC # ✅ CORRECT
# MAGIC endpoint="databricks-meta-llama-3-3-70b-instruct"
# MAGIC
# MAGIC # ❌ WRONG - Don't use the full URL
# MAGIC endpoint="https://adb-1234567890.15.azuredatabricks.net/..."
# MAGIC ```
# MAGIC
# MAGIC Databricks automatically resolves the endpoint name to the correct URL when running in your workspace.
# MAGIC
# MAGIC #### **3. Authentication**
# MAGIC
# MAGIC When running in Databricks:
# MAGIC - ✅ No API keys needed - Databricks uses your workspace session
# MAGIC - ✅ No URLs needed - Just the endpoint name
# MAGIC - ✅ Completely FREE - Foundation Models are included
# MAGIC
# MAGIC #### **4. Alternative Model Options (Optional)**
# MAGIC
# MAGIC If you prefer to use other models instead of Databricks Foundation Models:
# MAGIC - **OpenAI**: Requires API key and billing setup
# MAGIC - **Azure OpenAI**: Requires Azure subscription
# MAGIC - **Local Ollama**: Requires local installation
# MAGIC
# MAGIC See the model configuration section in **Step 5** for details.
# MAGIC
# MAGIC ---
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔧 Step 1: Install Required Packages
# MAGIC
# MAGIC To work with LangChain agents in a modern, non-deprecated way, you need to install the latest versions of the required libraries.
# MAGIC
# MAGIC **What's being installed:**
# MAGIC
# MAGIC - `langchain-openai`: The modern package for OpenAI integration with LangChain (replaces deprecated `langchain.llms.OpenAI`)
# MAGIC - `langchain`: Core LangChain framework for building agent workflows (v1.0+)
# MAGIC - `langchain-community`: Community-contributed tools and integrations (includes Databricks models)
# MAGIC - `langchain-core`: Core abstractions including prompts, chains, and runnables
# MAGIC - `langgraph`: Modern agent framework (required for `create_agent`)
# MAGIC - `openai`: Official OpenAI Python client
# MAGIC
# MAGIC This ensures you're using the **current, supported APIs** without any deprecated imports.
# MAGIC
# MAGIC > ⚠️ **Note**: After installation, you'll need to restart the Python kernel to ensure the new packages are loaded properly.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 💡 **Model Options Available**
# MAGIC
# MAGIC After installing these packages, you can use:
# MAGIC - ✅ **OpenAI models** (requires API key and billing)
# MAGIC - ✅ **Databricks Foundation Models** (FREE for Databricks users)
# MAGIC - ✅ **Azure OpenAI** (for enterprise users)
# MAGIC - ✅ **Local models via Ollama** (completely free)
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %pip install --upgrade langchain-openai langchain langchain-community langchain-core langgraph openai
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔄 Step 2: Restart the Python Kernel
# MAGIC
# MAGIC After installing or upgrading packages in Databricks, it's important to restart the Python runtime so your notebook picks up the new dependencies cleanly.
# MAGIC
# MAGIC This ensures that:
# MAGIC - All newly installed packages are available in the Python environment
# MAGIC - No conflicts exist between old and new package versions
# MAGIC - Import statements will reference the correct module versions
# MAGIC

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔑 Step 3: API Key Setup (OPTIONAL - Skip if Using Databricks Models)
# MAGIC
# MAGIC > ⚡ **Quick Note**: If you're using **Databricks Foundation Models** (the default in this notebook), you can **SKIP this entire step**. No API key needed!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 When Do You Need This Step?
# MAGIC
# MAGIC **Skip this step if:**
# MAGIC - ✅ You're using Databricks Foundation Models (default option)
# MAGIC - ✅ You want a completely FREE experience
# MAGIC
# MAGIC **Complete this step only if:**
# MAGIC - ❌ You want to use OpenAI models (requires billing)
# MAGIC - ❌ You want to use Azure OpenAI
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📌 How to Get Your OpenAI API Key (Optional)
# MAGIC
# MAGIC 1. Go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
# MAGIC 2. Log in or create an OpenAI account
# MAGIC 3. Click **"Create new secret key"**
# MAGIC 4. Copy the generated key (it starts with `sk-...`)
# MAGIC 5. Keep it safe — you won't see it again!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔐 How to Set the Key in Databricks
# MAGIC
# MAGIC You have **two options** for setting your API key:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Option 1: Quick Setup (Development/Learning)**
# MAGIC
# MAGIC Paste your key directly in the cell below using `os.environ`:
# MAGIC
# MAGIC ```python
# MAGIC import os
# MAGIC os.environ["OPENAI_API_KEY"] = "sk-proj-..."  # Replace with your actual key
# MAGIC ```
# MAGIC
# MAGIC > ⚠️ **Important**: This method is for development and learning purposes only. Never commit API keys to version control!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Option 2: Secure Setup (Production/Best Practice)**
# MAGIC
# MAGIC Use **Databricks Secrets** to securely store your API key:
# MAGIC
# MAGIC **Step 1: Create a secret scope (one-time setup)**
# MAGIC ```bash
# MAGIC # In Databricks CLI or notebook
# MAGIC databricks secrets create-scope --scope my-secrets
# MAGIC ```
# MAGIC
# MAGIC **Step 2: Store your API key**
# MAGIC ```bash
# MAGIC databricks secrets put --scope my-secrets --key openai-api-key
# MAGIC # This will open an editor where you paste your key
# MAGIC ```
# MAGIC
# MAGIC **Step 3: Retrieve the key in your notebook**
# MAGIC ```python
# MAGIC import os
# MAGIC os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="my-secrets", key="openai-api-key")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 **Choose Your Method Below**
# MAGIC

# COMMAND ----------

import os

# ⚡ SKIP THIS CELL if you're using Databricks Foundation Models (the default)

# OPTION 1: Direct setup (for learning/development with OpenAI)
# Uncomment the line below and add your key if you want to use OpenAI:
# os.environ["OPENAI_API_KEY"] = "sk-..."  # ⚠️ Replace with your actual key

# OPTION 2: Secure setup (for production with OpenAI)
# Uncomment the line below:
# os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="my-secrets", key="openai-api-key")

# Verify the key is set (optional)
if os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
    print("✅ OpenAI API key is set correctly")
else:
    print("ℹ️  No OpenAI API key set - that's OK if you're using Databricks Foundation Models!")
    print("   (This is the default and recommended option for this lab)")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🗃️ Step 4: Load Sample Claims Data into a Spark Table
# MAGIC
# MAGIC This step creates a simulated insurance claims dataset using PySpark. This represents the **raw, unstructured data** that your multi-agent workflow will process.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📊 Dataset Structure
# MAGIC
# MAGIC The dataset includes the following fields:
# MAGIC
# MAGIC - **`claim_id`**: Unique identifier for each claim (e.g., C101, C102)
# MAGIC - **`claimant_name`**: Name of the person who filed the claim
# MAGIC - **`damage_description`**: Free-form text describing the incident (unstructured input)
# MAGIC - **`estimated_damage`**: Approximate repair cost in dollars
# MAGIC - **`policy_id`**: Reference to the insurance policy number
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 Why This Matters
# MAGIC
# MAGIC In real-world scenarios, claims arrive as unstructured text from various sources (emails, forms, mobile apps). Your AI pipeline must:
# MAGIC
# MAGIC 1. **Extract** structured information from this raw text
# MAGIC 2. **Validate** that the policy is active
# MAGIC 3. **Assess** whether the claim meets auto-approval criteria
# MAGIC 4. **Resolve** the claim with a final decision
# MAGIC
# MAGIC Once loaded, the data is registered as a temporary SQL view called `claims`, which can be queried by your agents during the workflow.
# MAGIC
# MAGIC > 🧪 This simulates a production data lake or warehouse table that would feed into your AI pipeline.
# MAGIC

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Note: In Databricks, the 'spark' session is pre-initialized and ready to use
# No need to create a SparkSession manually

# Define schema and sample data
schema = StructType([
    StructField("claim_id", StringType(), True),
    StructField("claimant_name", StringType(), True),
    StructField("damage_description", StringType(), True),
    StructField("estimated_damage", DoubleType(), True),
    StructField("policy_id", StringType(), True)
])

data = [
    ("C101", "Alice Jones", "Rear-end collision, bumper damage", 2200.0, "P1001"),
    ("C102", "David Kim", "Broken windshield and headlight", 850.0, "P1002"),
    ("C103", "Maria Patel", "Side door dent and paint scratch", 1200.0, "P1003")
]

df = spark.createDataFrame(data, schema)
df.createOrReplaceTempView("claims")

display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🧠 Step 5: Initialize the LLM (Multiple Options)
# MAGIC
# MAGIC In this step, you'll initialize a **language model** using the modern, non-deprecated API. You have several options depending on your needs and budget.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔄 What Changed from Older Versions?
# MAGIC
# MAGIC **Old (Deprecated) Way:**
# MAGIC ```python
# MAGIC from langchain.llms import OpenAI  # ❌ Deprecated
# MAGIC llm = OpenAI(temperature=0)
# MAGIC ```
# MAGIC
# MAGIC **New (Current) Way:**
# MAGIC ```python
# MAGIC from langchain_openai import ChatOpenAI  # ✅ Modern
# MAGIC llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 Model Options
# MAGIC
# MAGIC Choose one of the following options based on your situation:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Option 1: OpenAI (Recommended for Production)**
# MAGIC
# MAGIC **Requirements**:
# MAGIC - Valid OpenAI API key
# MAGIC - Active billing account with available credits
# MAGIC
# MAGIC **Pros**:
# MAGIC - Best performance and reliability
# MAGIC - Excellent function calling support
# MAGIC - Industry standard
# MAGIC
# MAGIC **Cons**:
# MAGIC - Requires paid account
# MAGIC - Usage-based pricing
# MAGIC
# MAGIC ```python
# MAGIC from langchain_openai import ChatOpenAI
# MAGIC llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Option 2: Databricks Foundation Models (Free for Databricks Users)**
# MAGIC
# MAGIC **Requirements**:
# MAGIC - Databricks workspace with Foundation Model APIs enabled
# MAGIC
# MAGIC **Pros**:
# MAGIC - ✅ **FREE** for Databricks users
# MAGIC - No external API key needed
# MAGIC - Integrated with Databricks security
# MAGIC
# MAGIC **Cons**:
# MAGIC - Only available in Databricks environment
# MAGIC - May have different capabilities than OpenAI
# MAGIC
# MAGIC ```python
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC llm = ChatDatabricks(
# MAGIC     endpoint="databricks-meta-llama-3-3-70b-instruct",
# MAGIC     temperature=0
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Option 3: Azure OpenAI (Enterprise)**
# MAGIC
# MAGIC **Requirements**:
# MAGIC - Azure subscription
# MAGIC - Azure OpenAI resource deployed
# MAGIC
# MAGIC **Pros**:
# MAGIC - Enterprise-grade security and compliance
# MAGIC - Data residency control
# MAGIC - SLA guarantees
# MAGIC
# MAGIC **Cons**:
# MAGIC - Requires Azure setup
# MAGIC - More complex configuration
# MAGIC
# MAGIC ```python
# MAGIC from langchain_openai import AzureChatOpenAI
# MAGIC llm = AzureChatOpenAI(
# MAGIC     azure_endpoint="https://your-resource.openai.azure.com/",
# MAGIC     api_key="your-azure-key",
# MAGIC     api_version="2024-02-15-preview",
# MAGIC     deployment_name="gpt-35-turbo",
# MAGIC     temperature=0
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### **Option 4: Local Models with Ollama (Completely Free)**
# MAGIC
# MAGIC **Requirements**:
# MAGIC - Ollama installed locally or on cluster
# MAGIC - Sufficient compute resources
# MAGIC
# MAGIC **Pros**:
# MAGIC - ✅ **100% FREE**
# MAGIC - No API keys needed
# MAGIC - Complete data privacy
# MAGIC - No rate limits
# MAGIC
# MAGIC **Cons**:
# MAGIC - Requires local setup
# MAGIC - May have lower quality outputs
# MAGIC - Slower inference
# MAGIC
# MAGIC ```python
# MAGIC from langchain_community.chat_models import ChatOllama
# MAGIC llm = ChatOllama(
# MAGIC     model="llama3",
# MAGIC     temperature=0
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ⚙️ Configuration Parameters
# MAGIC
# MAGIC - **`model`**: Specifies which model to use
# MAGIC - **`temperature=0`**: Makes outputs deterministic (same input = same output), which is critical for production pipelines
# MAGIC
# MAGIC > 💡 **Best Practice**: Always use `temperature=0` for business logic and decision-making tasks to ensure consistency and reliability.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🚨 **Troubleshooting OpenAI Errors**
# MAGIC
# MAGIC If you see errors like:
# MAGIC - **`AuthenticationError (401)`**: Your API key is invalid or not set correctly
# MAGIC - **`RateLimitError (429)`**: You've exceeded your quota or don't have billing set up
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. Check your OpenAI billing at: https://platform.openai.com/account/billing
# MAGIC 2. Add credits to your account
# MAGIC 3. Or switch to **Option 2 (Databricks Foundation Models)** which is FREE
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 **Choose Your Model Below**
# MAGIC
# MAGIC > 📌 **IMPORTANT**: If you're using Databricks Foundation Models (default), make sure the endpoint name below matches what's available in YOUR workspace. See the **"Databricks Workspace Setup"** section at the top of this notebook for instructions.
# MAGIC

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

# OPTION 2: Databricks Foundation Models (FREE - RECOMMENDED) ⭐
# This is the default option - no API keys or billing required!
#
# ⚠️ CONFIGURATION: Update the endpoint name if needed
# - Go to your Databricks workspace: Serving → Serving Endpoints
# - Find an available Foundation Model endpoint
# - Replace the endpoint name below with YOUR endpoint name (just the name, NOT the full URL)
#
# ✅ CORRECT: endpoint="databricks-meta-llama-3-3-70b-instruct"
# ❌ WRONG:   endpoint="https://adb-123456.azuredatabricks.net/..."
#
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",  # ← Change this if needed
    temperature=0
)

# OPTION 1: OpenAI (requires valid API key and billing)
# Uncomment the lines below to use OpenAI instead:
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# OPTION 3: Azure OpenAI (for enterprise users)
# Uncomment and configure the lines below:
# from langchain_openai import AzureChatOpenAI
# llm = AzureChatOpenAI(
#     azure_endpoint="https://your-resource.openai.azure.com/",
#     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
#     api_version="2024-02-15-preview",
#     deployment_name="gpt-35-turbo",
#     temperature=0
# )

# OPTION 4: Local Ollama (completely free, requires Ollama installed)
# Uncomment the lines below:
# from langchain_community.chat_models import ChatOllama
# llm = ChatOllama(model="llama3", temperature=0)

print(f"✅ LLM initialized: {llm.__class__.__name__}")
print(f"✅ Using Databricks Foundation Model: databricks-meta-llama-3-3-70b-instruct")
print(f"✅ Cost: FREE (included with Databricks)")
print(f"✅ No API keys required!")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 📝 Step 6: Create Structured Prompts for Each Workflow Stage
# MAGIC
# MAGIC In this step, you'll define **prompt templates** for each of the four stages in your multi-agent workflow. Each prompt is designed to perform a specific task and produce structured, machine-readable output.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 The Four Stages
# MAGIC
# MAGIC 1. **Extraction Stage**: Extract structured fields from raw claim data
# MAGIC 2. **Policy Validation Stage**: Determine if a policy is active and eligible
# MAGIC 3. **Assessment Stage**: Decide if the claim should be auto-approved or manually reviewed
# MAGIC 4. **Resolution Stage**: Generate the final decision record
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔍 Why Structured Prompts Matter
# MAGIC
# MAGIC - **Consistency**: Each stage produces predictable output formats
# MAGIC - **Modularity**: Stages can be tested, debugged, and improved independently
# MAGIC - **Traceability**: You can audit each step of the decision-making process
# MAGIC - **Downstream Integration**: Structured outputs can be consumed by databases, APIs, or other systems
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📐 Prompt Design Principles
# MAGIC
# MAGIC Each prompt follows these best practices:
# MAGIC
# MAGIC - **Clear task definition**: Tells the LLM exactly what to do
# MAGIC - **Structured output format**: Specifies how the response should be formatted
# MAGIC - **Minimal ambiguity**: Uses precise language to reduce variability
# MAGIC - **Context-appropriate**: Tailored to the specific stage's responsibility
# MAGIC
# MAGIC > 🧩 This is a core principle of **prompt-task alignment**: matching the right prompt structure to the right type of task.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📦 Modern Import Structure
# MAGIC
# MAGIC **Important**: We're using `langchain_core.prompts` instead of the older `langchain.prompts`:
# MAGIC
# MAGIC ```python
# MAGIC from langchain_core.prompts import PromptTemplate  # ✅ Modern
# MAGIC ```
# MAGIC
# MAGIC This is part of LangChain's modular architecture where core abstractions live in `langchain-core`.
# MAGIC

# COMMAND ----------

from langchain_core.prompts import PromptTemplate

# Stage 1: Extraction Prompt
extraction_prompt = PromptTemplate.from_template(
    """You are an insurance claim data extraction specialist.

Extract the following structured information from this claim:

Claim Data: {claim_data}

Provide the output in this exact format:
- Claim ID: [value]
- Claimant Name: [value]
- Incident Type: [classify as: collision, vandalism, weather, or other]
- Damage Description: [value]
- Estimated Cost: [value]
- Policy ID: [value]
- Severity: [classify as: low, medium, or high based on cost]

Be precise and use only the information provided."""
)

# Stage 2: Policy Validation Prompt
validation_prompt = PromptTemplate.from_template(
    """You are a policy validation specialist.

Policy ID: {policy_id}
Policy Status: {policy_status}

Determine if this policy is eligible for claim processing.

Provide the output in this exact format:
- Policy ID: [value]
- Status: [Active/Inactive]
- Eligible for Coverage: [Yes/No]
- Reason: [brief explanation]"""
)

# Stage 3: Assessment Prompt
assessment_prompt = PromptTemplate.from_template(
    """You are a claim assessment specialist.

Claim Information:
{claim_info}

Policy Status:
{policy_status}

Based on the following rules:
1. If estimated cost > $2000, flag for manual review
2. If policy is not active, reject automatically
3. If cost <= $2000 and policy is active, auto-approve

Provide the output in this exact format:
- Decision: [Auto-Approve/Manual Review/Reject]
- Reason: [brief explanation]
- Estimated Cost: [value]
- Risk Level: [Low/Medium/High]"""
)

# Stage 4: Resolution Prompt
resolution_prompt = PromptTemplate.from_template(
    """You are a claim resolution specialist.

Assessment Result:
{assessment}

Generate a final decision record in this exact format:
- Final Decision: [Approved/Pending Review/Rejected]
- Next Steps: [specific actions required]
- Processing Status: [Complete/Requires Human Review]
- Timestamp: [current stage]"""
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🛠️ Step 7: Define Tool Functions for Multi-Agent Workflow
# MAGIC
# MAGIC This step defines four custom Python functions that act as **tools** for your LangChain agent. Each tool corresponds to one of the four workflow stages and encapsulates the business logic for that stage.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 Tool 1: Extract Claim Details (Extraction Stage)
# MAGIC
# MAGIC **Purpose**: Query the Spark table and retrieve raw claim data
# MAGIC
# MAGIC **Input**: `claim_id` (string)
# MAGIC
# MAGIC **Output**: Formatted string with claim details
# MAGIC
# MAGIC **Business Logic**:
# MAGIC - Queries the `claims` table using Spark SQL
# MAGIC - Returns structured claim information if found
# MAGIC - Returns error message if claim ID doesn't exist
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 Tool 2: Validate Policy (Policy Validation Stage)
# MAGIC
# MAGIC **Purpose**: Check if a policy is active and eligible for coverage
# MAGIC
# MAGIC **Input**: `policy_id` (string)
# MAGIC
# MAGIC **Output**: Policy validation status
# MAGIC
# MAGIC **Business Logic**:
# MAGIC - Simulates a policy database lookup
# MAGIC - Checks against a list of valid policy IDs
# MAGIC - Returns validation result with eligibility status
# MAGIC
# MAGIC > 📝 **Note**: In production, this would query a real policy management system or database.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 Tool 3: Assess Damage with LLM (Assessment Stage)
# MAGIC
# MAGIC **Purpose**: Use the LLM to evaluate claim data and make an approval decision
# MAGIC
# MAGIC **Input**: Claim information text
# MAGIC
# MAGIC **Output**: Assessment decision (Auto-Approve/Manual Review/Reject)
# MAGIC
# MAGIC **Business Logic**:
# MAGIC - Extracts the dollar amount from the claim text
# MAGIC - Invokes the LLM with the assessment prompt
# MAGIC - Returns structured decision based on business rules
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 Tool 4: Finalize Resolution (Resolution Stage)
# MAGIC
# MAGIC **Purpose**: Generate the final decision record
# MAGIC
# MAGIC **Input**: Assessment result
# MAGIC
# MAGIC **Output**: Final formatted decision message
# MAGIC
# MAGIC **Business Logic**:
# MAGIC - Takes the assessment output
# MAGIC - Formats it into a final resolution message
# MAGIC - Prepares the output for downstream systems
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > 🧩 These functions will be wrapped as LangChain tools in the next step, allowing a reasoning agent to call them dynamically based on the task prompt.
# MAGIC

# COMMAND ----------

import re

# Tool 1: Extract Claim Details from Spark Table
def extract_claim_details(claim_id):
    """
    Extraction Stage: Retrieve raw claim data from the Spark table.

    Args:
        claim_id: Unique identifier for the claim

    Returns:
        Formatted string with claim details or error message
    """
    # Use DataFrame API with column-based filtering (Databricks best practice)
    # This avoids SQL injection and is more efficient
    from pyspark.sql import functions as F

    claim_df = spark.table("claims").filter(F.col("claim_id") == claim_id)
    claim = claim_df.first()

    if not claim:
        return f"No claim found for ID {claim_id}"

    # Return structured claim data
    return f"Claimant: {claim.claimant_name}, Damage: {claim.damage_description}, Estimate: ${claim.estimated_damage}, Policy#: {claim.policy_id}"


# Tool 2: Validate Policy Status
def validate_policy(policy_id):
    """
    Policy Validation Stage: Check if a policy is active and eligible.

    Args:
        policy_id: Policy identifier to validate

    Returns:
        Policy validation status message
    """
    # Simulate valid policies (in production, this would query a policy database)
    valid_policies = {"P1001", "P1002", "P1003"}

    if policy_id in valid_policies:
        return f"Policy {policy_id} is valid and active."
    else:
        return f"Policy {policy_id} is NOT valid or inactive."


# Tool 3: Assess Damage Using LLM
def assess_damage_llm(text):
    """
    Assessment Stage: Use LLM to evaluate claim and determine approval decision.

    Args:
        text: Claim information text containing cost estimate

    Returns:
        LLM-generated assessment decision
    """
    # Extract dollar amount from text
    amount_match = re.search(r"\$?(\d+[.,]?\d*)", text)
    if not amount_match:
        return "Could not extract a valid dollar amount from the claim."

    amount = amount_match.group(1)

    # Invoke LLM with assessment prompt
    assessment_result = llm.invoke(
        assessment_prompt.format(
            claim_info=text,
            policy_status="Active"  # This would come from validate_policy in real workflow
        )
    )

    return assessment_result.content


# Tool 4: Finalize Resolution
def finalize_resolution(assessment):
    """
    Resolution Stage: Generate final decision record.

    Args:
        assessment: Assessment result from previous stage

    Returns:
        Final formatted decision message
    """
    return f"Final decision: {assessment}"


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🤖 Step 8: Register Tools and Initialize a Reasoning Agent
# MAGIC
# MAGIC Now that your helper functions are ready, you'll convert them into **LangChain-compatible tools** and initialize a **reasoning agent** using the modern OpenAI API.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔧 What Are LangChain Tools?
# MAGIC
# MAGIC Tools are functions that an agent can call to perform specific tasks. Each tool has:
# MAGIC
# MAGIC - **Name**: A unique identifier the agent uses to reference the tool
# MAGIC - **Function**: The Python function to execute
# MAGIC - **Description**: Instructions that tell the agent when and how to use the tool
# MAGIC
# MAGIC The agent reads these descriptions and decides which tools to call based on the user's request.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🧠 Agent Type: `create_agent`
# MAGIC
# MAGIC We're using the **modern `create_agent` API** from LangChain v1, which is the current recommended way to build agents.
# MAGIC
# MAGIC **How the Agent Works:**
# MAGIC
# MAGIC 1. **Reason**: The agent analyzes the task and decides what to do
# MAGIC 2. **Act**: The agent calls a tool to perform an action
# MAGIC 3. **Observe**: The agent examines the tool's output
# MAGIC 4. **Repeat**: The agent continues reasoning and acting until the task is complete
# MAGIC
# MAGIC This uses the ReAct (Reasoning + Acting) pattern under the hood.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔄 Modern vs. Deprecated Approach
# MAGIC
# MAGIC **Old (Deprecated) Way:**
# MAGIC ```python
# MAGIC from langchain.agents import initialize_agent, AgentType
# MAGIC agent = initialize_agent(
# MAGIC     tools=tools,
# MAGIC     llm=llm,
# MAGIC     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# MAGIC     verbose=True
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **New (Current) Way:**
# MAGIC ```python
# MAGIC from langchain.agents import create_agent
# MAGIC
# MAGIC agent = create_agent(
# MAGIC     model=llm,
# MAGIC     tools=tools,
# MAGIC     system_prompt="You are a helpful insurance claim processing assistant."
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 Why This Matters
# MAGIC
# MAGIC - **Non-deprecated**: Uses the current LangChain API
# MAGIC - **More flexible**: Allows custom prompts and better control
# MAGIC - **Production-ready**: Follows modern best practices
# MAGIC
# MAGIC > 💡 The agent will automatically orchestrate the four-stage workflow by calling the tools in the correct order.
# MAGIC

# COMMAND ----------

from langchain.agents import create_agent
from langchain_core.tools import tool

# Define tools using the @tool decorator (modern approach)
@tool
def extract_claim(claim_id: str) -> str:
    """Look up claim details from the Spark database. Input should be a claim ID like 'C101'."""
    return extract_claim_details(claim_id)

@tool
def validate_policy_tool(policy_id: str) -> str:
    """Check if a policy is valid and active. Input should be a policy ID like 'P1001'."""
    return validate_policy(policy_id)

@tool
def assess_damage(claim_info: str) -> str:
    """Assess claim damage and determine approval decision. Input should be claim information text with cost estimate."""
    return assess_damage_llm(claim_info)

@tool
def finalize_resolution_tool(assessment: str) -> str:
    """Generate the final claim decision. Input should be the assessment result."""
    return finalize_resolution(assessment)

# Create list of tools
tools = [extract_claim, validate_policy_tool, assess_damage, finalize_resolution_tool]

# Create the agent using the modern create_agent API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a helpful insurance claim processing assistant.

Your job is to process insurance claims through a multi-stage workflow:
1. Extract claim details using the claim ID
2. Validate the policy is active
3. Assess the damage and determine if it should be auto-approved or manually reviewed
4. Finalize the resolution with a decision

Always follow these steps in order and provide clear reasoning for your decisions."""
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚀 Step 9: Execute the Multi-Agent Workflow
# MAGIC
# MAGIC Now it's time to run the complete workflow! The agent will process a claim by automatically orchestrating all four stages.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🔄 What Happens When You Run This?
# MAGIC
# MAGIC When you execute `agent.invoke()`, the agent will:
# MAGIC
# MAGIC 1. **Read your prompt**: "Process claim C101"
# MAGIC 2. **Plan the workflow**: Determine which tools to call and in what order
# MAGIC 3. **Execute Stage 1 (Extraction)**: Call `extract_claim` to retrieve claim data
# MAGIC 4. **Execute Stage 2 (Validation)**: Call `validate_policy_tool` to check policy status
# MAGIC 5. **Execute Stage 3 (Assessment)**: Call `assess_damage` to evaluate the claim
# MAGIC 6. **Execute Stage 4 (Resolution)**: Call `finalize_resolution_tool` to generate the final decision
# MAGIC 7. **Return the result**: Provide a complete, structured response
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🧠 Observing the Reasoning Process
# MAGIC
# MAGIC The agent will show you its internal reasoning process in the output:
# MAGIC
# MAGIC - **Tool Calls**: Which tools the agent decides to use
# MAGIC - **Tool Inputs**: What data it passes to each tool
# MAGIC - **Tool Outputs**: What each tool returns
# MAGIC - **Final Answer**: The complete result
# MAGIC
# MAGIC This transparency is crucial for:
# MAGIC - **Debugging**: Understanding why the agent made certain decisions
# MAGIC - **Auditing**: Tracking the decision-making process for compliance
# MAGIC - **Optimization**: Identifying bottlenecks or inefficiencies
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📊 Expected Output
# MAGIC
# MAGIC For claim C101 (Alice Jones, $2200 damage):
# MAGIC
# MAGIC - **Extraction**: Successfully retrieves claim details
# MAGIC - **Validation**: Confirms policy P1001 is active
# MAGIC - **Assessment**: Flags for manual review (cost > $2000)
# MAGIC - **Resolution**: Generates final decision record
# MAGIC
# MAGIC > 💡 Try running this with different claim IDs (C102, C103) to see how the workflow adapts to different scenarios!
# MAGIC

# COMMAND ----------

# Run the multi-agent workflow on claim C101
try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Process claim C101"}]})
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    # Extract the final message from the agent
    final_message = result["messages"][-1]
    print(final_message.content)
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative invocation method...")
    # Alternative method for some model providers
    result = agent.invoke({"messages": [("user", "Process claim C101")]})
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    final_message = result["messages"][-1]
    print(final_message.content)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Step 10: Test with Additional Claims
# MAGIC
# MAGIC Let's test the workflow with the other claims to see how it handles different scenarios.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📋 Test Scenarios
# MAGIC
# MAGIC **Claim C102** (David Kim):
# MAGIC - Estimated damage: $850
# MAGIC - Expected outcome: Auto-approve (cost < $2000)
# MAGIC
# MAGIC **Claim C103** (Maria Patel):
# MAGIC - Estimated damage: $1200
# MAGIC - Expected outcome: Auto-approve (cost < $2000)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 What to Observe
# MAGIC
# MAGIC Pay attention to how the agent:
# MAGIC
# MAGIC 1. **Adapts its reasoning** based on different cost amounts
# MAGIC 2. **Maintains consistency** in the workflow structure
# MAGIC 3. **Produces structured outputs** at each stage
# MAGIC 4. **Makes different decisions** based on business rules
# MAGIC
# MAGIC This demonstrates the power of **modular, multi-stage workflows** where each component has a clear responsibility.
# MAGIC

# COMMAND ----------

# Test with claim C102 (lower cost - should auto-approve)
print("\n" + "="*60)
print("TESTING CLAIM C102")
print("="*60)
try:
    result_c102 = agent.invoke({"messages": [{"role": "user", "content": "Process claim C102"}]})
except:
    result_c102 = agent.invoke({"messages": [("user", "Process claim C102")]})
print("\n" + "="*60)
print("FINAL RESULT FOR C102:")
print("="*60)
print(result_c102["messages"][-1].content)


# COMMAND ----------

# Test with claim C103 (medium cost - should auto-approve)
print("\n" + "="*60)
print("TESTING CLAIM C103")
print("="*60)
try:
    result_c103 = agent.invoke({"messages": [{"role": "user", "content": "Process claim C103"}]})
except:
    result_c103 = agent.invoke({"messages": [("user", "Process claim C103")]})
print("\n" + "="*60)
print("FINAL RESULT FOR C103:")
print("="*60)
print(result_c103["messages"][-1].content)


# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Understanding the Multi-Agent Workflow Output
# MAGIC
# MAGIC Congratulations! You've successfully built and executed a **multi-stage generative AI workflow** for insurance claim processing.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🎯 Key Takeaways
# MAGIC
# MAGIC **1. Prompt-Task Alignment**
# MAGIC - Each stage had a specific prompt designed for its task (extraction, validation, assessment, resolution)
# MAGIC - This ensures the LLM performs the correct type of reasoning at each step
# MAGIC
# MAGIC **2. Structured Outputs**
# MAGIC - Every stage produced machine-readable, consistent outputs
# MAGIC - These outputs can be validated, logged, and consumed by downstream systems
# MAGIC
# MAGIC **3. Modular Design**
# MAGIC - Each tool is independent and can be tested separately
# MAGIC - Changes to one stage don't break the entire workflow
# MAGIC - Easy to add new stages or modify existing ones
# MAGIC
# MAGIC **4. Tool Ordering**
# MAGIC - The agent automatically determined the correct sequence of operations
# MAGIC - Data flows logically from extraction → validation → assessment → resolution
# MAGIC
# MAGIC **5. Observability**
# MAGIC - The verbose output shows every step of the reasoning process
# MAGIC - This is essential for debugging, auditing, and compliance
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🏢 Real-World Applications
# MAGIC
# MAGIC This pattern applies to many enterprise scenarios:
# MAGIC
# MAGIC - **Financial Services**: Loan application processing, fraud detection
# MAGIC - **Healthcare**: Medical claim adjudication, patient triage
# MAGIC - **Customer Service**: Ticket routing, automated responses
# MAGIC - **Legal**: Contract analysis, compliance checking
# MAGIC - **HR**: Resume screening, candidate evaluation
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🚀 Next Steps
# MAGIC
# MAGIC To extend this lab, you could:
# MAGIC
# MAGIC 1. **Add more validation rules** (e.g., check claim history, verify claimant identity)
# MAGIC 2. **Implement structured output parsing** using Pydantic models
# MAGIC 3. **Add error handling** for edge cases and invalid inputs
# MAGIC 4. **Integrate with real databases** instead of simulated data
# MAGIC 5. **Add logging and monitoring** for production deployment
# MAGIC 6. **Implement human-in-the-loop** for manual review cases
# MAGIC 7. **Create unit tests** for each tool function
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 📚 What You've Learned
# MAGIC
# MAGIC By completing this lab, you've demonstrated:
# MAGIC
# MAGIC ✅ How to design multi-stage AI workflows with clear task boundaries
# MAGIC ✅ How to use modern LangChain APIs without deprecated code
# MAGIC ✅ How to create structured prompts for consistent outputs
# MAGIC ✅ How to orchestrate multiple tools using a reasoning agent
# MAGIC ✅ How to execute AI workflows in Databricks with PySpark integration
# MAGIC ✅ How modular design improves reliability and maintainability
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > 🎓 **Certification Tip**: The concepts in this lab—prompt-task alignment, structured outputs, tool composition, and modular workflows—are core topics in the Databricks Generative AI Engineer Associate certification exam.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎓 Summary: Architecture Principles Demonstrated
# MAGIC
# MAGIC This lab reinforced the following architectural principles from Chapter 2:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 1️⃣ **Separation of Concerns**
# MAGIC Each stage has a single, well-defined responsibility:
# MAGIC - Extraction: Parse raw data
# MAGIC - Validation: Check eligibility
# MAGIC - Assessment: Make decisions
# MAGIC - Resolution: Format outputs
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 2️⃣ **Composability**
# MAGIC Tools can be combined in different ways:
# MAGIC - Add new tools without changing existing ones
# MAGIC - Reorder stages for different workflows
# MAGIC - Reuse tools across multiple agents
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 3️⃣ **Testability**
# MAGIC Each component can be tested independently:
# MAGIC - Unit test individual tool functions
# MAGIC - Integration test the full workflow
# MAGIC - Mock external dependencies (databases, APIs)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 4️⃣ **Observability**
# MAGIC The workflow provides full visibility:
# MAGIC - Verbose logging shows reasoning steps
# MAGIC - Structured outputs enable monitoring
# MAGIC - Clear error messages aid debugging
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 5️⃣ **Scalability**
# MAGIC The design supports production deployment:
# MAGIC - Stateless tools can run in parallel
# MAGIC - Modular architecture enables horizontal scaling
# MAGIC - Clear interfaces support microservices architecture
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > 🏆 **Well done!** You've completed the hands-on lab for Chapter 2: Multi-Agent Workflow with LangChain + OpenAI in Databricks.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔧 Troubleshooting Common Issues
# MAGIC
# MAGIC If you encounter errors while running this notebook, here are solutions to common problems:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `Endpoint not found` or `Model endpoint does not exist`**
# MAGIC
# MAGIC **Problem**: The Databricks model endpoint name doesn't match what's available in your workspace
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Check available endpoints in your workspace**:
# MAGIC    - Navigate to: **Serving** → **Serving Endpoints** in Databricks
# MAGIC    - Look for Foundation Model endpoints (e.g., `databricks-meta-llama-3-3-70b-instruct`)
# MAGIC    - Copy the exact endpoint name
# MAGIC
# MAGIC 2. **Update the endpoint in Step 5**:
# MAGIC    ```python
# MAGIC    llm = ChatDatabricks(
# MAGIC        endpoint="YOUR-ENDPOINT-NAME-HERE",  # ← Paste your endpoint name
# MAGIC        temperature=0
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC 3. **Common endpoint names to try**:
# MAGIC    - `databricks-meta-llama-3-3-70b-instruct` (recommended)
# MAGIC    - `databricks-meta-llama-3-1-70b-instruct`
# MAGIC    - `databricks-dbrx-instruct`
# MAGIC    - `databricks-mixtral-8x7b-instruct`
# MAGIC
# MAGIC 4. **Make sure you're using the endpoint NAME, not the URL**:
# MAGIC    - ✅ CORRECT: `endpoint="databricks-meta-llama-3-3-70b-instruct"`
# MAGIC    - ❌ WRONG: `endpoint="https://adb-123456.azuredatabricks.net/..."`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `AuthenticationError: Error code: 401`**
# MAGIC
# MAGIC **Problem**: Invalid or missing OpenAI API key
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Check your API key is set correctly**:
# MAGIC    ```python
# MAGIC    import os
# MAGIC    print(os.environ.get("OPENAI_API_KEY", "NOT SET")[:10] + "...")
# MAGIC    ```
# MAGIC    Should show: `sk-proj-...` or `sk-...`
# MAGIC
# MAGIC 2. **Verify your key is valid** at: https://platform.openai.com/account/api-keys
# MAGIC
# MAGIC 3. **Switch to Databricks Foundation Models (FREE)**:
# MAGIC    ```python
# MAGIC    from langchain_community.chat_models import ChatDatabricks
# MAGIC    llm = ChatDatabricks(
# MAGIC        endpoint="databricks-meta-llama-3-3-70b-instruct",
# MAGIC        temperature=0
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `RateLimitError: Error code: 429`**
# MAGIC
# MAGIC **Problem**: OpenAI quota exceeded or no billing set up
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Check your OpenAI billing**: https://platform.openai.com/account/billing
# MAGIC    - Add credits to your account
# MAGIC    - Verify you have an active payment method
# MAGIC
# MAGIC 2. **Use Databricks Foundation Models instead (FREE)**:
# MAGIC    ```python
# MAGIC    from langchain_community.chat_models import ChatDatabricks
# MAGIC    llm = ChatDatabricks(
# MAGIC        endpoint="databricks-meta-llama-3-3-70b-instruct",
# MAGIC        temperature=0
# MAGIC    )
# MAGIC    ```
# MAGIC    Then re-run the agent creation cell and execution cells.
# MAGIC
# MAGIC 3. **Use a local model with Ollama (FREE)**:
# MAGIC    ```bash
# MAGIC    # Install Ollama first: https://ollama.ai
# MAGIC    ollama pull llama3
# MAGIC    ```
# MAGIC    ```python
# MAGIC    from langchain_community.chat_models import ChatOllama
# MAGIC    llm = ChatOllama(model="llama3", temperature=0)
# MAGIC    ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `ModuleNotFoundError: No module named 'langchain_openai'`**
# MAGIC
# MAGIC **Problem**: Packages not installed or kernel not restarted
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Re-run the installation cell**:
# MAGIC    ```python
# MAGIC    %pip install --upgrade langchain-openai langchain langchain-community langchain-core langgraph openai
# MAGIC    ```
# MAGIC
# MAGIC 2. **Restart the Python kernel**:
# MAGIC    ```python
# MAGIC    %restart_python
# MAGIC    ```
# MAGIC
# MAGIC 3. **Verify installation**:
# MAGIC    ```python
# MAGIC    import langchain
# MAGIC    import langchain_openai
# MAGIC    print(f"LangChain version: {langchain.__version__}")
# MAGIC    ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `ImportError: cannot import name 'create_agent'`**
# MAGIC
# MAGIC **Problem**: Old version of LangChain installed
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Upgrade to LangChain 1.0+**:
# MAGIC    ```python
# MAGIC    %pip install --upgrade langchain>=1.0.0
# MAGIC    %restart_python
# MAGIC    ```
# MAGIC
# MAGIC 2. **Verify version**:
# MAGIC    ```python
# MAGIC    import langchain
# MAGIC    print(langchain.__version__)  # Should be 1.0.0 or higher
# MAGIC    ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `AnalysisException: Table or view not found: claims`**
# MAGIC
# MAGIC **Problem**: The claims table wasn't created
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Re-run the data loading cell** (Step 4):
# MAGIC    ```python
# MAGIC    df = spark.createDataFrame(data, schema)
# MAGIC    df.createOrReplaceTempView("claims")
# MAGIC    ```
# MAGIC
# MAGIC 2. **Verify the table exists**:
# MAGIC    ```python
# MAGIC    spark.sql("SELECT * FROM claims").show()
# MAGIC    ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: Agent produces incorrect or incomplete results**
# MAGIC
# MAGIC **Problem**: Model quality or prompt issues
# MAGIC
# MAGIC **Solutions**:
# MAGIC 1. **Try a more capable model**:
# MAGIC    ```python
# MAGIC    llm = ChatOpenAI(model="gpt-4", temperature=0)  # More expensive but better
# MAGIC    ```
# MAGIC
# MAGIC 2. **Check tool descriptions are clear**:
# MAGIC    - Each `@tool` function should have a clear docstring
# MAGIC    - The system prompt should be specific
# MAGIC
# MAGIC 3. **Add more examples to prompts**:
# MAGIC    - Include few-shot examples in your prompt templates
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### ❌ **Error: `dbutils is not defined` or Running Outside Databricks**
# MAGIC
# MAGIC **Problem**: You're trying to run this notebook outside of a Databricks environment
# MAGIC
# MAGIC **Solutions**:
# MAGIC
# MAGIC **Option 1: Use OpenAI instead (requires API key)**
# MAGIC ```python
# MAGIC import os
# MAGIC from langchain_openai import ChatOpenAI
# MAGIC
# MAGIC # Set your API key
# MAGIC os.environ["OPENAI_API_KEY"] = "sk-..."  # Your OpenAI API key
# MAGIC
# MAGIC # Use OpenAI model
# MAGIC llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# MAGIC ```
# MAGIC
# MAGIC **Option 2: Use local Ollama (completely free)**
# MAGIC ```bash
# MAGIC # Install Ollama first: https://ollama.ai
# MAGIC ollama pull llama3
# MAGIC ```
# MAGIC ```python
# MAGIC from langchain_community.chat_models import ChatOllama
# MAGIC llm = ChatOllama(model="llama3", temperature=0)
# MAGIC ```
# MAGIC
# MAGIC **Option 3: Run in Databricks (recommended for this lab)**
# MAGIC - Upload this notebook to your Databricks workspace
# MAGIC - Databricks Community Edition is FREE: https://databricks.com/try-databricks
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 💡 **Best Practices for Success**
# MAGIC
# MAGIC 1. ✅ **Always restart the kernel** after installing packages
# MAGIC 2. ✅ **Run cells in order** from top to bottom
# MAGIC 3. ✅ **Verify your model endpoint** matches what's available in your workspace
# MAGIC 4. ✅ **Use endpoint NAMES, not URLs** in your code
# MAGIC 5. ✅ **Use Databricks Foundation Models** if you don't have OpenAI credits
# MAGIC 6. ✅ **Read error messages carefully** - they usually tell you exactly what's wrong
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 🆘 **Still Having Issues?**
# MAGIC
# MAGIC If you're still stuck:
# MAGIC 1. Check the [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
# MAGIC 2. Review the [Databricks Documentation](https://docs.databricks.com/)
# MAGIC 3. Search for your error message on [Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)
# MAGIC 4. Ask in the [LangChain Discord](https://discord.gg/langchain)
# MAGIC
# MAGIC
# MAGIC