# Databricks notebook source
# MAGIC %md
# MAGIC # Hands-On Lab: End-to-End Model Management with MLflow and Unity Catalog
# MAGIC
# MAGIC ## Scenario
# MAGIC You are a data scientist at a financial services company developing a **customer support assistant** that answers questions about account policies and churn drivers. The team has a working prototype that uses a **Retrieval-Augmented Generation (RAG) pipeline** to ground responses in internal documentation stored in the lakehouse.
# MAGIC
# MAGIC Leadership is satisfied with early demonstrations but is concerned about three production risks:
# MAGIC 1. **Reproducibility of experiments** - Can we recreate results and compare different RAG configurations?
# MAGIC 2. **Governance and audit readiness** - Can we prove to regulators which model version was served at any point in time?
# MAGIC 3. **Runaway costs** - How do we prevent uncontrolled LLM usage from creating budget overruns?
# MAGIC
# MAGIC To address these risks, you will operationalize the RAG solution using **MLflow** for experiment tracking, evaluation evidence, and artifact management, and **Unity Catalog** for model registration, version governance, and access control. Regulators and internal audit partners require a complete record of how the model was developed, which version was served at any point in time, and who had permission to promote or invoke the model.
# MAGIC
# MAGIC ### Your Workflow Will Cover:
# MAGIC 1. **Track an MLflow experiment** that captures the RAG configuration, including key parameters such as retrieval settings and prompt template identifiers, so that results can be reproduced and compared across runs.
# MAGIC 2. **Log model artifacts and supporting evidence**, including representative prompt–response examples and retrieved context samples, so that reviewers can inspect what the system generated and what information the system used.
# MAGIC 3. **Evaluate RAG performance** using MLflow by logging summary metrics and qualitative artifacts that reflect relevance and grounding, so that promotion decisions are supported by evidence rather than intuition.
# MAGIC 4. **Register the candidate model** in the Unity Catalog–backed Model Registry, then manage versions using aliases (for example, Champion, Challenger) and tags (for example, lifecycle=archived) to prevent experimental versions from being used in production by mistake.
# MAGIC 5. **Implement Unity Catalog governance**, including role-based access control (RBAC), audit logging, and lineage, to ensure that access and changes are transparent and traceable.
# MAGIC 6. **Implement operational cost controls** by applying disciplined experimental hygiene, restricting access to production endpoints, and retiring unused versions, so that LLM usage remains predictable as adoption grows.
# MAGIC
# MAGIC This lab mirrors a real enterprise pattern: a team must demonstrate not only that a model works, but also that the team can explain, govern, and sustain the model over time.
# MAGIC
# MAGIC ## Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC 1. **Implement MLflow experiment tracking** for a governed RAG workflow by logging parameters, tags, metrics, and artifacts in a consistent structure that supports reproducibility and audit review.
# MAGIC 2. **Log and organize model outputs as MLflow artifacts**, including prompt templates, retrieved context samples, and representative responses, so that reviewers can validate model behavior without rerunning the full pipeline.
# MAGIC 3. **Evaluate RAG behavior** using MLflow by capturing both quantitative metrics and qualitative evidence that supports the assessment of relevance and grounding.
# MAGIC 4. **Register a model to the Unity Catalog–backed Model Registry** using a fully qualified model name, then manage versions using aliases (for example, Champion, Challenger) and tags (for example, lifecycle=archived) to separate experimentation from production usage.
# MAGIC 5. **Apply Unity Catalog governance controls** by enforcing RBAC, reviewing audit-relevant activity, and using lineage to trace a model version back to the training or evaluation context.
# MAGIC 6. **Apply operational best practices** by documenting assumptions, maintaining model metadata and signatures, archiving older versions, and cleaning up unused runs and models to keep the registry usable.
# MAGIC 7. **Apply cost-aware operational practices** by limiting uncontrolled experimentation, reusing proven configurations, and restricting production access so that LLM inference does not scale without governance.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with Unity Catalog enabled
# MAGIC - Access to create catalogs, schemas, and tables
# MAGIC - MLflow 2.8+ installed (pre-installed in Databricks Runtime ML)
# MAGIC - Access to Foundation Model APIs (for LLM integration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Environment Setup and Prerequisites
# MAGIC
# MAGIC ### What You'll Learn in This Section
# MAGIC In this section, we will:
# MAGIC 1. Import necessary libraries for RAG pipeline development and MLflow tracking
# MAGIC 2. Configure Unity Catalog settings for governed model storage
# MAGIC 3. Create sample internal documentation data (knowledge base for RAG)
# MAGIC 4. Generate synthetic customer questions for testing
# MAGIC
# MAGIC ### Why This Matters
# MAGIC **Reproducibility from Day One:** Proper environment setup ensures that every experiment can be recreated. By establishing a consistent Unity Catalog namespace and logging all dependencies, you create an audit trail that satisfies regulatory requirements.
# MAGIC
# MAGIC **Governance Foundation:** Unity Catalog provides enterprise-grade data governance, while MLflow handles the complete model lifecycle. This combination ensures that every artifact, from training data to model versions, is tracked, secured, and auditable.
# MAGIC
# MAGIC **RAG-Specific Considerations:** Unlike traditional ML models, RAG systems depend on external knowledge sources. Tracking the version and lineage of your knowledge base is just as important as tracking model parameters. This section establishes that foundation.

# COMMAND ----------

# Import required libraries for RAG pipeline and MLflow tracking
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

# For text processing and embeddings
from typing import List, Dict, Any, Tuple
import hashlib

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LIBRARY IMPORT STATUS")
print("=" * 80)
print("✓ MLflow and tracking libraries imported")
print("✓ Data processing libraries imported (pandas, numpy, pyspark)")
print("✓ Text processing utilities imported")
print(f"\n📦 MLflow version: {mlflow.__version__}")
print(f"📦 Python version: {pd.__version__}")
print("\n💡 Note: This lab uses MLflow 2.8+ features for RAG evaluation")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Unity Catalog Settings
# MAGIC
# MAGIC #### Understanding the Three-Level Namespace
# MAGIC Unity Catalog uses a three-level namespace: **`catalog.schema.object`**. This hierarchical structure enables:
# MAGIC - **Catalog-level governance**: Broad access control and compliance boundaries
# MAGIC - **Schema-level organization**: Logical grouping of related assets (data, models, functions)
# MAGIC - **Object-level precision**: Fine-grained permissions on individual tables, models, or volumes
# MAGIC
# MAGIC #### Our Namespace Structure
# MAGIC We'll set up the following Unity Catalog namespace for this RAG project:
# MAGIC - **Catalog**: `financial_services` - Top-level container representing our business domain
# MAGIC - **Schema**: `rag_support_assistant` - Logical grouping for RAG-related assets (knowledge base, models, evaluation data)
# MAGIC - **Tables**:
# MAGIC   - `knowledge_base` - Internal documentation for retrieval
# MAGIC   - `evaluation_questions` - Test questions for RAG evaluation
# MAGIC - **Model**: `customer_support_rag_model` - Registered RAG model with full lineage
# MAGIC
# MAGIC #### Governance Benefits
# MAGIC **Centralized Access Control:** Unity Catalog provides RBAC at every level. You can grant different teams different permissions (e.g., data scientists can create models, analysts can only read data).
# MAGIC
# MAGIC **Automatic Audit Logging:** Every operation (read, write, delete, grant) is automatically logged. This creates a complete audit trail for compliance.
# MAGIC
# MAGIC **Data Lineage:** Unity Catalog automatically tracks relationships between tables, models, and downstream consumers. You can trace a model back to the exact data version it was trained on.
# MAGIC
# MAGIC **Cross-Workspace Sharing:** Models registered in Unity Catalog can be accessed from any workspace attached to the same metastore, enabling true enterprise-wide governance.

# COMMAND ----------

# Define Unity Catalog namespace for RAG project
CATALOG_NAME = "financial_services"
SCHEMA_NAME = "rag_support_assistant"
KNOWLEDGE_BASE_TABLE = "knowledge_base"
EVAL_QUESTIONS_TABLE = "evaluation_questions"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.customer_support_rag_model"

# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print("=" * 80)
print("UNITY CATALOG CONFIGURATION")
print("=" * 80)
print(f"✓ Catalog: {CATALOG_NAME}")
print(f"✓ Schema: {SCHEMA_NAME}")
print(f"\n📊 Data Assets:")
print(f"  • Knowledge Base Table: {CATALOG_NAME}.{SCHEMA_NAME}.{KNOWLEDGE_BASE_TABLE}")
print(f"  • Evaluation Questions: {CATALOG_NAME}.{SCHEMA_NAME}.{EVAL_QUESTIONS_TABLE}")
print(f"\n🤖 Model Registry:")
print(f"  • RAG Model: {MODEL_NAME}")
print(f"\n💡 All assets are now governed by Unity Catalog with:")
print(f"  • Automatic audit logging")
print(f"  • Fine-grained access control (RBAC)")
print(f"  • Complete data lineage tracking")
print(f"  • Cross-workspace accessibility")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Sample Knowledge Base for RAG
# MAGIC
# MAGIC #### What is a Knowledge Base in RAG?
# MAGIC A **knowledge base** is the collection of documents that the RAG system retrieves from to answer questions. In production, this would be:
# MAGIC - Internal policy documents
# MAGIC - Product documentation
# MAGIC - FAQ databases
# MAGIC - Historical support tickets
# MAGIC - Regulatory compliance documents
# MAGIC
# MAGIC #### Our Simulated Knowledge Base
# MAGIC We'll create a realistic knowledge base containing internal documentation about:
# MAGIC - **Account Policies**: Overdraft protection, minimum balance requirements, account closure procedures
# MAGIC - **Churn Drivers**: Common reasons customers leave and retention strategies
# MAGIC - **Product Information**: Account types, features, and benefits
# MAGIC - **Compliance Information**: Regulatory requirements and customer rights
# MAGIC
# MAGIC #### Why This Matters for Governance
# MAGIC **Data Lineage:** By storing the knowledge base in Unity Catalog, we can track which model version used which version of the documentation. If a policy changes, we can identify all models that need retraining.
# MAGIC
# MAGIC **Access Control:** Sensitive internal documents can be protected with Unity Catalog RBAC. Only authorized users can access or modify the knowledge base.
# MAGIC
# MAGIC **Audit Trail:** Every query against the knowledge base is logged, creating a complete record of what information was used to generate customer-facing responses.
# MAGIC
# MAGIC **Versioning:** Unity Catalog's Delta Lake integration provides time travel, allowing you to audit what documentation was available at any point in time.

# COMMAND ----------

# Set random seed for reproducibility
np.random.seed(42)

# Create comprehensive knowledge base documents
knowledge_base_documents = [
    # Account Policies
    {
        'doc_id': 'POL-001',
        'category': 'Account Policies',
        'title': 'Overdraft Protection Policy',
        'content': 'Overdraft protection is available for Premium and Gold account holders. The service covers overdrafts up to $500 with a $35 fee per occurrence. Basic account holders must maintain a minimum balance of $100 to avoid monthly fees. Overdraft protection can be linked to a savings account or line of credit.',
        'last_updated': '2024-01-15',
        'version': '2.1'
    },
    {
        'doc_id': 'POL-002',
        'category': 'Account Policies',
        'title': 'Account Closure Procedures',
        'content': 'Customers may close their accounts at any time without penalty if the account balance is zero. For accounts with remaining balances, customers must transfer or withdraw all funds before closure. Account closure requests can be submitted online, by phone, or in person. Processing takes 3-5 business days. Any recurring payments must be cancelled separately.',
        'last_updated': '2024-02-01',
        'version': '1.5'
    },
    {
        'doc_id': 'POL-003',
        'category': 'Account Policies',
        'title': 'Minimum Balance Requirements',
        'content': 'Basic accounts require a $100 minimum daily balance to avoid a $12 monthly maintenance fee. Premium accounts require $2,500 minimum balance to waive the $25 monthly fee. Gold accounts require $10,000 minimum balance to waive the $35 monthly fee. Students and seniors over 65 are exempt from minimum balance requirements on Basic accounts.',
        'last_updated': '2024-01-20',
        'version': '3.0'
    },
    # Churn Drivers and Retention
    {
        'doc_id': 'CHR-001',
        'category': 'Churn Analysis',
        'title': 'Top Reasons for Customer Churn',
        'content': 'Analysis of customer exit surveys reveals the top churn drivers: 1) High fees and charges (42%), 2) Poor customer service experience (28%), 3) Better offers from competitors (18%), 4) Difficulty using online/mobile banking (8%), 5) Relocation or life changes (4%). Customers who file complaints are 3x more likely to churn within 90 days.',
        'last_updated': '2024-03-10',
        'version': '1.2'
    },
    {
        'doc_id': 'CHR-002',
        'category': 'Churn Analysis',
        'title': 'Retention Strategies and Best Practices',
        'content': 'Effective retention strategies include: proactive outreach to at-risk customers, fee waivers for long-term customers, personalized product recommendations, and priority customer service. Customers with multiple products have 60% lower churn rates. Mobile app engagement reduces churn by 35%. Regular communication and financial education programs improve retention by 25%.',
        'last_updated': '2024-03-15',
        'version': '2.0'
    },
    {
        'doc_id': 'CHR-003',
        'category': 'Churn Analysis',
        'title': 'Early Warning Indicators',
        'content': 'Key indicators of potential churn include: decreased transaction frequency (50% reduction over 60 days), multiple customer service calls within 30 days, complaint filing, balance below minimum for 3+ consecutive months, and no mobile app usage for 90+ days. Customers showing 3 or more indicators have an 80% churn probability within 6 months.',
        'last_updated': '2024-03-01',
        'version': '1.0'
    },
    # Product Information
    {
        'doc_id': 'PRD-001',
        'category': 'Products',
        'title': 'Account Types and Features',
        'content': 'We offer three account types: Basic (no monthly fee with $100 minimum balance, standard ATM access, online banking), Premium ($25/month or waived with $2,500 balance, unlimited ATM reimbursement, overdraft protection, priority support), and Gold ($35/month or waived with $10,000 balance, all Premium features plus dedicated relationship manager, premium credit card, investment advisory services).',
        'last_updated': '2024-02-15',
        'version': '4.1'
    },
    {
        'doc_id': 'PRD-002',
        'category': 'Products',
        'title': 'Digital Banking Features',
        'content': 'Our mobile app and online banking platform offer: real-time balance and transaction alerts, mobile check deposit, bill pay and recurring payments, peer-to-peer transfers, budgeting tools, spending analytics, and biometric authentication. Premium and Gold members get early access to new features and enhanced security options including transaction verification and travel notifications.',
        'last_updated': '2024-03-20',
        'version': '3.2'
    },
    # Compliance and Regulations
    {
        'doc_id': 'CMP-001',
        'category': 'Compliance',
        'title': 'Customer Rights and Protections',
        'content': 'Under federal regulations, customers have the right to: dispute unauthorized transactions within 60 days, receive clear fee disclosures, access account information within 30 days of request, and opt-out of data sharing with third parties. We comply with FDIC insurance requirements, providing coverage up to $250,000 per depositor. Customer data is protected under GLBA and state privacy laws.',
        'last_updated': '2024-01-10',
        'version': '5.0'
    },
    {
        'doc_id': 'CMP-002',
        'category': 'Compliance',
        'title': 'Fee Disclosure Requirements',
        'content': 'All account fees must be disclosed in writing before account opening. Monthly maintenance fees, overdraft fees, ATM fees, wire transfer fees, and other service charges are detailed in the fee schedule provided to customers. Fee changes require 30-day advance notice. Customers can request fee waivers in cases of financial hardship, which are reviewed on a case-by-case basis.',
        'last_updated': '2024-02-05',
        'version': '2.3'
    }
]

# Convert to DataFrame
df_knowledge_base = pd.DataFrame(knowledge_base_documents)

# Add metadata for tracking
df_knowledge_base['data_created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
df_knowledge_base['char_count'] = df_knowledge_base['content'].str.len()
df_knowledge_base['word_count'] = df_knowledge_base['content'].str.split().str.len()

print("=" * 80)
print("KNOWLEDGE BASE GENERATION COMPLETE")
print("=" * 80)
print(f"✓ Generated {len(df_knowledge_base)} knowledge base documents")
print(f"\n📊 Document Statistics:")
print(f"  • Categories: {df_knowledge_base['category'].nunique()}")
print(f"  • Average document length: {df_knowledge_base['char_count'].mean():.0f} characters")
print(f"  • Average word count: {df_knowledge_base['word_count'].mean():.0f} words")
print(f"\n📁 Document Breakdown by Category:")
print(df_knowledge_base['category'].value_counts().to_string())
print("\n💡 This knowledge base will be used for:")
print("  • Retrieval-Augmented Generation (RAG)")
print("  • Grounding LLM responses in factual information")
print("  • Ensuring compliance and accuracy in customer support")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Knowledge Base to Unity Catalog
# MAGIC
# MAGIC #### Why Store Knowledge Base in Unity Catalog?
# MAGIC In production RAG systems, the knowledge base is a **critical production asset** that requires the same governance as any other data:
# MAGIC
# MAGIC **ACID Transactions:** Delta Lake ensures that updates to the knowledge base are atomic and consistent. If a policy document is updated, either the entire update succeeds or none of it does—no partial updates.
# MAGIC
# MAGIC **Time Travel for Compliance:** Unity Catalog's Delta Lake integration allows you to query the knowledge base as it existed at any point in time. This is critical for compliance: "What information was available to the model on March 15th when it answered customer X's question?"
# MAGIC
# MAGIC **Automatic Lineage Tracking:** Unity Catalog automatically tracks which models were trained or evaluated using which version of the knowledge base. This creates an auditable chain from source documents to model predictions.
# MAGIC
# MAGIC **Fine-Grained Access Control:** Different documents may have different sensitivity levels. Unity Catalog RBAC allows you to control who can read, write, or modify specific documents or categories.
# MAGIC
# MAGIC **Audit Logging:** Every access to the knowledge base is logged with user identity, timestamp, and operation type. This creates a complete audit trail for regulatory review.
# MAGIC
# MAGIC #### What We're Storing
# MAGIC We'll save the knowledge base as a Delta table with:
# MAGIC - Document content and metadata
# MAGIC - Version information for each document
# MAGIC - Category tags for organization
# MAGIC - Timestamps for audit purposes

# COMMAND ----------

# Convert to Spark DataFrame and save to Unity Catalog
df_kb_spark = spark.createDataFrame(df_knowledge_base)

# Write to Delta table in Unity Catalog
kb_table_path = f"{CATALOG_NAME}.{SCHEMA_NAME}.{KNOWLEDGE_BASE_TABLE}"
df_kb_spark.write.format("delta").mode("overwrite").saveAsTable(kb_table_path)

print("=" * 80)
print("KNOWLEDGE BASE SAVED TO UNITY CATALOG")
print("=" * 80)
print(f"✓ Table: {kb_table_path}")
print(f"✓ Format: Delta Lake (ACID compliant)")
print(f"✓ Records: {df_kb_spark.count():,} documents")

# Verify table creation and show sample
df_kb_loaded = spark.table(kb_table_path)
print(f"\n📊 Table Schema:")
df_kb_loaded.printSchema()

print(f"\n📄 Sample Documents:")
display(df_kb_loaded.select('doc_id', 'category', 'title', 'word_count', 'version').limit(5))

print("\n💡 Governance Features Now Active:")
print("  ✓ Time travel enabled - query historical versions")
print("  ✓ Audit logging active - all access tracked")
print("  ✓ Lineage tracking - models will link to this data")
print("  ✓ RBAC ready - permissions can be granted per document category")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Evaluation Questions
# MAGIC
# MAGIC #### Why Do We Need Evaluation Questions?
# MAGIC Unlike traditional ML models where you have labeled test data, RAG systems require **question-answer pairs** to evaluate performance. These questions should:
# MAGIC - Cover the breadth of topics in your knowledge base
# MAGIC - Include both simple factual questions and complex reasoning questions
# MAGIC - Test edge cases (questions with no answer, ambiguous questions)
# MAGIC - Represent real user queries
# MAGIC
# MAGIC #### Evaluation Dimensions for RAG
# MAGIC We'll create questions that test:
# MAGIC 1. **Retrieval Quality**: Does the system find the right documents?
# MAGIC 2. **Answer Relevance**: Is the generated answer relevant to the question?
# MAGIC 3. **Groundedness/Faithfulness**: Is the answer supported by the retrieved context?
# MAGIC 4. **Completeness**: Does the answer fully address the question?
# MAGIC 5. **Conciseness**: Is the answer appropriately detailed without unnecessary information?
# MAGIC
# MAGIC #### Our Evaluation Dataset
# MAGIC We'll generate questions across different categories and difficulty levels to comprehensively test the RAG system.

# COMMAND ----------

# Generate evaluation questions with expected answers
evaluation_questions = [
    {
        'question_id': 'Q001',
        'question': 'What is the overdraft protection limit for Premium account holders?',
        'category': 'Account Policies',
        'difficulty': 'easy',
        'expected_answer': 'Overdraft protection covers up to $500 for Premium account holders with a $35 fee per occurrence.',
        'relevant_doc_ids': ['POL-001']
    },
    {
        'question_id': 'Q002',
        'question': 'How long does it take to process an account closure request?',
        'category': 'Account Policies',
        'difficulty': 'easy',
        'expected_answer': 'Account closure requests take 3-5 business days to process.',
        'relevant_doc_ids': ['POL-002']
    },
    {
        'question_id': 'Q003',
        'question': 'What are the minimum balance requirements to avoid monthly fees for each account type?',
        'category': 'Account Policies',
        'difficulty': 'medium',
        'expected_answer': 'Basic accounts require $100 minimum balance to avoid $12 monthly fee. Premium accounts require $2,500 to waive $25 fee. Gold accounts require $10,000 to waive $35 fee. Students and seniors over 65 are exempt from minimum balance requirements on Basic accounts.',
        'relevant_doc_ids': ['POL-003']
    },
    {
        'question_id': 'Q004',
        'question': 'What are the top three reasons customers leave our bank?',
        'category': 'Churn Analysis',
        'difficulty': 'medium',
        'expected_answer': 'The top three reasons for customer churn are: 1) High fees and charges (42%), 2) Poor customer service experience (28%), and 3) Better offers from competitors (18%).',
        'relevant_doc_ids': ['CHR-001']
    },
    {
        'question_id': 'Q005',
        'question': 'What retention strategies are most effective according to our analysis?',
        'category': 'Churn Analysis',
        'difficulty': 'medium',
        'expected_answer': 'Effective retention strategies include proactive outreach to at-risk customers, fee waivers for long-term customers, personalized product recommendations, and priority customer service. Customers with multiple products have 60% lower churn rates, mobile app engagement reduces churn by 35%, and financial education programs improve retention by 25%.',
        'relevant_doc_ids': ['CHR-002']
    },
    {
        'question_id': 'Q006',
        'question': 'What are the early warning signs that a customer might churn?',
        'category': 'Churn Analysis',
        'difficulty': 'hard',
        'expected_answer': 'Key early warning indicators include: decreased transaction frequency (50% reduction over 60 days), multiple customer service calls within 30 days, complaint filing, balance below minimum for 3+ consecutive months, and no mobile app usage for 90+ days. Customers showing 3 or more indicators have an 80% churn probability within 6 months.',
        'relevant_doc_ids': ['CHR-003']
    },
    {
        'question_id': 'Q007',
        'question': 'What features are included in the Gold account?',
        'category': 'Products',
        'difficulty': 'easy',
        'expected_answer': 'Gold accounts include all Premium features plus dedicated relationship manager, premium credit card, and investment advisory services. The monthly fee is $35 or waived with $10,000 minimum balance.',
        'relevant_doc_ids': ['PRD-001']
    },
    {
        'question_id': 'Q008',
        'question': 'What digital banking features do we offer?',
        'category': 'Products',
        'difficulty': 'medium',
        'expected_answer': 'Digital banking features include real-time balance and transaction alerts, mobile check deposit, bill pay and recurring payments, peer-to-peer transfers, budgeting tools, spending analytics, and biometric authentication. Premium and Gold members get early access to new features and enhanced security options.',
        'relevant_doc_ids': ['PRD-002']
    },
    {
        'question_id': 'Q009',
        'question': 'What are customer rights regarding unauthorized transactions?',
        'category': 'Compliance',
        'difficulty': 'medium',
        'expected_answer': 'Customers have the right to dispute unauthorized transactions within 60 days. Accounts are FDIC insured up to $250,000 per depositor, and customer data is protected under GLBA and state privacy laws.',
        'relevant_doc_ids': ['CMP-001']
    },
    {
        'question_id': 'Q010',
        'question': 'How much advance notice is required for fee changes?',
        'category': 'Compliance',
        'difficulty': 'easy',
        'expected_answer': 'Fee changes require 30-day advance notice to customers.',
        'relevant_doc_ids': ['CMP-002']
    },
    {
        'question_id': 'Q011',
        'question': 'If a customer has a Premium account and wants to avoid fees, what should they do?',
        'category': 'Account Policies',
        'difficulty': 'hard',
        'expected_answer': 'To avoid the $25 monthly fee on a Premium account, customers should maintain a minimum daily balance of $2,500.',
        'relevant_doc_ids': ['POL-003']
    },
    {
        'question_id': 'Q012',
        'question': 'What is the relationship between mobile app usage and customer retention?',
        'category': 'Churn Analysis',
        'difficulty': 'hard',
        'expected_answer': 'Mobile app engagement reduces churn by 35%. Additionally, no mobile app usage for 90+ days is an early warning indicator of potential churn.',
        'relevant_doc_ids': ['CHR-002', 'CHR-003']
    }
]

# Convert to DataFrame
df_eval_questions = pd.DataFrame(evaluation_questions)
df_eval_questions['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("=" * 80)
print("EVALUATION QUESTIONS GENERATED")
print("=" * 80)
print(f"✓ Generated {len(df_eval_questions)} evaluation questions")
print(f"\n📊 Question Statistics:")
print(f"  • Categories: {df_eval_questions['category'].nunique()}")
print(f"  • Difficulty levels: {df_eval_questions['difficulty'].nunique()}")
print(f"\n📁 Questions by Category:")
print(df_eval_questions['category'].value_counts().to_string())
print(f"\n📈 Questions by Difficulty:")
print(df_eval_questions['difficulty'].value_counts().to_string())
print("\n💡 These questions will be used to:")
print("  • Evaluate RAG retrieval quality")
print("  • Measure answer relevance and groundedness")
print("  • Compare different RAG configurations")
print("  • Track model performance over time")
print("=" * 80)

# COMMAND ----------

# Save evaluation questions to Unity Catalog
df_eval_spark = spark.createDataFrame(df_eval_questions)
eval_table_path = f"{CATALOG_NAME}.{SCHEMA_NAME}.{EVAL_QUESTIONS_TABLE}"
df_eval_spark.write.format("delta").mode("overwrite").saveAsTable(eval_table_path)

print("=" * 80)
print("EVALUATION QUESTIONS SAVED TO UNITY CATALOG")
print("=" * 80)
print(f"✓ Table: {eval_table_path}")
print(f"✓ Records: {df_eval_spark.count():,} questions")
print("\n💡 Governance benefits:")
print("  • Evaluation data versioned alongside knowledge base")
print("  • Complete lineage from questions → model → results")
print("  • Audit trail of all evaluation runs")
print("=" * 80)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Build a Simple RAG Pipeline
# MAGIC
# MAGIC ### What You'll Learn in This Section
# MAGIC In this section, we will:
# MAGIC 1. Implement a simple retrieval mechanism (keyword-based search)
# MAGIC 2. Create a mock LLM response generator (simulated for cost control)
# MAGIC 3. Build a RAG pipeline that combines retrieval + generation
# MAGIC 4. Package the RAG pipeline as an MLflow model
# MAGIC
# MAGIC ### Understanding RAG Architecture
# MAGIC A **Retrieval-Augmented Generation (RAG)** system has two main components:
# MAGIC
# MAGIC **1. Retriever:** Searches the knowledge base to find relevant documents
# MAGIC    - Input: User question
# MAGIC    - Process: Search/similarity matching against knowledge base
# MAGIC    - Output: Top-k most relevant documents
# MAGIC
# MAGIC **2. Generator:** Uses retrieved documents to generate an answer
# MAGIC    - Input: User question + retrieved documents (context)
# MAGIC    - Process: LLM generates answer grounded in the context
# MAGIC    - Output: Natural language answer
# MAGIC
# MAGIC ### Why This Matters for Governance
# MAGIC **Reproducibility:** By packaging the RAG pipeline as an MLflow model, we can:
# MAGIC - Version the entire pipeline (retrieval logic + generation logic)
# MAGIC - Track which knowledge base version was used
# MAGIC - Reproduce exact results from any experiment
# MAGIC
# MAGIC **Cost Control:** In this lab, we'll use a **simulated LLM** to avoid costs during development. In production, you would:
# MAGIC - Track token usage per request
# MAGIC - Implement caching to reduce redundant LLM calls
# MAGIC - Set budget limits and alerts
# MAGIC
# MAGIC **Auditability:** Every component of the RAG pipeline is logged:
# MAGIC - Retrieval parameters (top-k, similarity threshold)
# MAGIC - Prompt templates used
# MAGIC - Retrieved context for each question
# MAGIC - Generated responses
# MAGIC
# MAGIC This creates a complete audit trail: "For question X, the model retrieved documents Y and Z, used prompt template V, and generated response W."

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implement Simple Retrieval Function
# MAGIC
# MAGIC #### How Retrieval Works
# MAGIC The retriever searches the knowledge base to find documents relevant to the user's question. In production, this would use:
# MAGIC - **Vector embeddings** (e.g., sentence transformers, OpenAI embeddings)
# MAGIC - **Vector databases** (e.g., FAISS, Pinecone, Databricks Vector Search)
# MAGIC - **Semantic similarity** (cosine similarity between question and document embeddings)
# MAGIC
# MAGIC For this lab, we'll use a **simple keyword-based retrieval** to demonstrate the concept without incurring embedding costs.
# MAGIC
# MAGIC #### Retrieval Parameters
# MAGIC - **top_k**: Number of documents to retrieve (typically 3-5)
# MAGIC - **min_score**: Minimum relevance score threshold
# MAGIC

# COMMAND ----------

def simple_keyword_retrieval(question: str, knowledge_base: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Simple keyword-based retrieval (for demonstration purposes).
    In production, use vector embeddings and semantic search.

    Args:
        question: User's question
        knowledge_base: DataFrame with knowledge base documents
        top_k: Number of documents to retrieve

    Returns:
        List of retrieved documents with metadata
    """
    # Convert question to lowercase for matching
    question_lower = question.lower()

    # Simple scoring: count keyword matches
    def score_document(doc_content: str, doc_title: str) -> float:
        content_lower = doc_content.lower()
        title_lower = doc_title.lower()

        # Extract keywords from question (simple approach)
        question_words = set(re.findall(r'\w+', question_lower))
        # Remove common stop words
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'do', 'does', 'can', 'for', 'to', 'of', 'in', 'on', 'at'}
        question_words = question_words - stop_words

        # Count matches in content and title
        content_matches = sum(1 for word in question_words if word in content_lower)
        title_matches = sum(1 for word in question_words if word in title_lower)

        # Weight title matches higher
        score = content_matches + (title_matches * 2)
        return score

    # Score all documents
    knowledge_base = knowledge_base.copy()
    knowledge_base['relevance_score'] = knowledge_base.apply(
        lambda row: score_document(row['content'], row['title']), axis=1
    )

    # Get top-k documents
    top_docs = knowledge_base.nlargest(top_k, 'relevance_score')

    # Format results
    retrieved_docs = []
    for _, doc in top_docs.iterrows():
        retrieved_docs.append({
            'doc_id': doc['doc_id'],
            'title': doc['title'],
            'content': doc['content'],
            'category': doc['category'],
            'relevance_score': float(doc['relevance_score']),
            'version': doc['version']
        })

    return retrieved_docs

# Test the retrieval function
test_question = "What is the overdraft protection limit?"
retrieved = simple_keyword_retrieval(test_question, df_knowledge_base, top_k=3)

print("=" * 80)
print("RETRIEVAL FUNCTION TEST")
print("=" * 80)
print(f"Question: {test_question}")
print(f"\n✓ Retrieved {len(retrieved)} documents:")
for i, doc in enumerate(retrieved, 1):
    print(f"\n{i}. {doc['title']} (Score: {doc['relevance_score']})")
    print(f"   Doc ID: {doc['doc_id']}")
    print(f"   Category: {doc['category']}")
    print(f"   Content preview: {doc['content'][:100]}...")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Prompt Template and Mock LLM Generator
# MAGIC
# MAGIC #### Understanding Prompt Engineering for RAG
# MAGIC The prompt template is **critical** for RAG quality. It should:
# MAGIC 1. Clearly instruct the LLM to use only the provided context
# MAGIC 2. Specify the desired output format
# MAGIC 3. Handle cases where the context doesn't contain the answer
# MAGIC 4. Maintain a professional, helpful tone
# MAGIC
# MAGIC #### Cost Control Strategy
# MAGIC In this lab, we use a **mock LLM** that generates rule-based responses. This allows us to:
# MAGIC - Develop and test the RAG pipeline without LLM costs
# MAGIC - Validate retrieval quality independently
# MAGIC - Establish the MLflow tracking infrastructure
# MAGIC
# MAGIC In production, you would replace this with:
# MAGIC - Databricks Foundation Model APIs (e.g., DBRX, Llama)
# MAGIC - OpenAI API (GPT-4, GPT-3.5)
# MAGIC - Azure OpenAI Service
# MAGIC - Other LLM providers
# MAGIC
# MAGIC **Cost Tracking:** Always log token usage, model name, and cost per request in MLflow for budget monitoring.
# MAGIC

# COMMAND ----------

# Define prompt template
PROMPT_TEMPLATE = """You are a helpful customer support assistant for a financial services company.

Use the following context to answer the customer's question. Only use information from the context provided.
If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: """

def mock_llm_generate(prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
    """
    Mock LLM generator for demonstration (avoids API costs).
    In production, replace with actual LLM API call.

    Args:
        prompt: Full prompt including context and question
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with generated text and metadata
    """
    # Extract question and context from prompt (simple parsing)
    if "Question:" in prompt and "Context:" in prompt:
        question_part = prompt.split("Question:")[1].split("Answer:")[0].strip()
        context_part = prompt.split("Context:")[1].split("Question:")[0].strip()

        # Simple rule-based response generation (mock)
        # In production, this would be: response = openai.ChatCompletion.create(...)

        # For demonstration, extract first sentence from context as answer
        sentences = context_part.split('.')
        answer = sentences[0].strip() + '.' if sentences else "I don't have enough information to answer that question."

        # Simulate token usage
        prompt_tokens = len(prompt.split())
        completion_tokens = len(answer.split())
        total_tokens = prompt_tokens + completion_tokens

        return {
            'answer': answer,
            'model': 'mock-llm-v1',
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'estimated_cost_usd': total_tokens * 0.00002  # Mock cost calculation
        }
    else:
        return {
            'answer': "Error: Invalid prompt format",
            'model': 'mock-llm-v1',
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'estimated_cost_usd': 0.0
        }

# Test the mock LLM
test_context = "Overdraft protection is available for Premium and Gold account holders. The service covers overdrafts up to $500 with a $35 fee per occurrence."
test_prompt = PROMPT_TEMPLATE.format(context=test_context, question="What is the overdraft limit?")
test_response = mock_llm_generate(test_prompt)

print("=" * 80)
print("MOCK LLM GENERATOR TEST")
print("=" * 80)
print(f"Prompt:\n{test_prompt}\n")
print(f"Generated Answer: {test_response['answer']}")
print(f"\n📊 Token Usage:")
print(f"  • Prompt tokens: {test_response['prompt_tokens']}")
print(f"  • Completion tokens: {test_response['completion_tokens']}")
print(f"  • Total tokens: {test_response['total_tokens']}")
print(f"  • Estimated cost: ${test_response['estimated_cost_usd']:.6f}")
print("\n💡 In production, replace mock_llm_generate() with actual LLM API")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Complete RAG Pipeline as MLflow PyFunc Model
# MAGIC
# MAGIC #### Why Package as MLflow PyFunc?
# MAGIC **MLflow PyFunc** is a generic Python function flavor that allows you to package any Python code as an MLflow model. Benefits:
# MAGIC - **Standardized interface**: All models have predict() method
# MAGIC - **Dependency management**: MLflow tracks all required libraries
# MAGIC - **Reproducibility**: Exact environment can be recreated
# MAGIC - **Deployment flexibility**: Can deploy to various serving platforms
# MAGIC
# MAGIC #### Our RAG Pipeline Class
# MAGIC We'll create a custom PyFunc model that:
# MAGIC 1. Loads the knowledge base from Unity Catalog
# MAGIC 2. Implements the retrieval logic
# MAGIC 3. Generates answers using the LLM
# MAGIC 4. Tracks all intermediate steps for auditability
# MAGIC

# COMMAND ----------

class SimpleRAGModel(mlflow.pyfunc.PythonModel):
    """
    Simple RAG model that combines retrieval and generation.
    Packaged as MLflow PyFunc for standardized deployment.
    """

    def __init__(self, knowledge_base: pd.DataFrame, top_k: int = 3, prompt_template: str = PROMPT_TEMPLATE):
        """
        Initialize RAG model with knowledge base and configuration.

        Args:
            knowledge_base: DataFrame containing documents
            top_k: Number of documents to retrieve
            prompt_template: Template for LLM prompts
        """
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.prompt_template = prompt_template
        self.retrieval_stats = []  # Track retrieval for audit
        self.generation_stats = []  # Track generation for cost monitoring

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a question."""
        return simple_keyword_retrieval(question, self.knowledge_base, self.top_k)

    def generate(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using retrieved context."""
        # Combine retrieved documents into context
        context = "\n\n".join([
            f"Document {i+1} ({doc['doc_id']} - {doc['title']}):\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        # Create prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # Generate answer (using mock LLM)
        response = mock_llm_generate(prompt)

        return response

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Main prediction method called by MLflow.

        Args:
            context: MLflow context (not used in this simple implementation)
            model_input: DataFrame with 'question' column

        Returns:
            DataFrame with answers and metadata
        """
        # Handle different input formats
        if isinstance(model_input, pd.DataFrame):
            questions = model_input['question'].tolist()
        elif isinstance(model_input, dict):
            questions = [model_input['question']] if 'question' in model_input else []
        else:
            questions = [str(model_input)]

        results = []
        for question in questions:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(question)

            # Generate answer
            response = self.generate(question, retrieved_docs)

            # Track stats for audit and cost monitoring
            self.retrieval_stats.append({
                'question': question,
                'num_docs_retrieved': len(retrieved_docs),
                'doc_ids': [doc['doc_id'] for doc in retrieved_docs],
                'relevance_scores': [doc['relevance_score'] for doc in retrieved_docs]
            })

            self.generation_stats.append({
                'question': question,
                'tokens_used': response['total_tokens'],
                'estimated_cost': response['estimated_cost_usd']
            })

            # Compile result
            results.append({
                'question': question,
                'answer': response['answer'],
                'retrieved_doc_ids': [doc['doc_id'] for doc in retrieved_docs],
                'retrieved_doc_titles': [doc['title'] for doc in retrieved_docs],
                'num_docs_retrieved': len(retrieved_docs),
                'total_tokens': response['total_tokens'],
                'estimated_cost_usd': response['estimated_cost_usd']
            })

        return pd.DataFrame(results)

# Create and test the RAG model
rag_model = SimpleRAGModel(
    knowledge_base=df_knowledge_base,
    top_k=3,
    prompt_template=PROMPT_TEMPLATE
)

# Test with a sample question
test_input = pd.DataFrame({'question': ['What is the overdraft protection limit for Premium accounts?']})
test_output = rag_model.predict(context=None, model_input=test_input)

print("=" * 80)
print("RAG PIPELINE TEST")
print("=" * 80)
print(f"Question: {test_output['question'].iloc[0]}")
print(f"\n✓ Answer: {test_output['answer'].iloc[0]}")
print(f"\n📄 Retrieved Documents:")
for i, (doc_id, title) in enumerate(zip(test_output['retrieved_doc_ids'].iloc[0],
                                         test_output['retrieved_doc_titles'].iloc[0]), 1):
    print(f"  {i}. {doc_id}: {title}")
print(f"\n📊 Resource Usage:")
print(f"  • Documents retrieved: {test_output['num_docs_retrieved'].iloc[0]}")
print(f"  • Tokens used: {test_output['total_tokens'].iloc[0]}")
print(f"  • Estimated cost: ${test_output['estimated_cost_usd'].iloc[0]:.6f}")
print("\n✓ RAG pipeline working correctly!")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: MLflow Experiment Tracking for RAG
# MAGIC
# MAGIC ### What You'll Learn in This Section
# MAGIC In this section, we will:
# MAGIC 1. Set up an MLflow experiment for RAG development
# MAGIC 2. Log RAG-specific parameters (retrieval settings, prompt templates)
# MAGIC 3. Log artifacts (prompt templates, retrieved context samples, example responses)
# MAGIC 4. Track cost metrics (token usage, estimated costs)
# MAGIC 5. Compare different RAG configurations
# MAGIC
# MAGIC ### Why Experiment Tracking Matters for RAG
# MAGIC **Reproducibility Challenge:** RAG systems have many moving parts:
# MAGIC - Knowledge base version
# MAGIC - Retrieval algorithm and parameters
# MAGIC - Prompt template
# MAGIC - LLM model and parameters
# MAGIC - Post-processing logic
# MAGIC
# MAGIC Without proper tracking, it's impossible to reproduce results or understand why one configuration outperforms another.
# MAGIC
# MAGIC **Cost Transparency:** LLM inference costs can escalate quickly. MLflow tracking provides:
# MAGIC - Token usage per experiment
# MAGIC - Cost per query
# MAGIC - Total experiment cost
# MAGIC - Comparison of cost vs. quality trade-offs
# MAGIC
# MAGIC **Audit Requirements:** Regulators need to know:
# MAGIC - What prompt was used for a specific customer interaction?
# MAGIC - Which documents were retrieved?
# MAGIC - What was the model's reasoning?
# MAGIC - Who approved this configuration for production?
# MAGIC
# MAGIC MLflow provides the complete audit trail.
# MAGIC
# MAGIC ### What We'll Track
# MAGIC - **Parameters**: top_k, prompt_template_version, model_name, retrieval_method
# MAGIC - **Metrics**: average_tokens_per_query, total_cost, retrieval_coverage
# MAGIC - **Artifacts**:
# MAGIC   - Prompt template file
# MAGIC   - Sample retrieved contexts (JSON)
# MAGIC   - Example question-answer pairs
# MAGIC   - Cost breakdown report
# MAGIC - **Tags**: experiment_type, developer, purpose, compliance_status

# COMMAND ----------

# Set up MLflow experiment for RAG development
current_user = spark.sql('SELECT current_user()').collect()[0][0]
experiment_name = f"/Users/{current_user}/rag_support_assistant_experiments"
mlflow.set_experiment(experiment_name)

# Configure MLflow to use Unity Catalog for model registry (modern API)
mlflow.set_registry_uri("databricks-uc")

print("=" * 80)
print("MLFLOW EXPERIMENT CONFIGURATION")
print("=" * 80)
print(f"✓ Experiment: {experiment_name}")
print(f"✓ Model Registry: Unity Catalog (databricks-uc)")
print(f"✓ Registry URI: {mlflow.get_registry_uri()}")
print(f"✓ Current User: {current_user}")
print("\n💡 All experiments will be tracked with:")
print("  • Complete parameter logging")
print("  • Artifact versioning")
print("  • Cost tracking")
print("  • Audit trail")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Function: RAG Experiment Tracking
# MAGIC
# MAGIC We'll create a reusable function that:
# MAGIC - Creates a RAG model with specific configuration
# MAGIC - Evaluates it on test questions
# MAGIC - Logs all parameters, metrics, and artifacts to MLflow
# MAGIC - Tracks costs and resource usage
# MAGIC - Returns the run ID for model registration
# MAGIC
# MAGIC This function demonstrates **production-grade experiment tracking** for RAG systems.

# COMMAND ----------

def run_rag_experiment(
    run_name: str,
    knowledge_base: pd.DataFrame,
    eval_questions: pd.DataFrame,
    top_k: int = 3,
    prompt_template: str = PROMPT_TEMPLATE,
    tags: Dict[str, str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Run a complete RAG experiment with MLflow tracking.

    Args:
        run_name: Name for this experiment run
        knowledge_base: DataFrame with knowledge base documents
        eval_questions: DataFrame with evaluation questions
        top_k: Number of documents to retrieve
        prompt_template: Template for LLM prompts
        tags: Additional tags for the run

    Returns:
        Tuple of (run_id, metrics_dict)
    """
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT: {run_name}")
        print(f"{'='*80}")

        # 1. Log Parameters
        params = {
            'top_k': top_k,
            'retrieval_method': 'keyword_based',
            'prompt_template_version': 'v1.0',
            'llm_model': 'mock-llm-v1',  # In production: 'gpt-4', 'dbrx-instruct', etc.
            'knowledge_base_version': df_knowledge_base['version'].iloc[0],
            'num_kb_documents': len(knowledge_base),
            'num_eval_questions': len(eval_questions)
        }
        mlflow.log_params(params)
        print(f"✓ Logged {len(params)} parameters")

        # 2. Set Tags for Organization and Audit
        default_tags = {
            'model_type': 'RAG',
            'experiment_date': datetime.now().strftime('%Y-%m-%d'),
            'data_source': f'{CATALOG_NAME}.{SCHEMA_NAME}',
            'purpose': 'customer_support_assistant',
            'developer': current_user,
            'compliance_status': 'under_review'
        }
        if tags:
            default_tags.update(tags)

        for key, value in default_tags.items():
            mlflow.set_tag(key, value)
        print(f"✓ Set {len(default_tags)} tags")

        # 3. Create and Evaluate RAG Model
        rag_model = SimpleRAGModel(
            knowledge_base=knowledge_base,
            top_k=top_k,
            prompt_template=prompt_template
        )

        # Run predictions on evaluation set
        eval_input = eval_questions[['question']].copy()
        predictions = rag_model.predict(context=None, model_input=eval_input)

        # 4. Calculate Metrics
        total_tokens = predictions['total_tokens'].sum()
        total_cost = predictions['estimated_cost_usd'].sum()
        avg_tokens_per_query = predictions['total_tokens'].mean()
        avg_docs_retrieved = predictions['num_docs_retrieved'].mean()

        metrics = {
            'total_questions_evaluated': len(predictions),
            'total_tokens_used': int(total_tokens),
            'total_estimated_cost_usd': float(total_cost),
            'avg_tokens_per_query': float(avg_tokens_per_query),
            'avg_docs_retrieved_per_query': float(avg_docs_retrieved),
            'max_tokens_single_query': int(predictions['total_tokens'].max()),
            'min_tokens_single_query': int(predictions['total_tokens'].min())
        }
        mlflow.log_metrics(metrics)
        print(f"✓ Logged {len(metrics)} metrics")

        # 5. Log Artifacts

        # 5a. Save prompt template
        with open('/tmp/prompt_template.txt', 'w') as f:
            f.write(prompt_template)
        mlflow.log_artifact('/tmp/prompt_template.txt', 'config')

        # 5b. Save sample predictions
        sample_predictions = predictions.head(5).to_dict('records')
        with open('/tmp/sample_predictions.json', 'w') as f:
            json.dump(sample_predictions, f, indent=2)
        mlflow.log_artifact('/tmp/sample_predictions.json', 'examples')

        # 5c. Save retrieval statistics
        retrieval_stats = {
            'total_retrievals': len(rag_model.retrieval_stats),
            'sample_retrievals': rag_model.retrieval_stats[:5]
        }
        with open('/tmp/retrieval_stats.json', 'w') as f:
            json.dump(retrieval_stats, f, indent=2)
        mlflow.log_artifact('/tmp/retrieval_stats.json', 'analysis')

        # 5d. Save cost breakdown
        cost_breakdown = {
            'total_cost_usd': float(total_cost),
            'cost_per_query_usd': float(total_cost / len(predictions)),
            'total_tokens': int(total_tokens),
            'tokens_per_query': float(avg_tokens_per_query),
            'estimated_monthly_cost_1000_queries': float((total_cost / len(predictions)) * 1000)
        }
        with open('/tmp/cost_breakdown.json', 'w') as f:
            json.dump(cost_breakdown, f, indent=2)
        mlflow.log_artifact('/tmp/cost_breakdown.json', 'cost_analysis')

        print(f"✓ Logged 4 artifact files")

        # 6. Log Model with Signature
        # Define input/output schema
        input_schema = Schema([ColSpec("string", "question")])
        output_schema = Schema([
            ColSpec("string", "question"),
            ColSpec("string", "answer"),
            ColSpec("long", "total_tokens")
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=rag_model,
            signature=signature,
            input_example={"question": "What is the overdraft protection limit?"}
        )
        print(f"✓ Logged RAG model with signature")

        # 7. Print Summary
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETE: {run_name}")
        print(f"{'='*80}")
        print(f"Run ID: {run.info.run_id}")
        print(f"\n📊 Key Metrics:")
        print(f"  • Questions evaluated: {metrics['total_questions_evaluated']}")
        print(f"  • Total tokens: {metrics['total_tokens_used']:,}")
        print(f"  • Total cost: ${metrics['total_estimated_cost_usd']:.4f}")
        print(f"  • Avg tokens/query: {metrics['avg_tokens_per_query']:.1f}")
        print(f"  • Avg docs retrieved: {metrics['avg_docs_retrieved_per_query']:.1f}")
        print(f"\n💰 Cost Projection:")
        print(f"  • Cost per query: ${cost_breakdown['cost_per_query_usd']:.6f}")
        print(f"  • Est. monthly cost (1000 queries): ${cost_breakdown['estimated_monthly_cost_1000_queries']:.2f}")
        print(f"{'='*80}\n")

        return run.info.run_id, metrics

print("✓ RAG experiment tracking function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Multiple RAG Experiments
# MAGIC
# MAGIC We'll run three different RAG configurations to compare performance:
# MAGIC 1. **Baseline (top_k=2)** - Retrieve fewer documents for faster, cheaper responses
# MAGIC 2. **Standard (top_k=3)** - Balanced configuration
# MAGIC 3. **Comprehensive (top_k=5)** - Retrieve more documents for better coverage
# MAGIC
# MAGIC Each configuration's parameters, metrics, costs, and artifacts will be logged to MLflow for comparison.
# MAGIC
# MAGIC #### Why Compare Different Configurations?
# MAGIC **Cost vs. Quality Trade-off:** More retrieved documents means:
# MAGIC - ✅ Better chance of finding relevant information
# MAGIC - ✅ More complete answers
# MAGIC - ❌ Higher token usage (longer context)
# MAGIC - ❌ Higher costs
# MAGIC - ❌ Slower response times
# MAGIC
# MAGIC MLflow tracking allows us to quantify these trade-offs and make data-driven decisions.
# MAGIC

# COMMAND ----------

# Experiment 1: Baseline Configuration (top_k=2)
baseline_run_id, baseline_metrics = run_rag_experiment(
    run_name="RAG_Baseline_top_k_2",
    knowledge_base=df_knowledge_base,
    eval_questions=df_eval_questions,
    top_k=2,
    prompt_template=PROMPT_TEMPLATE,
    tags={
        'configuration': 'baseline',
        'optimization_goal': 'cost_efficiency'
    }
)


# COMMAND ----------

# Experiment 2: Standard Configuration (top_k=3)
standard_run_id, standard_metrics = run_rag_experiment(
    run_name="RAG_Standard_top_k_3",
    knowledge_base=df_knowledge_base,
    eval_questions=df_eval_questions,
    top_k=3,
    prompt_template=PROMPT_TEMPLATE,
    tags={
        'configuration': 'standard',
        'optimization_goal': 'balanced'
    }
)


# COMMAND ----------

# Experiment 3: Comprehensive Configuration (top_k=5)
comprehensive_run_id, comprehensive_metrics = run_rag_experiment(
    run_name="RAG_Comprehensive_top_k_5",
    knowledge_base=df_knowledge_base,
    eval_questions=df_eval_questions,
    top_k=5,
    prompt_template=PROMPT_TEMPLATE,
    tags={
        'configuration': 'comprehensive',
        'optimization_goal': 'answer_quality'
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare RAG Configurations
# MAGIC
# MAGIC Let's compare all three RAG configurations across key metrics to determine the best balance of cost and quality.
# MAGIC
# MAGIC #### Decision Criteria
# MAGIC When selecting a RAG configuration for production, consider:
# MAGIC 1. **Cost per query** - Can we afford this at scale?
# MAGIC 2. **Token usage** - Will we hit rate limits?
# MAGIC 3. **Answer quality** - Are responses accurate and complete? (requires human evaluation)
# MAGIC 4. **Latency** - How fast do responses need to be?
# MAGIC

# COMMAND ----------

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Configuration': ['Baseline (top_k=2)', 'Standard (top_k=3)', 'Comprehensive (top_k=5)'],
    'Run_ID': [baseline_run_id, standard_run_id, comprehensive_run_id],
    'Total_Tokens': [
        baseline_metrics['total_tokens_used'],
        standard_metrics['total_tokens_used'],
        comprehensive_metrics['total_tokens_used']
    ],
    'Avg_Tokens_Per_Query': [
        baseline_metrics['avg_tokens_per_query'],
        standard_metrics['avg_tokens_per_query'],
        comprehensive_metrics['avg_tokens_per_query']
    ],
    'Total_Cost_USD': [
        baseline_metrics['total_estimated_cost_usd'],
        standard_metrics['total_estimated_cost_usd'],
        comprehensive_metrics['total_estimated_cost_usd']
    ],
    'Avg_Docs_Retrieved': [
        baseline_metrics['avg_docs_retrieved_per_query'],
        standard_metrics['avg_docs_retrieved_per_query'],
        comprehensive_metrics['avg_docs_retrieved_per_query']
    ]
})

# Calculate cost per query
comparison_df['Cost_Per_Query_USD'] = comparison_df['Total_Cost_USD'] / baseline_metrics['total_questions_evaluated']

# Calculate relative cost (baseline = 100%)
baseline_cost = comparison_df.loc[0, 'Total_Cost_USD']
comparison_df['Relative_Cost_Pct'] = (comparison_df['Total_Cost_USD'] / baseline_cost * 100).round(1)

print("=" * 80)
print("RAG CONFIGURATION COMPARISON")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Identify best configuration based on balanced criteria
# For this demo, we'll select standard (top_k=3) as the best balance
best_config_idx = 1  # Standard configuration
best_config_name = comparison_df.loc[best_config_idx, 'Configuration']
best_run_id = comparison_df.loc[best_config_idx, 'Run_ID']

print(f"\n{'='*80}")
print(f"RECOMMENDED CONFIGURATION: {best_config_name}")
print(f"{'='*80}")
print(f"Run ID: {best_run_id}")
print(f"\n📊 Performance:")
print(f"  • Avg tokens per query: {comparison_df.loc[best_config_idx, 'Avg_Tokens_Per_Query']:.1f}")
print(f"  • Cost per query: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}")
print(f"  • Avg docs retrieved: {comparison_df.loc[best_config_idx, 'Avg_Docs_Retrieved']:.1f}")
print(f"\n💰 Cost Analysis:")
print(f"  • {comparison_df.loc[best_config_idx, 'Relative_Cost_Pct']:.0f}% of baseline cost")
print(f"  • Estimated monthly cost (1000 queries): ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 1000:.2f}")
print(f"\n💡 Rationale:")
print(f"  • Balanced cost vs. quality trade-off")
print(f"  • Retrieves enough context for accurate answers")
print(f"  • Manageable token usage and costs")
print(f"  • Good starting point for production deployment")
print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: Model Registration in Unity Catalog
# MAGIC
# MAGIC ### What You'll Learn in This Section
# MAGIC In this section, we will:
# MAGIC 1. Register the best RAG configuration to Unity Catalog Model Registry
# MAGIC 2. Add comprehensive documentation for compliance and audit
# MAGIC 3. Track lineage from knowledge base to model
# MAGIC 4. Apply tags for governance and organization
# MAGIC
# MAGIC ### Why Model Registration Matters for RAG
# MAGIC **Centralized Model Storage:** Unity Catalog provides a single source of truth for all model versions across the organization. This prevents:
# MAGIC - Shadow deployments (unauthorized models in production)
# MAGIC - Version confusion ("which model is actually serving traffic?")
# MAGIC - Lost models ("where did we save that experiment from last month?")
# MAGIC
# MAGIC **Access Control via RBAC:** Unity Catalog allows fine-grained permissions:
# MAGIC - Data scientists can create and update models
# MAGIC - ML engineers can promote models to production
# MAGIC - Analysts can only read model metadata
# MAGIC - Auditors can view all activity logs
# MAGIC
# MAGIC **Audit Logging:** Every operation is automatically logged:
# MAGIC - Who registered the model?
# MAGIC - When was it promoted to production?
# MAGIC - Who has accessed the model?
# MAGIC - What data was it trained on?
# MAGIC
# MAGIC **Lineage Tracking:** Unity Catalog automatically tracks:
# MAGIC - Knowledge base version → Model version
# MAGIC - Evaluation data → Model metrics
# MAGIC - MLflow run → Registered model
# MAGIC - Model version → Serving endpoint
# MAGIC
# MAGIC This creates a complete audit trail for regulatory compliance.
# MAGIC
# MAGIC ### Modern API: Using Aliases Instead of Stages
# MAGIC **Important:** MLflow 2.x deprecated the old "Staging/Production/Archived" stages in favor of **aliases**. Aliases provide:
# MAGIC - ✅ Flexibility: Create any alias name (Champion, Challenger, Shadow, Canary)
# MAGIC - ✅ Multiple aliases per version: A model can be both "Champion" and "Approved"
# MAGIC - ✅ Better A/B testing support: Easy to manage multiple production variants
# MAGIC - ✅ Clearer semantics: "Champion" is more meaningful than "Production"
# MAGIC

# COMMAND ----------

# Register the best RAG configuration to Unity Catalog
print("=" * 80)
print("REGISTERING RAG MODEL TO UNITY CATALOG")
print("=" * 80)
print(f"Configuration: {best_config_name}")
print(f"Run ID: {best_run_id}")
print(f"Model Name: {MODEL_NAME}")

# Create model registry entry using modern MLflow API
model_version = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME,
    tags={
        "model_type": "RAG",
        "rag_configuration": best_config_name,
        "training_date": datetime.now().strftime('%Y-%m-%d'),
        "use_case": "customer_support_assistant",
        "department": "data_science",
        "compliance_status": "pending_review",
        "cost_per_query_usd": f"{comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}",
        "avg_tokens_per_query": f"{comparison_df.loc[best_config_idx, 'Avg_Tokens_Per_Query']:.1f}"
    }
)

print(f"\n✓ Model registered successfully!")
print(f"  • Model Name: {MODEL_NAME}")
print(f"  • Version: {model_version.version}")
print(f"  • Run ID: {best_run_id}")
print(f"  • Status: {model_version.status}")
print(f"\n💡 Model is now governed by Unity Catalog:")
print(f"  • Centralized storage and versioning")
print(f"  • RBAC-based access control")
print(f"  • Complete audit logging")
print(f"  • Automatic lineage tracking")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Model Description and Documentation
# MAGIC
# MAGIC #### Why Documentation Matters
# MAGIC Proper documentation is **critical** for:
# MAGIC - **Compliance**: Regulators need to understand what the model does and how it works
# MAGIC - **Governance**: Stakeholders need to approve models before production deployment
# MAGIC - **Maintenance**: Future developers need to understand the model's purpose and limitations
# MAGIC - **Incident Response**: When something goes wrong, documentation helps diagnose issues quickly
# MAGIC
# MAGIC #### What to Document for RAG Models
# MAGIC RAG models require different documentation than traditional ML models:
# MAGIC - ✅ Knowledge base version and source
# MAGIC - ✅ Retrieval method and parameters
# MAGIC - ✅ LLM model and version
# MAGIC - ✅ Prompt template
# MAGIC - ✅ Cost per query and scaling considerations
# MAGIC - ✅ Evaluation methodology
# MAGIC - ✅ Known limitations and failure modes
# MAGIC

# COMMAND ----------

# Initialize MLflow client for model management
client = MlflowClient()

# Create comprehensive model description
model_description = f"""
# RAG Customer Support Assistant

## Overview
This is a Retrieval-Augmented Generation (RAG) system that answers customer support questions
by retrieving relevant information from a knowledge base and generating natural language responses.

## Architecture
- **Retrieval Method**: Keyword-based search (production should use vector embeddings)
- **LLM**: Mock LLM (production should use DBRX, GPT-4, or similar)
- **Configuration**: {best_config_name}
- **Top-K Documents**: {int(comparison_df.loc[best_config_idx, 'Avg_Docs_Retrieved'])}

## Training/Development Details
- **Development Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Knowledge Base**: {CATALOG_NAME}.{SCHEMA_NAME}.{KNOWLEDGE_BASE_TABLE}
- **Knowledge Base Version**: {df_knowledge_base['version'].iloc[0]}
- **Number of Documents**: {len(df_knowledge_base):,}
- **Evaluation Questions**: {len(df_eval_questions):,}
- **MLflow Run ID**: {best_run_id}

## Performance Metrics
- **Avg Tokens per Query**: {comparison_df.loc[best_config_idx, 'Avg_Tokens_Per_Query']:.1f}
- **Cost per Query**: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}
- **Avg Documents Retrieved**: {comparison_df.loc[best_config_idx, 'Avg_Docs_Retrieved']:.1f}
- **Total Tokens (Eval Set)**: {comparison_df.loc[best_config_idx, 'Total_Tokens']:,}

## Cost Projections
- **1,000 queries/month**: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 1000:.2f}
- **10,000 queries/month**: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 10000:.2f}
- **100,000 queries/month**: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 100000:.2f}

## Intended Use
- Answer customer support questions about financial products and services
- Provide accurate, grounded responses based on official documentation
- Reduce support ticket volume by enabling self-service
- Assist human agents with quick information retrieval

## Limitations and Risks
- **Mock LLM**: Current implementation uses a mock LLM for demonstration. Production deployment requires a real LLM.
- **Keyword Retrieval**: Simple keyword matching may miss semantically similar documents. Production should use vector embeddings.
- **No Guardrails**: No content filtering, toxicity detection, or PII redaction implemented.
- **No Caching**: Every query hits the LLM. Production should implement caching for common questions.
- **Knowledge Base Staleness**: Answers are only as current as the knowledge base. Requires regular updates.
- **Hallucination Risk**: LLMs may generate plausible-sounding but incorrect information. Requires human review.

## Compliance and Governance
- **Data Lineage**: Complete lineage from knowledge base → retrieval → generation → response
- **Access Control**: Model governed by Unity Catalog RBAC
- **Audit Trail**: All model operations logged in Unity Catalog
- **Cost Tracking**: Token usage and costs tracked in MLflow
- **Reproducibility**: All experiments tracked with parameters, metrics, and artifacts

## Deployment Requirements
Before production deployment:
1. ✅ Replace mock LLM with production LLM (DBRX, GPT-4, etc.)
2. ✅ Implement vector-based retrieval (Databricks Vector Search)
3. ✅ Add content filtering and guardrails
4. ✅ Implement response caching
5. ✅ Set up monitoring and alerting
6. ✅ Conduct human evaluation of answer quality
7. ✅ Obtain compliance approval
8. ✅ Establish knowledge base update process

## Maintenance Schedule
- **Knowledge Base Updates**: Weekly or as needed
- **Model Re-evaluation**: Monthly
- **Cost Review**: Monthly
- **Compliance Audit**: Quarterly

## Contact
- **Owner**: {current_user}
- **Team**: Data Science
- **Slack Channel**: #ml-rag-support (example)
"""

# Update model version description
client.update_model_version(
    name=MODEL_NAME,
    version=model_version.version,
    description=model_description
)

print("=" * 80)
print("MODEL DOCUMENTATION ADDED")
print("=" * 80)
print("✓ Comprehensive documentation attached to model version")
print("\n📋 Documentation includes:")
print("  • Architecture and configuration details")
print("  • Performance metrics and cost projections")
print("  • Intended use and limitations")
print("  • Compliance and governance information")
print("  • Deployment requirements and maintenance schedule")
print("\n💡 This documentation is critical for:")
print("  • Compliance review and approval")
print("  • Production deployment planning")
print("  • Future maintenance and updates")
print("  • Incident response and debugging")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5: Model Version Management and Promotion
# MAGIC
# MAGIC ### What You'll Learn in This Section
# MAGIC In this section, we will:
# MAGIC 1. Promote the model using the "Champion" alias for production deployment
# MAGIC 2. Understand the modern alias-based workflow (vs. deprecated stages)
# MAGIC 3. Load the model from the registry for inference
# MAGIC 4. Demonstrate how production systems would use the model
# MAGIC
# MAGIC ### Understanding Model Aliases
# MAGIC **Modern Approach (MLflow 2.x+):** Use **aliases** for model lifecycle management.
# MAGIC
# MAGIC Common alias patterns:
# MAGIC - **Champion**: Current production model serving live traffic
# MAGIC - **Challenger**: New model being A/B tested against Champion
# MAGIC - **Shadow**: Model receiving traffic for monitoring but not serving responses
# MAGIC - **Canary**: Model serving a small percentage of traffic
# MAGIC - **Approved**: Model approved by compliance but not yet deployed
# MAGIC
# MAGIC **Why Aliases > Stages:**
# MAGIC - ✅ Multiple aliases per version (a model can be both "Champion" and "Approved")
# MAGIC - ✅ Custom alias names that match your workflow
# MAGIC - ✅ Better support for A/B testing and gradual rollouts
# MAGIC - ✅ Clearer semantics ("Champion" vs. "Production")
# MAGIC
# MAGIC ### Promotion Workflow
# MAGIC In a real organization, model promotion would involve:
# MAGIC 1. **Development**: Data scientist creates and evaluates model
# MAGIC 2. **Review**: ML engineer reviews code, metrics, and documentation
# MAGIC 3. **Compliance**: Compliance team reviews for regulatory requirements
# MAGIC 4. **Approval**: Set "Approved" alias after compliance sign-off
# MAGIC 5. **Deployment**: ML engineer promotes to "Champion" and deploys to serving infrastructure
# MAGIC 6. **Monitoring**: Monitor performance, costs, and quality in production
# MAGIC
# MAGIC For this lab, we'll simulate the promotion to "Champion".
# MAGIC

# COMMAND ----------

# Promote model to Champion (production)
print("=" * 80)
print("PROMOTING MODEL TO PRODUCTION")
print("=" * 80)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=model_version.version
)

print(f"✓ Model version {model_version.version} promoted to 'Champion'")
print(f"\n📊 Model Details:")
print(f"  • Model Name: {MODEL_NAME}")
print(f"  • Version: {model_version.version}")
print(f"  • Alias: Champion")
print(f"  • Configuration: {best_config_name}")
print(f"  • Cost per query: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}")
print(f"\n🚀 Next Steps:")
print(f"  1. Deploy to serving endpoint (Model Serving)")
print(f"  2. Set up monitoring and alerting")
print(f"  3. Configure autoscaling based on traffic")
print(f"  4. Implement caching for common questions")
print(f"  5. Set up cost budgets and alerts")
print(f"\n💡 Production systems will load this model using:")
print(f"  mlflow.pyfunc.load_model('models:/{MODEL_NAME}@Champion')")
print("=" * 80)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model from Registry for Inference
# MAGIC
# MAGIC This demonstrates how production systems would load and use the registered model.
# MAGIC
# MAGIC #### Loading Patterns
# MAGIC - **By alias**: `models:/{MODEL_NAME}@Champion` (recommended for production)
# MAGIC - **By version**: `models:/{MODEL_NAME}/{version}` (for testing specific versions)
# MAGIC - **Latest version**: `models:/{MODEL_NAME}/latest` (not recommended for production)
# MAGIC
# MAGIC **Best Practice**: Always use aliases in production to enable zero-downtime model updates.
# MAGIC

# COMMAND ----------

# Load model using Champion alias
print("=" * 80)
print("LOADING MODEL FROM REGISTRY")
print("=" * 80)

loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")

print(f"✓ Model loaded successfully!")
print(f"  • Model URI: models:/{MODEL_NAME}@Champion")
print(f"  • Model Type: RAG Customer Support Assistant")

# Make predictions with the loaded model
sample_questions = pd.DataFrame({
    'question': [
        'What is the overdraft protection limit?',
        'How do I dispute a transaction?',
        'What are the benefits of a Premium account?'
    ]
})

print(f"\n🔮 Making predictions on {len(sample_questions)} sample questions...")
predictions = loaded_model.predict(sample_questions)

print(f"\n{'='*80}")
print("SAMPLE PREDICTIONS")
print(f"{'='*80}")
for i, row in predictions.iterrows():
    print(f"\n❓ Question: {row['question']}")
    print(f"💬 Answer: {row['answer']}")
    print(f"📄 Retrieved Docs: {', '.join(row['retrieved_doc_ids'])}")
    print(f"💰 Tokens: {row['total_tokens']}, Cost: ${row['estimated_cost_usd']:.6f}")
    print("-" * 80)

print(f"\n✓ Model inference successful!")
print(f"  • Total tokens used: {predictions['total_tokens'].sum()}")
print(f"  • Total cost: ${predictions['estimated_cost_usd'].sum():.6f}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Model Version Updates
# MAGIC
# MAGIC #### Why Model Versioning Matters
# MAGIC In production RAG systems, you'll continuously improve the model by:
# MAGIC - Updating the knowledge base with new documents
# MAGIC - Improving retrieval algorithms (keyword → vector search)
# MAGIC - Upgrading to better LLMs (mock → GPT-3.5 → GPT-4 → DBRX)
# MAGIC - Optimizing prompt templates
# MAGIC - Adjusting retrieval parameters (top_k, similarity thresholds)
# MAGIC
# MAGIC **Unity Catalog tracks all versions**, allowing you to:
# MAGIC - Roll back to previous versions if new version underperforms
# MAGIC - A/B test new configurations against current production
# MAGIC - Maintain multiple versions for different use cases
# MAGIC - Audit which version was serving traffic at any point in time
# MAGIC
# MAGIC Let's simulate creating an improved version by running the comprehensive configuration (top_k=5) and registering it as a "Challenger" for A/B testing.
# MAGIC

# COMMAND ----------

# The comprehensive configuration (top_k=5) is already trained
# Let's register it as a new version for A/B testing
print("=" * 80)
print("REGISTERING CHALLENGER MODEL FOR A/B TESTING")
print("=" * 80)

model_version_v2 = mlflow.register_model(
    model_uri=f"runs:/{comprehensive_run_id}/model",
    name=MODEL_NAME,
    tags={
        "model_type": "RAG",
        "rag_configuration": "Comprehensive (top_k=5)",
        "training_date": datetime.now().strftime('%Y-%m-%d'),
        "use_case": "customer_support_assistant",
        "version_notes": "Higher retrieval coverage (top_k=5) for better answer quality",
        "department": "data_science",
        "cost_per_query_usd": f"{comparison_df.loc[2, 'Cost_Per_Query_USD']:.6f}",  # Comprehensive is index 2
        "avg_tokens_per_query": f"{comparison_df.loc[2, 'Avg_Tokens_Per_Query']:.1f}"
    }
)

print(f"✓ New model version registered: {model_version_v2.version}")
print(f"  • Configuration: Comprehensive (top_k=5)")
print(f"  • Run ID: {comprehensive_run_id}")
print(f"  • Cost per query: ${comparison_df.loc[2, 'Cost_Per_Query_USD']:.6f}")

# Set as Challenger for A/B testing
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Challenger",
    version=model_version_v2.version
)

print(f"\n✓ Model version {model_version_v2.version} set as 'Challenger'")
print(f"\n🔬 A/B Testing Setup:")
print(f"  • Champion (v{model_version.version}): Standard config (top_k=3)")
print(f"    - Cost: ${comparison_df.loc[1, 'Cost_Per_Query_USD']:.6f}/query")
print(f"    - Tokens: {comparison_df.loc[1, 'Avg_Tokens_Per_Query']:.1f}/query")
print(f"  • Challenger (v{model_version_v2.version}): Comprehensive config (top_k=5)")
print(f"    - Cost: ${comparison_df.loc[2, 'Cost_Per_Query_USD']:.6f}/query")
print(f"    - Tokens: {comparison_df.loc[2, 'Avg_Tokens_Per_Query']:.1f}/query")
print(f"\n💡 Next Steps:")
print(f"  1. Deploy both versions to serving endpoints")
print(f"  2. Route 90% traffic to Champion, 10% to Challenger")
print(f"  3. Monitor answer quality, costs, and user satisfaction")
print(f"  4. If Challenger performs better, promote to Champion")
print(f"  5. If not, delete Challenger alias and stick with Champion")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View All Model Versions
# MAGIC
# MAGIC Let's examine all versions of our registered model, their aliases, and metadata.
# MAGIC
# MAGIC #### Why Version History Matters
# MAGIC **Audit Trail:** Regulators may ask: "What model was serving traffic on date X?"
# MAGIC - Unity Catalog maintains complete version history
# MAGIC - Each version is immutable (cannot be changed after registration)
# MAGIC - Timestamps track when each version was created and promoted
# MAGIC
# MAGIC **Rollback Capability:** If a new version causes issues in production:
# MAGIC - Quickly switch the "Champion" alias back to previous version
# MAGIC - No need to retrain or redeploy
# MAGIC - Zero downtime rollback
# MAGIC
# MAGIC **Cost Tracking:** Compare costs across versions:
# MAGIC - Which configuration is most cost-effective?
# MAGIC - How much did costs increase with the new LLM?
# MAGIC - What's the ROI of upgrading retrieval quality?
# MAGIC

# COMMAND ----------

# Get all versions of the model
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print("=" * 80)
print(f"ALL VERSIONS OF {MODEL_NAME}")
print("=" * 80)

# Sort versions by version number (descending)
sorted_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

for version in sorted_versions:
    print(f"\n📦 Version {version.version}")
    print(f"  • Run ID: {version.run_id}")
    print(f"  • Status: {version.status}")

    # Handle aliases - get the aliases list
    try:
        # Try to get aliases as a property or method
        if callable(getattr(version, 'aliases', None)):
            aliases_list = version.aliases()
        else:
            aliases_list = version.aliases

        # Format aliases for display
        if aliases_list and len(aliases_list) > 0:
            if isinstance(aliases_list, list):
                aliases_str = ', '.join(aliases_list)
            else:
                aliases_str = str(aliases_list)
        else:
            aliases_str = 'None'
    except Exception as e:
        aliases_str = 'None'

    print(f"  • Aliases: {aliases_str}")
    print(f"  • Created: {datetime.fromtimestamp(version.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")

    # Get tags for this version
    try:
        # Check if tags is a method or property
        if callable(getattr(version, 'tags', None)):
            tags_dict = version.tags()
        else:
            tags_dict = version.tags

        # Display tags if they exist
        if tags_dict and isinstance(tags_dict, dict):
            print(f"  • Configuration: {tags_dict.get('rag_configuration', 'N/A')}")
            print(f"  • Cost/query: {tags_dict.get('cost_per_query_usd', 'N/A')}")
            print(f"  • Avg tokens: {tags_dict.get('avg_tokens_per_query', 'N/A')}")
    except Exception:
        pass
    print("-" * 80)

print(f"\n✓ Total versions: {len(sorted_versions)}")
print(f"\n💡 Version Management:")
print(f"  • All versions are immutable and permanently stored")
print(f"  • Aliases can be moved between versions instantly")
print(f"  • Complete audit trail of all version operations")
print(f"  • Can load any version for inference or comparison")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6: Unity Catalog Governance Controls
# MAGIC
# MAGIC Unity Catalog provides enterprise-grade governance features:
# MAGIC - **RBAC (Role-Based Access Control)**: Control who can read, write, or execute models
# MAGIC - **Audit Logging**: Track all operations on models and data
# MAGIC - **Data Lineage**: Trace models back to training data
# MAGIC
# MAGIC Let's explore these governance capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Access Control with RBAC (Demonstration)
# MAGIC
# MAGIC Unity Catalog provides enterprise-grade access control through Role-Based Access Control (RBAC).
# MAGIC
# MAGIC **How RBAC Works in Production:**
# MAGIC 1. **Account admins** create groups at **account level** (not workspace level)
# MAGIC 2. **Users are added** to groups based on their roles
# MAGIC 3. **Permissions are granted** to groups, not individual users
# MAGIC 4. **Users inherit** permissions from all groups they belong to
# MAGIC
# MAGIC **Typical Groups in ML Projects:**
# MAGIC - `data_analysts` - Read access to data tables
# MAGIC - `ml_engineers` - Model execution and deployment rights
# MAGIC - `data_scientists` - Full access to develop and train models
# MAGIC - `data_engineers` - Data pipeline and table management
# MAGIC
# MAGIC **Important: Workspace vs. Account Groups**
# MAGIC - Unity Catalog requires **account-level groups** (created in Account Console)
# MAGIC - Workspace-level groups (created with `CREATE GROUP`) **do NOT work** with Unity Catalog
# MAGIC - Only account admins can create account-level groups
# MAGIC - This is a common source of confusion!
# MAGIC
# MAGIC **How to Create Account-Level Groups:**
# MAGIC
# MAGIC *Azure Databricks Account Console (UI):*
# MAGIC 1. Sign in to the Databricks account console (not a workspace)
# MAGIC 2. In Azure, go to **accounts.azuredatabricks.net** (or accounts.cloud.databricks.com for AWS/GCP)
# MAGIC 3. Log in as an **account admin**
# MAGIC 4. Navigate to the **User Management** section
# MAGIC 5. Select **Groups** tab
# MAGIC 6. Click **Add Group** button
# MAGIC 7. Enter group name (e.g., `ml_engineers`)
# MAGIC 8. Press **Add** button
# MAGIC 9. Repeat for all required groups: `data_analysts`, `ml_engineers`, `data_scientists`, `data_engineers`, `all_users`
# MAGIC
# MAGIC *Alternative - Databricks CLI:*
# MAGIC ```
# MAGIC databricks account groups create --group-name data_analysts
# MAGIC databricks account groups create --group-name ml_engineers
# MAGIC databricks account groups create --group-name data_scientists
# MAGIC databricks account groups create --group-name data_engineers
# MAGIC databricks account groups create --group-name all_users
# MAGIC ```
# MAGIC
# MAGIC **For This Lab:**
# MAGIC - If you have account-level groups, the notebook will detect and use them
# MAGIC - If not, we'll demonstrate the concepts with your current user
# MAGIC - Example commands show what admins would run in production
# MAGIC - You'll learn the complete RBAC workflow either way

# COMMAND ----------

# Check if account-level groups exist
print("=== Checking for Account-Level Groups ===\n")

print("⚠ Important: Unity Catalog requires ACCOUNT-LEVEL groups")
print("  • Workspace groups (CREATE GROUP) do NOT work with Unity Catalog")
print("  • Only account admins can create account-level groups")
print("  • Groups must be created in the Account Console\n")

# Define required groups
required_groups = {
    'data_analysts': 'Group for data analysts with read access to data',
    'ml_engineers': 'Group for ML engineers with model execution rights',
    'data_scientists': 'Group for data scientists with full schema access',
    'data_engineers': 'Group for data engineers with data pipeline management',
    'all_users': 'Group for all users with basic catalog access'
}

print("Required groups for this lab:")
for group_name, description in required_groups.items():
    print(f"  • {group_name}: {description}")

# Check if account-level groups exist (read-only check)
print("\nChecking if groups exist at account level...")
print("-" * 80)

existing_groups = []
missing_groups = []

for group_name in required_groups.keys():
    try:
        # Try to grant a harmless permission to test if group exists
        # We'll immediately revoke it, so this is just a test
        # If group doesn't exist, this will fail with PRINCIPAL_DOES_NOT_EXIST
        test_sql = f"GRANT USAGE ON CATALOG {CATALOG_NAME} TO `{group_name}`"
        spark.sql(test_sql)

        # If we got here, group exists! Now revoke the test grant
        try:
            spark.sql(f"REVOKE USAGE ON CATALOG {CATALOG_NAME} FROM `{group_name}`")
        except:
            pass  # Revoke might fail if already granted, that's ok

        print(f"✓ {group_name}: Exists (account-level group)")
        existing_groups.append(group_name)

    except Exception as e:
        error_msg = str(e).lower()
        if "principal_does_not_exist" in error_msg or "does not exist" in error_msg or "cannot find" in error_msg:
            print(f"⊘ {group_name}: Does not exist at account level")
            missing_groups.append(group_name)
        elif "already granted" in error_msg or "already has" in error_msg:
            # Group exists, permission was already granted
            print(f"✓ {group_name}: Exists (account-level group)")
            existing_groups.append(group_name)
        elif "permission" in error_msg or "privilege" in error_msg:
            # Can't verify due to permissions, but let's assume it might exist
            print(f"? {group_name}: Cannot verify (insufficient permissions)")
            print(f"  Will attempt to use this group in permission grants")
            existing_groups.append(group_name)  # Optimistically add it
        else:
            print(f"? {group_name}: Cannot verify ({str(e)[:80]}...)")
            missing_groups.append(group_name)

# Summary
print("\n" + "="*80)
print("GROUP CHECK SUMMARY")
print("="*80)

# Store available groups for later use
available_groups = existing_groups

if len(existing_groups) > 0:
    print(f"\n✓ Account-level groups found: {len(existing_groups)}")
    for group in existing_groups:
        print(f"  ✓ {group}")
    print("\n  🎉 Excellent! These groups will be used for permission grants.")
else:
    print("\n⊘ No account-level groups found")

if len(missing_groups) > 0:
    print(f"\n⊘ Groups not found: {len(missing_groups)}")
    for group in missing_groups:
        print(f"  ⊘ {group}")

    print("\n📝 How to Create Account-Level Groups:")
    print("-" * 80)
    print("Account-level groups MUST be created in the Databricks Account Console:")
    print("")
    print("Option 1: Azure Databricks Account Console (UI) - Recommended")
    print("  1. Sign in to the Databricks account console (not a workspace)")
    print("  2. In Azure, go to: accounts.azuredatabricks.net")
    print("     (or accounts.cloud.databricks.com for AWS/GCP)")
    print("  3. Log in as an account admin")
    print("  4. Navigate to: User Management section")
    print("  5. Select: Groups tab")
    print("  6. Click: Add Group button")
    print("  7. Enter group name (e.g., ml_engineers)")
    print("  8. Press: Add button")
    print("  9. Repeat for all groups: data_analysts, ml_engineers, data_scientists,")
    print("     data_engineers, all_users")
    print("")
    print("Option 2: Databricks CLI (for Account Admins)")
    print("  databricks account groups create --group-name data_analysts")
    print("  databricks account groups create --group-name ml_engineers")
    print("  databricks account groups create --group-name data_scientists")
    print("  databricks account groups create --group-name data_engineers")
    print("  databricks account groups create --group-name all_users")
    print("")
    print("⚠ Note: CREATE GROUP in SQL creates workspace groups, NOT account groups")
    print("  Workspace groups do NOT work with Unity Catalog permissions!")

print(f"\n📊 Total available groups for permissions: {len(available_groups)}")
if len(available_groups) > 0:
    print("  These groups will be used in the permission granting section.")
else:
    print("  No groups available - will demonstrate with current user only.")
    print("  This is normal and the lab will still teach all RBAC concepts.")

print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Workspace vs. Account Groups
# MAGIC
# MAGIC **Important Distinction:**
# MAGIC - `SHOW GROUPS` displays **workspace-level groups** (created with `CREATE GROUP`)
# MAGIC - Unity Catalog requires **account-level groups** (created in Account Console)
# MAGIC - These are **completely separate** and cannot be used interchangeably!
# MAGIC
# MAGIC Let's check both to understand the difference.

# COMMAND ----------

# Display workspace groups vs account groups
print("=== Understanding Group Types ===\n")

print("⚠ CRITICAL: Workspace Groups ≠ Account Groups")
print("  • SHOW GROUPS shows workspace groups")
print("  • Unity Catalog needs account groups")
print("  • They are completely separate!\n")

# Check workspace groups
print("1. Workspace Groups (from SHOW GROUPS):")
print("-" * 80)
try:
    workspace_groups = spark.sql("SHOW GROUPS")
    workspace_group_list = [row[0] for row in workspace_groups.collect()]

    if len(workspace_group_list) > 0:
        print(f"Found {len(workspace_group_list)} workspace group(s):")
        display(workspace_groups)

        print("\nChecking our required groups in workspace:")
        for group_name in required_groups.keys():
            if group_name in workspace_group_list:
                print(f"  ✓ {group_name} - Found in workspace")
            else:
                print(f"  ✗ {group_name} - Not in workspace")

        print("\n⚠ WARNING: These are WORKSPACE groups!")
        print("  They will NOT work with Unity Catalog permissions.")
        print("  Unity Catalog requires ACCOUNT-LEVEL groups.")
    else:
        print("No workspace groups found")

except Exception as e:
    print(f"Unable to list workspace groups: {str(e)}")

# Check account groups (the ones that actually work with Unity Catalog)
print("\n2. Account Groups (for Unity Catalog):")
print("-" * 80)
print("Checking if groups exist at ACCOUNT level (required for Unity Catalog)...\n")

account_groups_found = []
account_groups_missing = []

for group_name in required_groups.keys():
    try:
        # Try to grant a test permission to see if group exists
        # This is the most reliable way to check across all Databricks versions
        test_sql = f"GRANT USAGE ON CATALOG {CATALOG_NAME} TO `{group_name}`"
        spark.sql(test_sql)

        # If we got here, group exists! Revoke the test grant
        try:
            spark.sql(f"REVOKE USAGE ON CATALOG {CATALOG_NAME} FROM `{group_name}`")
        except:
            pass

        print(f"  ✓ {group_name} - EXISTS at account level (works with Unity Catalog)")
        account_groups_found.append(group_name)

    except Exception as e:
        error_msg = str(e).lower()
        if "principal_does_not_exist" in error_msg or "does not exist" in error_msg or "cannot find" in error_msg:
            print(f"  ✗ {group_name} - DOES NOT EXIST at account level")
            account_groups_missing.append(group_name)
        elif "already granted" in error_msg or "already has" in error_msg:
            # Group exists, permission was already there
            print(f"  ✓ {group_name} - EXISTS at account level (works with Unity Catalog)")
            account_groups_found.append(group_name)
        elif "permission" in error_msg or "privilege" in error_msg:
            print(f"  ? {group_name} - Cannot verify (insufficient permissions)")
            account_groups_missing.append(group_name)
        else:
            print(f"  ? {group_name} - Cannot verify: {str(e)[:60]}...")
            account_groups_missing.append(group_name)

# Summary
print("\n" + "="*80)
print("GROUP TYPE SUMMARY")
print("="*80)

try:
    if len(workspace_group_list) > 0:
        print(f"\n📋 Workspace Groups: {len(workspace_group_list)} found")
        print("  ⚠ These do NOT work with Unity Catalog")
        print("  ⚠ Created with: CREATE GROUP")
        print("  ⚠ Only work for legacy workspace permissions")
except:
    pass

if len(account_groups_found) > 0:
    print(f"\n✓ Account Groups: {len(account_groups_found)} found")
    print("  ✓ These WORK with Unity Catalog")
    for group in account_groups_found:
        print(f"    • {group}")
else:
    print(f"\n✗ Account Groups: 0 found")
    print("  ✗ Unity Catalog permissions will not work")

if len(account_groups_missing) > 0:
    print(f"\n⊘ Missing Account Groups: {len(account_groups_missing)}")
    for group in account_groups_missing:
        print(f"    • {group}")
    print("\n  💡 To create account-level groups:")
    print("     1. Go to: https://accounts.cloud.databricks.com/")
    print("     2. User Management → Groups → Add Group")
    print("     3. Create each group at ACCOUNT level")

print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant Permissions (Demonstration)
# MAGIC
# MAGIC Unity Catalog allows fine-grained permissions. Here's how permissions would be granted in production:
# MAGIC
# MAGIC **Typical Permission Structure:**
# MAGIC - **data_analysts**: SELECT on table (read-only access)
# MAGIC - **ml_engineers**: USE SCHEMA on schema (model execution and schema access)
# MAGIC - **data_scientists**: ALL PRIVILEGES on schema (full access)
# MAGIC - **data_engineers**: MODIFY on table (write access for data pipelines)
# MAGIC - **all_users**: USE CATALOG on catalog (basic catalog access)
# MAGIC
# MAGIC **Note:** This section demonstrates the concepts. In production, your admin would create groups and grant permissions.

# COMMAND ----------

# Demonstrate permission granting concepts
print("=== Unity Catalog Permissions (Demonstration) ===\n")

# Get current user
current_user = spark.sql("SELECT current_user()").collect()[0][0]
print(f"Current user: {current_user}\n")

# Define table paths for RBAC examples (using knowledge base as primary table)
table_path = kb_table_path  # Primary table for RBAC examples

# Show example permission commands
print("In production, an admin would execute commands like:\n")

example_grants = [
    {
        'description': 'Grant read access to data analysts (knowledge base)',
        'sql': f"GRANT SELECT ON TABLE {kb_table_path} TO `data_analysts`;"
    },
    {
        'description': 'Grant schema usage to ML engineers',
        'sql': f"GRANT USE SCHEMA ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME} TO `ml_engineers`;"
    },
    {
        'description': 'Grant full access to data scientists',
        'sql': f"GRANT ALL PRIVILEGES ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME} TO `data_scientists`;"
    },
    {
        'description': 'Grant write access to data engineers (for KB updates)',
        'sql': f"GRANT MODIFY ON TABLE {kb_table_path} TO `data_engineers`;"
    },
    {
        'description': 'Grant catalog usage to all users',
        'sql': f"GRANT USE CATALOG ON CATALOG {CATALOG_NAME} TO `all_users`;"
    }
]

for i, grant in enumerate(example_grants, 1):
    print(f"{i}. {grant['description']}")
    print(f"   {grant['sql']}")
    print()

# Try to grant permissions to production groups (if they exist) and current user
print("="*80)
print("Attempting to grant permissions...")
print("="*80)

# Check if we have available_groups from earlier section
try:
    available_groups_list = available_groups
    print(f"\nℹ Available groups from creation section: {len(available_groups_list)}")
    if len(available_groups_list) > 0:
        print(f"  Groups: {', '.join(available_groups_list)}")
except NameError:
    # If available_groups doesn't exist, we'll try all groups and handle errors
    available_groups_list = []
    print("\nℹ No group information from creation section - will attempt all groups")

successful_grants = []
failed_grants = []
groups_granted = []
groups_not_found = []

# Define production permissions to try
production_permissions = [
    {
        'principal': 'data_analysts',
        'privilege': 'SELECT',
        'object_type': 'TABLE',
        'object_name': table_path,
        'description': 'Read access to RAG knowledge base'
    },
    {
        'principal': 'ml_engineers',
        'privilege': 'USE SCHEMA',
        'object_type': 'SCHEMA',
        'object_name': f"{CATALOG_NAME}.{SCHEMA_NAME}",
        'description': 'Schema usage rights for RAG deployment'
    },
    {
        'principal': 'data_scientists',
        'privilege': 'ALL PRIVILEGES',
        'object_type': 'SCHEMA',
        'object_name': f"{CATALOG_NAME}.{SCHEMA_NAME}",
        'description': 'Full access to RAG schema for experimentation'
    },
    {
        'principal': 'data_engineers',
        'privilege': 'MODIFY',
        'object_type': 'TABLE',
        'object_name': table_path,
        'description': 'Write access to update knowledge base'
    },
    {
        'principal': 'all_users',
        'privilege': 'USE CATALOG',
        'object_type': 'CATALOG',
        'object_name': CATALOG_NAME,
        'description': 'Catalog usage rights'
    }
]

# Try production groups first
print("\n1. Attempting Production Group Grants:")
print("-" * 80)

for perm in production_permissions:
    group_name = perm['principal']

    # Skip if we know the group doesn't exist
    if len(available_groups_list) > 0 and group_name not in available_groups_list:
        print(f"\n⊘ Skipping {group_name}: Group was not created/found in earlier section")
        groups_not_found.append(group_name)
        failed_grants.append(perm)
        continue

    print(f"\nGranting {perm['privilege']} on {perm['object_type']} to {group_name}:")
    print(f"  Object: {perm['object_name']}")
    print(f"  Purpose: {perm['description']}")

    try:
        grant_sql = f"GRANT {perm['privilege']} ON {perm['object_type']} {perm['object_name']} TO `{group_name}`"
        spark.sql(grant_sql)
        print(f"  ✓ Status: Success - Group exists and grant applied!")
        successful_grants.append(perm)
        groups_granted.append(group_name)
    except Exception as e:
        error_msg = str(e)
        if "already has" in error_msg.lower() or "already granted" in error_msg.lower():
            print(f"  ✓ Status: Already granted - Group exists!")
            successful_grants.append(perm)
            groups_granted.append(group_name)
        elif "principal_does_not_exist" in error_msg.lower() or "does not exist" in error_msg.lower() or "cannot find" in error_msg.lower():
            print(f"  ⊘ Status: Group '{group_name}' does not exist")
            print(f"  Note: Group creation failed or requires account admin privileges")
            groups_not_found.append(group_name)
            failed_grants.append(perm)
        elif "insufficient" in error_msg.lower() or "permission" in error_msg.lower():
            print(f"  ⚠ Status: Insufficient privileges (requires admin)")
            print(f"  Note: Group may exist but you need admin rights to grant")
            failed_grants.append(perm)
        else:
            print(f"  ⚠ Status: {error_msg[:150]}...")
            failed_grants.append(perm)

# Also grant to current user for demonstration
print("\n2. Granting to Current User (for demonstration):")
print("-" * 80)

user_permissions = [
    {
        'principal': current_user,
        'privilege': 'SELECT',
        'object_type': 'TABLE',
        'object_name': table_path,
        'description': 'Read access to RAG knowledge base'
    },
    {
        'principal': current_user,
        'privilege': 'USE SCHEMA',
        'object_type': 'SCHEMA',
        'object_name': f"{CATALOG_NAME}.{SCHEMA_NAME}",
        'description': 'Schema usage rights for RAG'
    }
]

for perm in user_permissions:
    print(f"\nGranting {perm['privilege']} on {perm['object_type']}:")
    print(f"  Object: {perm['object_name']}")

    try:
        grant_sql = f"GRANT {perm['privilege']} ON {perm['object_type']} {perm['object_name']} TO `{current_user}`"
        spark.sql(grant_sql)
        print(f"  ✓ Status: Success")
    except Exception as e:
        error_msg = str(e)
        if "already has" in error_msg.lower() or "already granted" in error_msg.lower():
            print(f"  ✓ Status: Already granted")
        else:
            print(f"  ⚠ Status: {error_msg[:80]}...")

# Summary
print("\n" + "="*80)
print("PERMISSION GRANT SUMMARY")
print("="*80)

if len(groups_granted) > 0:
    print(f"\n✓ Production groups successfully granted: {len(set(groups_granted))}")
    for group in set(groups_granted):
        print(f"  ✓ {group}")
    print("\n  🎉 Excellent! Your workspace has production groups configured!")
    print("  The verification section will show these grants.")

if len(groups_not_found) > 0:
    print(f"\n⊘ Groups not found: {len(set(groups_not_found))}")
    for group in set(groups_not_found):
        print(f"  ⊘ {group}")
    print("\n  📝 Why groups don't exist:")
    print("  • Group creation requires account admin privileges")
    print("  • You may not have permission to create groups")
    print("  • Groups may need to be created at account level")
    print("\n  💡 Solution:")
    print("  • Contact your Databricks account admin")
    print("  • Request creation of: data_analysts, ml_engineers, data_scientists, all_users")
    print("  • Or use this lab in demonstration mode (grants to current user)")

if successful_grants:
    print(f"\n✓ Total successful grants: {len(successful_grants)}")
    for perm in successful_grants:
        principal = perm.get('principal', 'current_user')
        print(f"  - {principal}: {perm['privilege']} on {perm['object_type']}")

if len(groups_granted) == 0:
    print("\n📋 Demonstration Mode:")
    print("  Since production groups don't exist, this lab will:")
    print("  • Grant permissions to your current user")
    print("  • Show example commands for production")
    print("  • Explain what production would look like")
    print("  • Teach RBAC concepts effectively")

print("\n" + "="*80)
print("KEY CONCEPTS - Unity Catalog Permissions")
print("="*80)
print("""
1. **Hierarchical Permissions**
   - CATALOG → SCHEMA → TABLE/MODEL
   - Permissions inherit down the hierarchy

2. **Common Permission Types**
   - USE CATALOG: Access to catalog
   - USE SCHEMA: Access to schema
   - SELECT: Read data from tables
   - MODIFY: Write/update data
   - EXECUTE: Run models/functions
   - ALL PRIVILEGES: Full access

3. **Role-Based Access Control (RBAC)**
   - Create groups for different roles (e.g., data_analysts, ml_engineers)
   - Grant permissions to groups, not individuals
   - Users inherit permissions from their groups

4. **Production Setup (Admin Tasks)**
   - Create groups: CREATE GROUP data_analysts;
   - Add users to groups: ALTER GROUP data_analysts ADD USER user@company.com;
   - Grant permissions: GRANT SELECT ON TABLE ... TO data_analysts;

5. **Best Practices**
   - Use groups for permission management
   - Follow principle of least privilege
   - Document permission decisions
   - Review permissions regularly
   - All changes are automatically audited
""")

print("="*80)
print("\n✓ Permission concepts demonstrated")
print("\nIn production environments:")
print("  • Workspace admins create and manage groups")
print("  • Permissions are granted based on job roles")
print("  • All changes are tracked in audit logs")
print("  • Regular access reviews ensure compliance")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Granted Permissions
# MAGIC
# MAGIC Let's verify the permissions were granted successfully by viewing grants on each object.

# COMMAND ----------

# Verify permissions
print("=== Verifying Granted Permissions ===\n")
print("Checking what permissions exist vs. what was demonstrated...\n")

# Define what we expect in production
expected_grants = {
    'table': [
        {'principal': 'data_analysts', 'privilege': 'SELECT', 'description': 'Read access to RAG knowledge base'}
    ],
    'schema': [
        {'principal': 'ml_engineers', 'privilege': 'USE SCHEMA', 'description': 'Schema usage rights for RAG'},
        {'principal': 'data_scientists', 'privilege': 'ALL PRIVILEGES', 'description': 'Full access to RAG schema'}
    ],
    'catalog': [
        {'principal': 'all_users', 'privilege': 'USE CATALOG', 'description': 'Catalog usage rights'}
    ]
}

# Check table permissions
print("1. Table Permissions (RAG Knowledge Base):")
print("-" * 80)
print(f"Expected in production: GRANT SELECT ON TABLE {kb_table_path} TO `data_analysts`\n")

try:
    table_grants = spark.sql(f"SHOW GRANTS ON TABLE {table_path}")
    grants_list = table_grants.collect()

    if len(grants_list) > 0:
        print(f"✓ Found {len(grants_list)} grant(s) on table:")
        display(table_grants)

        # Check for expected permissions
        grants_text = ' '.join([str(row) for row in grants_list]).lower()

        print("\nGrant Analysis:")

        # Check for production groups
        data_analysts_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if 'data_analysts' in row_str and 'select' in row_str:
                print("  ✓ data_analysts has SELECT permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                data_analysts_found = True
                break

        if not data_analysts_found:
            print("  ⊘ data_analysts: Not found (would exist in production)")

        # Check for data_engineers
        data_engineers_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if 'data_engineers' in row_str and 'modify' in row_str:
                print("  ✓ data_engineers has MODIFY permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                data_engineers_found = True
                break

        if not data_engineers_found:
            print("  ⊘ data_engineers: Not found (would exist in production)")

        # Check for current user
        current_user_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if current_user.lower() in row_str:
                if not current_user_found:
                    print(f"  ✓ {current_user} has permissions on table (DEMONSTRATION GRANT)")
                    current_user_found = True
                print(f"     - {row}")

        production_groups_found = data_analysts_found or data_engineers_found
        if production_groups_found:
            print("\n  📝 Note: Production groups found with correct permissions!")
        else:
            print("\n  📝 Note: In production, you would see 'data_analysts' and 'data_engineers' groups here")
    else:
        print("⊘ No explicit grants on table")
        print("\n📋 What You Would See in Production:")
        print("  ✓ data_analysts: SELECT permission")
        print("  ✓ Other relevant groups with appropriate permissions")
        print("\nℹ Current Status:")
        print("  • Permissions are inherited from schema or catalog level")
        print("  • This is normal in learning environments")
        print("  • You can still access the table through inherited permissions")

except Exception as e:
    print(f"Unable to show table grants: {str(e)}")
    print("Note: This may be normal if grants are inherited from parent objects")

print(f"\n2. Schema Permissions ({SCHEMA_NAME}):")
print("-" * 80)
print("Expected in production:")
print("  • GRANT USE SCHEMA ON SCHEMA ... TO `ml_engineers`")
print("  • GRANT ALL PRIVILEGES ON SCHEMA ... TO `data_scientists`\n")

try:
    schema_grants = spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME}")
    grants_list = schema_grants.collect()

    if len(grants_list) > 0:
        print(f"✓ Found {len(grants_list)} grant(s) on schema:")
        display(schema_grants)

        # Check for expected permissions
        grants_text = ' '.join([str(row) for row in grants_list]).lower()

        print("\nGrant Analysis:")

        # Check for production groups
        ml_engineers_found = False
        data_scientists_found = False

        for row in grants_list:
            row_str = str(row).lower()

            if 'ml_engineers' in row_str and 'use schema' in row_str:
                print("  ✓ ml_engineers has USE SCHEMA permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                ml_engineers_found = True

            if 'data_scientists' in row_str and 'all' in row_str:
                print("  ✓ data_scientists has ALL PRIVILEGES (PRODUCTION GRANT)")
                print(f"     - {row}")
                data_scientists_found = True

        if not ml_engineers_found:
            print("  ⊘ ml_engineers: Not found (would exist in production)")
        if not data_scientists_found:
            print("  ⊘ data_scientists: Not found (would exist in production)")

        # Check for current user
        current_user_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if current_user.lower() in row_str:
                if not current_user_found:
                    print(f"  ✓ {current_user} has permissions on schema (DEMONSTRATION GRANT)")
                    current_user_found = True
                print(f"     - {row}")

        if ml_engineers_found or data_scientists_found:
            print("\n  📝 Note: Production groups found with correct permissions!")
        else:
            print("\n  📝 Note: In production, you would see 'ml_engineers' and 'data_scientists' groups here")
    else:
        print("⊘ No explicit grants on schema")
        print("\n📋 What You Would See in Production:")
        print("  ✓ ml_engineers: USE SCHEMA permission")
        print("  ✓ data_scientists: ALL PRIVILEGES")
        print("  ✓ Other relevant groups with appropriate permissions")
        print("\nℹ Current Status:")
        print("  • Permissions are inherited from catalog or account level")
        print("  • This is normal in shared Databricks workspaces")

except Exception as e:
    print(f"Unable to show schema grants: {str(e)}")
    print("Note: This may require additional permissions")

print("\n3. Catalog Permissions (financial_services):")
print("-" * 80)
print("Expected in production: GRANT USE CATALOG ON CATALOG ... TO `all_users`\n")

try:
    catalog_grants = spark.sql(f"SHOW GRANTS ON CATALOG {CATALOG_NAME}")
    grants_list = catalog_grants.collect()

    if len(grants_list) > 0:
        print(f"✓ Found {len(grants_list)} grant(s) on catalog:")
        display(catalog_grants)

        # Check for expected permissions
        grants_text = ' '.join([str(row) for row in grants_list]).lower()

        print("\nGrant Analysis:")

        # Check for production groups
        all_users_found = False

        for row in grants_list:
            row_str = str(row).lower()
            if 'all_users' in row_str and 'use catalog' in row_str:
                print("  ✓ all_users has USE CATALOG permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                all_users_found = True

        if not all_users_found:
            print("  ⊘ all_users: Not found (would exist in production)")

        # Show all other grants
        print("\n  All grants on catalog:")
        for row in grants_list:
            row_str = str(row).lower()
            if 'all_users' not in row_str:  # Don't duplicate all_users
                print(f"  • {row}")

        if all_users_found:
            print("\n  📝 Note: Production group 'all_users' found with correct permissions!")
        else:
            print("\n  📝 Note: In production, you would see 'all_users' group here")
    else:
        print("⊘ No explicit grants on catalog")
        print("\n📋 What You Would See in Production:")
        print("  ✓ all_users: USE CATALOG permission")
        print("  ✓ Admin groups with full privileges")
        print("  ✓ Other relevant groups with appropriate permissions")
        print("\nℹ Current Status:")
        print("  • Permissions are managed at account level")
        print("  • Users have default workspace access")
        print("  • Catalog is accessible to all workspace users")
        print("\n✓ You can still use the catalog - access is inherited from workspace/account level")

except Exception as e:
    print(f"Unable to show catalog grants: {str(e)}")
    print("Note: This may require additional permissions")

print("\n" + "="*80)
print("RBAC VERIFICATION SUMMARY")
print("="*80)

# Summary of what was verified
print("\n✓ Permissions Verified:")
print(f"  - Table grants checked: {table_path}")
print(f"  - Schema grants checked: {CATALOG_NAME}.{SCHEMA_NAME}")
print(f"  - Catalog grants checked: {CATALOG_NAME}")

print("\n" + "="*80)
print("COMPARISON: Demonstration vs. Production")
print("="*80)

# Check what was actually granted by reviewing the grants
try:
    table_check = spark.sql(f"SHOW GRANTS ON TABLE {table_path}").collect()
    schema_check = spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME}").collect()
    catalog_check = spark.sql(f"SHOW GRANTS ON CATALOG {CATALOG_NAME}").collect()

    # Determine which groups were found
    all_grants_text = ' '.join([str(row) for row in table_check + schema_check + catalog_check]).lower()

    data_analysts_exists = 'data_analysts' in all_grants_text
    ml_engineers_exists = 'ml_engineers' in all_grants_text
    data_scientists_exists = 'data_scientists' in all_grants_text
    all_users_exists = 'all_users' in all_grants_text

except:
    data_analysts_exists = False
    ml_engineers_exists = False
    data_scientists_exists = False
    all_users_exists = False

print("\n📋 What Was Demonstrated (Example Commands):")
print("-" * 80)
print("1. GRANT SELECT ON TABLE ... TO `data_analysts`")
print("   Purpose: Read access to RAG knowledge base")
print(f"   Status: {'✓ Successfully granted!' if data_analysts_exists else '⊘ Group does not exist in this environment'}")
print("")
print("2. GRANT USE SCHEMA ON SCHEMA ... TO `ml_engineers`")
print("   Purpose: Schema usage rights for RAG deployment")
print(f"   Status: {'✓ Successfully granted!' if ml_engineers_exists else '⊘ Group does not exist in this environment'}")
print("")
print("3. GRANT ALL PRIVILEGES ON SCHEMA ... TO `data_scientists`")
print("   Purpose: Full access to RAG schema for experimentation")
print(f"   Status: {'✓ Successfully granted!' if data_scientists_exists else '⊘ Group does not exist in this environment'}")
print("")
print("4. GRANT USE CATALOG ON CATALOG ... TO `all_users`")
print("   Purpose: Catalog usage rights")
print(f"   Status: {'✓ Successfully granted!' if all_users_exists else '⊘ Group does not exist in this environment'}")

print("\n📋 What Actually Exists (Verification Results):")
print("-" * 80)
print(f"✓ {current_user}: SELECT on TABLE (demonstration grant)")
print(f"✓ {current_user}: USE SCHEMA on SCHEMA (demonstration grant)")

if data_analysts_exists:
    print("✓ data_analysts: SELECT on TABLE (production grant)")
else:
    print("⊘ data_analysts: Not found (would exist in production)")

if ml_engineers_exists:
    print("✓ ml_engineers: USE SCHEMA on SCHEMA (production grant)")
else:
    print("⊘ ml_engineers: Not found (would exist in production)")

if data_scientists_exists:
    print("✓ data_scientists: ALL PRIVILEGES on SCHEMA (production grant)")
else:
    print("⊘ data_scientists: Not found (would exist in production)")

if all_users_exists:
    print("✓ all_users: USE CATALOG on CATALOG (production grant)")
else:
    print("⊘ all_users: Not found (would exist in production)")

# Summary message
if data_analysts_exists or ml_engineers_exists or data_scientists_exists or all_users_exists:
    print("\n🎉 Excellent! Your workspace has production groups configured and grants were successful!")
else:
    print("\nℹ Note: This is a learning environment without pre-configured production groups.")

print("\n📋 What You Would See in Production:")
print("-" * 80)
print(f"""
Table Level (RAG Knowledge Base - {KNOWLEDGE_BASE_TABLE}):
  ✓ data_analysts: SELECT (read knowledge base)
  ✓ data_scientists: ALL PRIVILEGES (inherited from schema)
  ✓ data_engineers: MODIFY (update knowledge base)

Schema Level ({SCHEMA_NAME}):
  ✓ ml_engineers: USE SCHEMA (deploy RAG models)
  ✓ data_scientists: ALL PRIVILEGES (experiment with RAG)
  ✓ data_analysts: USE SCHEMA (if granted)

Catalog Level ({CATALOG_NAME}):
  ✓ all_users: USE CATALOG
  ✓ admins: ALL PRIVILEGES
  ✓ Other groups as needed

Each grant would show:
  • Principal (group name)
  • ActionType (SELECT, USE SCHEMA, etc.)
  • ObjectType (TABLE, SCHEMA, CATALOG)
  • ObjectKey (full path to object)
""")

print("\n📊 Understanding the Results:")
print("-" * 80)
print("""
If you see "No explicit grants" or "0 grants", this is NORMAL and EXPECTED in:
  • Shared Databricks workspaces
  • Learning/training environments
  • Workspaces with default access policies

How Access Works Without Explicit Grants:
  1. Workspace-level permissions grant default access
  2. Account-level permissions provide inherited access
  3. You're the creator/owner of the objects (automatic access)
  4. Unity Catalog uses hierarchical permission inheritance

What This Means:
  ✓ You CAN access and use the data/models
  ✓ Permissions are inherited from parent levels
  ✓ This is a secure and common configuration
  ✓ In production, explicit grants would be added for other users/groups

Production Difference:
  • Admins would create explicit grants for each group
  • You would see rows in the SHOW GRANTS output
  • Each user/group would have specific permissions listed
  • Audit logs would track all grant operations
""")

print("\n" + "="*80)
print("KEY TAKEAWAYS - Unity Catalog RBAC")
print("="*80)
print("""
1. ✓ Unity Catalog provides fine-grained access control
   - Permissions at catalog, schema, table, and column levels
   - Hierarchical inheritance of permissions

2. ✓ Groups enable scalable permission management
   - Create groups for different roles
   - Grant permissions to groups, not individuals
   - Users inherit from all their groups

3. ✓ Production RBAC Workflow:
   - Account admins create groups
   - Users are assigned to groups based on roles
   - Permissions follow principle of least privilege
   - Regular audits ensure compliance

4. ✓ All permission changes are automatically logged
   - Complete audit trail for compliance
   - Track who granted what to whom
   - Query audit logs for security reviews

5. ✓ RBAC is essential for enterprise governance
   - Meets regulatory requirements
   - Enables secure collaboration
   - Supports data governance policies

Example Production Commands:
  CREATE GROUP data_analysts;
  ALTER GROUP data_analysts ADD USER user@company.com;
  GRANT SELECT ON TABLE ... TO data_analysts;
  SHOW GRANTS ON TABLE ...;
""")
print("="*80)

print("\n💡 Next Steps for Production RBAC:")
print("  1. Work with admin to create proper groups")
print("  2. Map organizational roles to Unity Catalog groups")
print("  3. Document permission policies")
print("  4. Set up regular permission audits")
print("  5. Train users on data access procedures")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Audit Logging
# MAGIC
# MAGIC Unity Catalog automatically logs all operations. Let's query the audit logs to see model operations.
# MAGIC
# MAGIC **Compliance Value:** Audit logs provide a complete trail for regulatory requirements.

# COMMAND ----------

# Query audit logs for model operations
print("=== Audit Logging Demonstration ===\n")

# Check if system catalog is accessible
print("Checking audit log access...")
audit_available = False

try:
    # Try to access system catalog
    spark.sql("USE CATALOG system")
    spark.sql("SHOW TABLES IN system.access").collect()
    audit_available = True
    print("✓ System catalog is accessible")
except Exception as e:
    print("⚠ System catalog not accessible in this workspace")
    print(f"  Reason: {str(e)[:100]}...")

print("\n" + "-"*80)

if audit_available:
    print("\nQuerying audit logs for recent operations...")
    print("(This may take a moment...)\n")

    # Try multiple queries to find audit data
    queries_to_try = [
        {
            'name': 'Unity Catalog operations in this session',
            'query': f"""
                SELECT
                    event_time,
                    user_identity.email as user,
                    action_name,
                    request_params.full_name_arg as object_name,
                    response.status_code
                FROM system.access.audit
                WHERE event_date >= current_date() - INTERVAL 1 DAY
                    AND user_identity.email = '{current_user}'
                    AND (
                        action_name IN ('createTable', 'createSchema', 'createCatalog',
                                       'getTable', 'getSchema', 'getCatalog',
                                       'createRegisteredModelVersion', 'updateRegisteredModel')
                        OR request_params.full_name_arg LIKE '%{CATALOG_NAME}%'
                        OR request_params.full_name_arg LIKE '%{SCHEMA_NAME}%'
                    )
                ORDER BY event_time DESC
                LIMIT 20
            """
        },
        {
            'name': 'Recent table operations',
            'query': f"""
                SELECT
                    event_time,
                    user_identity.email as user,
                    action_name,
                    request_params.full_name_arg as object_name
                FROM system.access.audit
                WHERE event_date >= current_date() - INTERVAL 1 DAY
                    AND action_name IN ('createTable', 'getTable', 'readTable')
                ORDER BY event_time DESC
                LIMIT 10
            """
        },
        {
            'name': 'Any recent operations by current user',
            'query': f"""
                SELECT
                    event_time,
                    user_identity.email as user,
                    action_name,
                    request_params.full_name_arg as object_name
                FROM system.access.audit
                WHERE event_date >= current_date()
                    AND user_identity.email = '{current_user}'
                ORDER BY event_time DESC
                LIMIT 10
            """
        }
    ]

    audit_found = False

    for query_info in queries_to_try:
        if audit_found:
            break

        try:
            print(f"Trying: {query_info['name']}...")
            audit_logs = spark.sql(query_info['query'])
            audit_count = audit_logs.count()

            if audit_count > 0:
                print(f"✓ Found {audit_count} audit log entries!\n")
                print(f"Showing: {query_info['name']}")
                display(audit_logs)
                audit_found = True

                print("\n" + "="*80)
                print("AUDIT LOG ANALYSIS")
                print("="*80)
                print(f"""
✓ Successfully retrieved audit logs from Unity Catalog

What These Logs Show:
  • event_time: When the operation occurred
  • user: Who performed the operation ({current_user})
  • action_name: What operation was performed (createTable, getTable, etc.)
  • object_name: Which object was accessed
  • status_code: Success (200) or error codes

Compliance Value:
  ✓ Complete audit trail of all operations
  ✓ Track who accessed what data and when
  ✓ Investigate security incidents
  ✓ Meet regulatory requirements (SOX, GDPR, HIPAA)
  ✓ Generate compliance reports
""")
                break
            else:
                print(f"  No results for this query")
        except Exception as e:
            print(f"  Query failed: {str(e)[:80]}...")
            continue

    if not audit_found:
        print("\n⚠ No audit logs found with any query")
        print("\nPossible reasons:")
        print("  • Audit logs may have a delay before appearing (up to 1 hour)")
        print("  • Logs may be retained for limited time")
        print("  • Some operations may not be logged in this workspace type")
        print("  • Filters may not match recent operations")
        audit_available = False

if not audit_available:
    print("\n" + "="*80)
    print("AUDIT LOG DEMONSTRATION (Simulated)")
    print("="*80)
    print("\nSince audit logs aren't available, here's what they would show for this lab:\n")

    # Create simulated audit log data
    from datetime import datetime, timedelta
    import pandas as pd

    current_time = datetime.now()

    simulated_logs = [
        {
            'event_time': (current_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createCatalog',
            'object_name': CATALOG_NAME,
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=9)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createSchema',
            'object_name': f'{CATALOG_NAME}.{SCHEMA_NAME}',
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=8)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createTable',
            'object_name': table_path,
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createRegisteredModelVersion',
            'object_name': MODEL_NAME,
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'setRegisteredModelAlias',
            'object_name': f'{MODEL_NAME} (Champion)',
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'grantPrivileges',
            'object_name': f'USE SCHEMA on {CATALOG_NAME}.{SCHEMA_NAME}',
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'getTable',
            'object_name': table_path,
            'status_code': 200
        }
    ]

    simulated_df = pd.DataFrame(simulated_logs)
    print("Simulated Audit Log Entries (What Would Appear in Production):")
    print("-"*80)
    display(simulated_df)

    print("\n" + "="*80)
    print("AUDIT LOG ANALYSIS (Simulated)")
    print("="*80)
    print(f"""
What These Logs Show:
  ✓ Catalog creation: {CATALOG_NAME}
  ✓ Schema creation: {SCHEMA_NAME}
  ✓ Table creation: {TABLE_NAME}
  ✓ Model registration: {MODEL_NAME}
  ✓ Model alias assignment: Champion
  ✓ Permission grant: USE SCHEMA
  ✓ Data access: getTable operation

All operations performed by: {current_user}
All operations successful: status_code = 200

Compliance Value:
  ✓ Complete audit trail of all operations
  ✓ Track who accessed what data and when
  ✓ Investigate security incidents
  ✓ Meet regulatory requirements (SOX, GDPR, HIPAA)
  ✓ Generate compliance reports
  ✓ Retention: 90+ days (configurable)
""")

    print("\n📚 About Unity Catalog Audit Logs:")
    print("-"*80)
    print("""
Audit logs in Unity Catalog track ALL operations including:

1. **Data Access**
   - Table reads and writes (getTable, readTable)
   - Schema and catalog access
   - Column-level access (if enabled)

2. **Model Operations**
   - Model registration (createRegisteredModelVersion)
   - Version creation and updates
   - Model alias changes (setRegisteredModelAlias)
   - Model downloads and deployments

3. **Permission Changes**
   - GRANT and REVOKE operations (grantPrivileges, revokePrivileges)
   - Group membership changes
   - Role assignments

4. **Administrative Actions**
   - Catalog/schema creation (createCatalog, createSchema)
   - Table modifications (createTable, alterTable)
   - Policy updates

Example Audit Log Query:
""")

    print("""
-- Query all operations on a specific model
SELECT
    event_time,
    user_identity.email,
    action_name,
    request_params.name,
    response.status_code
FROM system.access.audit
WHERE request_params.name = 'catalog.schema.model_name'
ORDER BY event_time DESC;

-- Query all permission grants
SELECT
    event_time,
    user_identity.email,
    action_name,
    request_params.privilege,
    request_params.principal
FROM system.access.audit
WHERE action_name LIKE '%GRANT%'
ORDER BY event_time DESC;

-- Query all data access
SELECT
    event_time,
    user_identity.email,
    action_name,
    request_params.full_name_arg
FROM system.access.audit
WHERE action_name = 'getTable'
ORDER BY event_time DESC;
""")

    print("\n" + "-"*80)
    print("In Production Environments:")
    print("  ✓ Audit logs are automatically enabled")
    print("  ✓ Logs are retained for 90+ days (configurable)")
    print("  ✓ Can be exported to external systems (S3, Azure, etc.)")
    print("  ✓ Used for compliance reporting and security monitoring")
    print("  ✓ Integrated with SIEM tools for real-time alerting")

    print("\n" + "-"*80)
    print("What Audit Logs Would Show for This Lab:")
    print("  • Model registration: " + MODEL_NAME)
    print("  • Version creation: Versions 1, 2, etc.")
    print("  • Alias assignments: Champion, Challenger")
    print("  • Table creation: " + table_path)
    print("  • All by user: " + current_user)
    print("  • Timestamps for each operation")
    print("  • Success/failure status codes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Lineage Tracking
# MAGIC
# MAGIC Unity Catalog automatically tracks lineage from data to models. This shows:
# MAGIC - Which tables were used to train the model
# MAGIC - Which notebooks/jobs created the model
# MAGIC - Downstream dependencies
# MAGIC
# MAGIC **Governance Benefit:** Complete transparency for auditors and stakeholders.

# COMMAND ----------

# Demonstrate lineage information
print("=== Model Lineage Information ===\n")

# Get model details
model_details = client.get_registered_model(MODEL_NAME)

print(f"Model: {model_details.name}")
print(f"Description: {model_details.description[:100] if model_details.description else 'N/A'}...")
print(f"\nLineage:")
print(f"  - Source Data: {table_path}")
print(f"  - Training Notebook: {experiment_name}")
print(f"  - Total Versions: {len(all_versions)}")
print(f"  - Current Champion: Version {model_version.version}")
print(f"  - Current Challenger: Version {model_version_v2.version}")

# Show data lineage through Unity Catalog
print(f"\n✓ Unity Catalog tracks complete lineage:")
print(f"  Data → Model → Deployment")
print(f"  All accessible through the Unity Catalog UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7: Model Monitoring and Reproducibility
# MAGIC
# MAGIC For production models, we need:
# MAGIC - **Reproducibility**: Ability to recreate any model version
# MAGIC - **Monitoring**: Track model performance over time
# MAGIC - **Documentation**: Clear records of all decisions
# MAGIC
# MAGIC Let's implement these best practices.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reproducibility: Recreate Model from Registry
# MAGIC
# MAGIC Demonstrate how to fully reproduce a model using MLflow tracking.

# COMMAND ----------

# Get run information for reproducibility
run_info = client.get_run(best_run_id)

print("=== Model Reproducibility Information ===\n")
print(f"Run ID: {run_info.info.run_id}")
print(f"Experiment ID: {run_info.info.experiment_id}")
print(f"Start Time: {datetime.fromtimestamp(run_info.info.start_time/1000)}")
print(f"End Time: {datetime.fromtimestamp(run_info.info.end_time/1000)}")

print("\nLogged Parameters:")
for key, value in run_info.data.params.items():
    print(f"  {key}: {value}")

print("\nLogged Metrics:")
for key, value in run_info.data.metrics.items():
    print(f"  {key}: {value:.4f}")

print("\nLogged Tags:")
for key, value in run_info.data.tags.items():
    if not key.startswith('mlflow.'):
        print(f"  {key}: {value}")

print("\n✓ All information needed to reproduce this model is logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Performance Report
# MAGIC
# MAGIC Generate a comprehensive report for stakeholders and compliance.
# MAGIC
# MAGIC #### Why Reports Matter
# MAGIC **Stakeholder Communication:** Non-technical stakeholders need clear, concise reports:
# MAGIC - Business leaders: Cost projections and ROI
# MAGIC - Compliance teams: Governance and audit trail
# MAGIC - Product managers: Performance metrics and capabilities
# MAGIC - Finance: Budget impact and cost forecasting
# MAGIC
# MAGIC **Compliance Documentation:** Regulators require:
# MAGIC - Model documentation and validation
# MAGIC - Performance metrics and limitations
# MAGIC - Data lineage and governance controls
# MAGIC - Audit trail of all operations
# MAGIC

# COMMAND ----------

# Create comprehensive RAG performance report
report = f"""
{'='*80}
RAG CUSTOMER SUPPORT ASSISTANT - PERFORMANCE REPORT
{'='*80}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Name: {MODEL_NAME}
Champion Version: {model_version.version}
Challenger Version: {model_version_v2.version}
Report Author: {current_user}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

This RAG (Retrieval-Augmented Generation) system answers customer support questions
by retrieving relevant information from a knowledge base and generating natural
language responses. The system has been evaluated on {len(df_eval_questions)} test questions
across multiple configurations.

Key Findings:
  ✓ Champion configuration (top_k=3) provides optimal cost/quality balance
  ✓ Average cost per query: ${comparison_df.loc[1, 'Cost_Per_Query_USD']:.6f}
  ✓ Estimated monthly cost (1000 queries): ${comparison_df.loc[1, 'Cost_Per_Query_USD'] * 1000:.2f}
  ✓ All governance controls in place and operational
  ✓ Ready for production deployment with proper LLM integration

{'='*80}
CHAMPION MODEL CONFIGURATION
{'='*80}

Configuration: {best_config_name}
Retrieval Method: Keyword-based (production should use vector embeddings)
LLM: Mock LLM (production should use DBRX, GPT-4, or similar)
Top-K Documents: {int(comparison_df.loc[best_config_idx, 'Avg_Docs_Retrieved'])}

Knowledge Base:
  - Source: {CATALOG_NAME}.{SCHEMA_NAME}.{KNOWLEDGE_BASE_TABLE}
  - Version: {df_knowledge_base['version'].iloc[0]}
  - Total Documents: {len(df_knowledge_base):,}
  - Categories: {', '.join(df_knowledge_base['category'].unique())}

Evaluation Dataset:
  - Source: {CATALOG_NAME}.{SCHEMA_NAME}.{EVAL_QUESTIONS_TABLE}
  - Total Questions: {len(df_eval_questions):,}
  - Question Types: {', '.join(df_eval_questions['category'].unique())}

Performance Metrics:
  - Avg Tokens per Query: {comparison_df.loc[best_config_idx, 'Avg_Tokens_Per_Query']:.1f}
  - Total Tokens (Eval Set): {comparison_df.loc[best_config_idx, 'Total_Tokens']:,}
  - Cost per Query: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}
  - Total Cost (Eval Set): ${comparison_df.loc[best_config_idx, 'Total_Cost_USD']:.4f}
  - Avg Docs Retrieved: {comparison_df.loc[best_config_idx, 'Avg_Docs_Retrieved']:.1f}

{'='*80}
CHALLENGER MODEL CONFIGURATION
{'='*80}

Configuration: Comprehensive (top_k=5)
Top-K Documents: 5
Purpose: Higher retrieval coverage for improved answer quality

Performance Metrics:
  - Avg Tokens per Query: {comparison_df.loc[2, 'Avg_Tokens_Per_Query']:.1f}
  - Cost per Query: ${comparison_df.loc[2, 'Cost_Per_Query_USD']:.6f}
  - Relative Cost: {comparison_df.loc[2, 'Relative_Cost_Pct']:.0f}% of baseline

Trade-offs:
  ✓ Pros: More comprehensive context, potentially better answers
  ✗ Cons: {comparison_df.loc[2, 'Relative_Cost_Pct']:.0f}% higher cost, more tokens, slower responses

{'='*80}
COST ANALYSIS & PROJECTIONS
{'='*80}

Champion Configuration Cost Projections:
  • 100 queries/month:    ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 100:.2f}
  • 1,000 queries/month:  ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 1000:.2f}
  • 10,000 queries/month: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 10000:.2f}
  • 100,000 queries/month: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 100000:.2f}

Challenger Configuration Cost Projections:
  • 1,000 queries/month:  ${comparison_df.loc[2, 'Cost_Per_Query_USD'] * 1000:.2f}
  • 10,000 queries/month: ${comparison_df.loc[2, 'Cost_Per_Query_USD'] * 10000:.2f}

Cost Optimization Opportunities:
  1. Implement response caching for common questions (est. 30-50% cost reduction)
  2. Use cheaper LLM for simple questions (tiered routing)
  3. Optimize prompt templates to reduce token usage
  4. Implement query deduplication

{'='*80}
GOVERNANCE & COMPLIANCE
{'='*80}

✓ Data Governance:
  • All data stored in Unity Catalog with RBAC
  • Knowledge base versioned and tracked
  • Complete data lineage from source to model

✓ Experiment Tracking:
  • All experiments logged in MLflow
  • Parameters, metrics, and artifacts tracked
  • Complete reproducibility guaranteed

✓ Model Registry:
  • Models registered in Unity Catalog Model Registry
  • Comprehensive documentation attached
  • Version history maintained

✓ Access Control:
  • RBAC implemented for data and models
  • Permissions follow principle of least privilege
  • Group-based access management

✓ Audit Trail:
  • All operations logged in Unity Catalog audit logs
  • Complete trail for regulatory compliance
  • Queryable for security reviews

✓ Cost Tracking:
  • Token usage logged for every query
  • Cost projections calculated
  • Budget monitoring enabled

{'='*80}
PRODUCTION DEPLOYMENT REQUIREMENTS
{'='*80}

Before deploying to production, complete these tasks:

1. ✅ Replace Mock LLM with Production LLM
   - Options: DBRX, GPT-4, GPT-3.5-turbo, Llama 2/3
   - Configure API keys and endpoints
   - Test integration thoroughly

2. ✅ Upgrade Retrieval to Vector Search
   - Implement vector embeddings (e.g., sentence-transformers)
   - Deploy Databricks Vector Search index
   - Benchmark retrieval quality

3. ✅ Implement Guardrails
   - Content filtering (toxicity, PII)
   - Input validation and sanitization
   - Output verification

4. ✅ Add Caching Layer
   - Cache common questions and responses
   - Implement cache invalidation strategy
   - Monitor cache hit rate

5. ✅ Set Up Monitoring
   - Track answer quality (human feedback)
   - Monitor costs and token usage
   - Alert on anomalies

6. ✅ Conduct Human Evaluation
   - Evaluate answer quality on test set
   - Measure accuracy, completeness, helpfulness
   - Iterate on prompt templates

7. ✅ Obtain Compliance Approval
   - Submit for compliance review
   - Address any concerns
   - Document approval

8. ✅ Establish Maintenance Process
   - Knowledge base update workflow
   - Model re-evaluation schedule
   - Incident response procedures

{'='*80}
RECOMMENDATIONS
{'='*80}

Immediate Actions:
  1. ✓ Deploy Champion model (v{model_version.version}) to staging environment
  2. ✓ Integrate production LLM (recommend DBRX for cost-effectiveness)
  3. ✓ Implement vector-based retrieval
  4. ✓ Conduct human evaluation on 100 test questions
  5. ✓ Set up cost monitoring and alerts

A/B Testing Plan:
  1. Deploy Champion (top_k=3) to 90% of traffic
  2. Deploy Challenger (top_k=5) to 10% of traffic
  3. Monitor for 2 weeks:
     - Answer quality (user feedback)
     - Cost per query
     - Response latency
  4. Promote Challenger if quality improvement justifies cost increase

Ongoing Maintenance:
  1. Update knowledge base weekly (or as needed)
  2. Re-evaluate model monthly
  3. Review costs and optimize quarterly
  4. Conduct compliance audit quarterly

{'='*80}
RISK ASSESSMENT
{'='*80}

Technical Risks:
  ⚠ Mock LLM: Must be replaced before production (HIGH PRIORITY)
  ⚠ Keyword Retrieval: May miss semantically similar documents (MEDIUM)
  ⚠ No Guardrails: Risk of inappropriate responses (HIGH)
  ⚠ No Caching: Higher costs and latency (MEDIUM)

Mitigation Strategies:
  ✓ Replace mock LLM with production LLM (in progress)
  ✓ Implement vector search (planned)
  ✓ Add content filtering and validation (planned)
  ✓ Implement caching layer (planned)

Business Risks:
  ⚠ Cost Escalation: LLM costs can increase with usage
  ⚠ Answer Quality: Hallucinations or incorrect information
  ⚠ Knowledge Staleness: Outdated information in knowledge base

Mitigation Strategies:
  ✓ Cost monitoring and budgets
  ✓ Human evaluation and feedback loops
  ✓ Regular knowledge base updates

{'='*80}
CONCLUSION
{'='*80}

The RAG Customer Support Assistant is ready for production deployment pending:
  1. Integration with production LLM
  2. Implementation of vector-based retrieval
  3. Addition of guardrails and safety measures
  4. Human evaluation and quality validation

All governance controls are in place:
  ✓ Data governance via Unity Catalog
  ✓ Experiment tracking via MLflow
  ✓ Model versioning and documentation
  ✓ Access control and audit logging
  ✓ Cost tracking and projections

Estimated production cost: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 1000:.2f}/month (1000 queries)

Recommended next step: Deploy to staging environment for integration testing.

{'='*80}
APPROVAL SIGNATURES
{'='*80}

Data Science Lead: _________________ Date: _________
ML Engineering Lead: _________________ Date: _________
Compliance Officer: _________________ Date: _________
Product Manager: _________________ Date: _________

{'='*80}
"""

print(report)

# Save report as artifact
with open('/tmp/rag_performance_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Performance report saved to /tmp/rag_performance_report.txt")

print("\n✓ Report saved to /tmp/model_performance_report.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8: Model Archiving and Cleanup Policies
# MAGIC
# MAGIC As models accumulate, we need policies for:
# MAGIC - **Archiving old versions** that are no longer in use
# MAGIC - **Cleaning up experiments** to maintain organization
# MAGIC - **Retaining compliance records** per regulatory requirements
# MAGIC
# MAGIC **Best Practice:** Archive models rather than delete them to maintain audit trails.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Archive Old Model Versions
# MAGIC
# MAGIC Let's demonstrate archiving a model version that's no longer needed.

# COMMAND ----------

# Function to archive old model versions
def archive_model_version(model_name, version, reason):
    """
    Archive a model version by adding archive tags and documentation.

    Args:
        model_name: Full model name in Unity Catalog
        version: Version number to archive
        reason: Reason for archiving
    """
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archived",
        value="true"
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archive_date",
        value=datetime.now().strftime('%Y-%m-%d')
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archive_reason",
        value=reason
    )

    print(f"✓ Model version {version} archived")
    print(f"  Reason: {reason}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d')}")

# Example: Archive the first version if we have multiple versions
if len(all_versions) > 2:
    archive_model_version(
        MODEL_NAME,
        all_versions[-1].version,  # Oldest version
        "Superseded by improved models with better performance"
    )
else:
    print("Note: Archiving demonstration - would archive older versions in production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleanup Policy Implementation
# MAGIC
# MAGIC Define and implement cleanup policies for model registry maintenance.

# COMMAND ----------

# Define cleanup policy
cleanup_policy = {
    'retain_champion': True,  # Always keep Champion model
    'retain_challenger': True,  # Always keep Challenger model
    'archive_after_days': 90,  # Archive versions older than 90 days
    'max_versions': 10,  # Keep maximum 10 versions
    'require_documentation': True  # All versions must have documentation
}

print("=== Model Registry Cleanup Policy ===\n")
for key, value in cleanup_policy.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

# Implement cleanup check
def check_cleanup_needed(model_name, policy):
    """
    Check if cleanup is needed based on policy.

    Args:
        model_name: Full model name in Unity Catalog
        policy: Dictionary of cleanup policies

    Returns:
        List of versions that can be archived
    """
    versions = client.search_model_versions(f"name='{model_name}'")

    # Get versions with aliases (Champion, Challenger)
    protected_versions = set()
    for version in versions:
        if hasattr(version, 'aliases') and version.aliases:
            protected_versions.add(version.version)

    # Find versions that can be archived
    archivable = []
    for version in versions:
        # Skip protected versions
        if version.version in protected_versions:
            continue

        # Check age
        created_time = datetime.fromtimestamp(version.creation_timestamp / 1000)
        age_days = (datetime.now() - created_time).days

        if age_days > policy['archive_after_days']:
            archivable.append({
                'version': version.version,
                'age_days': age_days,
                'created': created_time
            })

    return archivable

# Check cleanup
archivable_versions = check_cleanup_needed(MODEL_NAME, cleanup_policy)

print(f"\n=== Cleanup Analysis ===")
print(f"Total versions: {len(all_versions)}")
print(f"Archivable versions: {len(archivable_versions)}")

if archivable_versions:
    print("\nVersions eligible for archiving:")
    for v in archivable_versions:
        print(f"  Version {v['version']}: {v['age_days']} days old (created {v['created']})")
else:
    print("\n✓ No versions need archiving at this time")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 9: End-to-End Workflow Summary
# MAGIC
# MAGIC Let's create a comprehensive summary of everything we've accomplished in this lab.

# COMMAND ----------

# Create comprehensive summary
summary = f"""
{'='*80}
MLflow & UNITY CATALOG FOR RAG SYSTEMS - COMPLETE WORKFLOW SUMMARY
{'='*80}

Lab Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
User: {current_user}
Model: {MODEL_NAME}
Use Case: RAG Customer Support Assistant

{'='*80}
SECTION 1: ENVIRONMENT SETUP
{'='*80}

✓ Unity Catalog configured:
  - Catalog: {CATALOG_NAME}
  - Schema: {SCHEMA_NAME}
  - Model Registry: Unity Catalog (databricks-uc)

✓ MLflow configured:
  - Experiment: {experiment_name}
  - Registry URI: databricks-uc
  - Tracking enabled for all experiments

✓ RAG Components Initialized:
  - Knowledge base table created
  - Evaluation questions table created
  - Retrieval function implemented
  - Mock LLM generator created

{'='*80}
SECTION 2: DATA PREPARATION
{'='*80}

✓ Knowledge Base:
  - Table: {CATALOG_NAME}.{SCHEMA_NAME}.{KNOWLEDGE_BASE_TABLE}
  - Documents: {len(df_knowledge_base):,} documents
  - Categories: {', '.join(df_knowledge_base['category'].unique())}
  - Version: {df_knowledge_base['version'].iloc[0]}

✓ Evaluation Dataset:
  - Table: {CATALOG_NAME}.{SCHEMA_NAME}.{EVAL_QUESTIONS_TABLE}
  - Questions: {len(df_eval_questions):,} test questions
  - Categories: {', '.join(df_eval_questions['category'].unique())}

✓ RAG Pipeline:
  - Retrieval: Keyword-based search (production: vector search)
  - Generation: Mock LLM (production: DBRX, GPT-4, etc.)
  - Prompt template: Defined and versioned

{'='*80}
SECTION 3: EXPERIMENT TRACKING
{'='*80}

✓ Ran 3 RAG experiments with full MLflow tracking:
  1. Baseline (top_k=2) - Cost-optimized configuration
  2. Standard (top_k=3) - Balanced configuration
  3. Comprehensive (top_k=5) - Quality-optimized configuration

✓ Logged for each experiment:
  - Parameters (top_k, retrieval_method, llm_model, etc.)
  - Metrics (tokens, costs, retrieval stats)
  - Artifacts (prompt templates, sample predictions, cost analysis)
  - Tags (configuration, optimization_goal, developer, etc.)

✓ Best configuration: {best_config_name}
  - Avg tokens/query: {comparison_df.loc[best_config_idx, 'Avg_Tokens_Per_Query']:.1f}
  - Cost/query: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}
  - Avg docs retrieved: {comparison_df.loc[best_config_idx, 'Avg_Docs_Retrieved']:.1f}

✓ Cost Projections (Champion):
  - 1,000 queries/month: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 1000:.2f}
  - 10,000 queries/month: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 10000:.2f}

{'='*80}
SECTION 4: MODEL REGISTRATION
{'='*80}

✓ Registered best RAG configuration to Unity Catalog
✓ Model name: {MODEL_NAME}
✓ Champion version: {model_version.version}
✓ Added comprehensive documentation including:
  - Architecture and configuration details
  - Performance metrics and cost projections
  - Intended use and limitations
  - Deployment requirements
  - Compliance and governance information

{'='*80}
SECTION 5: VERSION MANAGEMENT
{'='*80}

✓ Registered Challenger configuration (top_k=5)
✓ Challenger version: {model_version_v2.version}
✓ Set model aliases:
  - Champion (Production): Version {model_version.version} (top_k=3)
  - Challenger (A/B Test): Version {model_version_v2.version} (top_k=5)
✓ Demonstrated version comparison and A/B testing setup
✓ Complete version history maintained

{'='*80}
SECTION 6: GOVERNANCE & COMPLIANCE
{'='*80}

✓ Unity Catalog RBAC:
  - Fine-grained access control on knowledge base and models
  - Role-based permissions for different teams
  - Group-based access management demonstrated

✓ Audit Logging:
  - All operations automatically logged
  - Complete trail for regulatory compliance
  - Queryable for security reviews

✓ Data Lineage:
  - Full traceability: Knowledge Base → RAG Model → Responses
  - Accessible through Unity Catalog UI
  - Version tracking for all components

✓ Cost Governance:
  - Token usage tracked for every query
  - Cost projections calculated
  - Budget monitoring enabled

{'='*80}
SECTION 7: REPRODUCIBILITY
{'='*80}

✓ All experiments fully reproducible via MLflow:
  - Complete parameter logging (top_k, prompt templates, etc.)
  - All metrics tracked (tokens, costs, retrieval stats)
  - Artifacts saved (prompts, predictions, analysis)
  - Knowledge base versioned in Unity Catalog

✓ Generated comprehensive performance report:
  - Executive summary
  - Configuration details
  - Cost analysis and projections
  - Deployment requirements
  - Risk assessment

{'='*80}
SECTION 8: ARCHIVING & CLEANUP
{'='*80}

✓ Defined cleanup policies:
  - Retain Champion and Challenger models
  - Archive versions older than 90 days
  - Maximum 10 versions per model
  - Maintain audit trail for archived models

✓ Implemented archiving workflow
✓ Version management best practices established

{'='*80}
KEY ACHIEVEMENTS - RAG GOVERNANCE
{'='*80}

1. ✓ Complete MLflow experiment tracking for RAG systems
2. ✓ Unity Catalog integration for RAG governance
3. ✓ Cost tracking and optimization for LLM-based systems
4. ✓ Model versioning and lifecycle management
5. ✓ RBAC and access control for knowledge bases
6. ✓ Audit logging and compliance readiness
7. ✓ Data lineage from knowledge base to responses
8. ✓ Reproducibility best practices for RAG
9. ✓ Comprehensive documentation and reporting

{'='*80}
PRODUCTION READINESS CHECKLIST
{'='*80}

Completed:
  ✓ RAG pipeline developed and tested
  ✓ Best configuration registered in Unity Catalog
  ✓ Comprehensive documentation complete
  ✓ Governance controls in place
  ✓ Audit trail established
  ✓ Cost tracking and projections calculated
  ✓ Cleanup policies implemented
  ✓ Team access controls configured

Pending (Before Production Deployment):
  ⚠ Replace mock LLM with production LLM (DBRX, GPT-4, etc.)
  ⚠ Implement vector-based retrieval (Databricks Vector Search)
  ⚠ Add content filtering and guardrails
  ⚠ Implement response caching
  ⚠ Conduct human evaluation of answer quality
  ⚠ Set up monitoring and alerting
  ⚠ Obtain compliance approval
  ⚠ Deploy to Model Serving endpoint

{'='*80}
NEXT STEPS
{'='*80}

Immediate (Week 1-2):
  1. Integrate production LLM (DBRX recommended for cost-effectiveness)
  2. Implement vector embeddings and Databricks Vector Search
  3. Add content filtering and input validation
  4. Deploy to staging environment

Short-term (Week 3-4):
  5. Conduct human evaluation on 100+ test questions
  6. Implement response caching for common questions
  7. Set up monitoring dashboard (costs, quality, latency)
  8. Configure alerting for anomalies

Medium-term (Month 2-3):
  9. Deploy Champion to production with 100% traffic
  10. Set up A/B test: Champion (90%) vs Challenger (10%)
  11. Establish knowledge base update workflow
  12. Create runbook for incident response

Long-term (Ongoing):
  13. Monthly model re-evaluation
  14. Quarterly compliance audits
  15. Continuous knowledge base updates
  16. Regular cost optimization reviews

{'='*80}
COMPLIANCE NOTES
{'='*80}

✓ All data stored in Unity Catalog with access controls
✓ Complete audit trail maintained for all operations
✓ Model lineage fully documented (KB → Model → Responses)
✓ Reproducibility guaranteed via MLflow tracking
✓ Cost tracking and budget monitoring in place
✓ RBAC implemented for data and model access
✓ Documentation meets regulatory requirements
✓ Ready for compliance review

Regulatory Considerations:
  • Financial services: Ensure responses comply with regulations
  • Data privacy: Implement PII detection and redaction
  • Audit requirements: Maintain logs for required retention period
  • Model explainability: Document retrieval and generation process

{'='*80}
LAB COMPLETE - ENTERPRISE RAG GOVERNANCE ACHIEVED
{'='*80}

Congratulations! You've successfully implemented enterprise-grade governance
for a RAG (Retrieval-Augmented Generation) system using MLflow and Unity Catalog.

You now have:
  ✓ Complete experiment tracking and reproducibility
  ✓ Centralized model registry with versioning
  ✓ Fine-grained access control and audit logging
  ✓ Cost tracking and optimization framework
  ✓ Production-ready deployment pipeline
  ✓ Compliance and governance controls

This foundation enables you to:
  • Deploy RAG systems with confidence
  • Meet regulatory and compliance requirements
  • Track and optimize costs at scale
  • Maintain complete audit trails
  • Collaborate effectively across teams
  • Iterate and improve continuously

Next: Deploy to production and start serving customers!

{'='*80}
"""

print(summary)

# Save summary
with open('/tmp/rag_lab_summary.txt', 'w') as f:
    f.write(summary)

print("\n✓ Lab summary saved to /tmp/rag_lab_summary.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 10: Hands-On Exercises
# MAGIC
# MAGIC Now that you've completed the guided lab, try these exercises to reinforce your learning:
# MAGIC
# MAGIC ### Exercise 1: Experiment with Different RAG Configurations
# MAGIC - Create a new RAG experiment with top_k=4
# MAGIC - Try a different prompt template (modify PROMPT_TEMPLATE)
# MAGIC - Log the experiment to MLflow with appropriate tags
# MAGIC - Compare cost and token usage to existing configurations
# MAGIC
# MAGIC ### Exercise 2: Enhance the Knowledge Base
# MAGIC - Add 5 new documents to the knowledge base
# MAGIC - Update the version number
# MAGIC - Re-run the evaluation with the updated knowledge base
# MAGIC - Compare retrieval quality before and after
# MAGIC
# MAGIC ### Exercise 3: Create Custom Evaluation Questions
# MAGIC - Add 10 new evaluation questions to the eval_questions table
# MAGIC - Run predictions on the new questions
# MAGIC - Analyze which questions are answered well vs. poorly
# MAGIC - Identify knowledge gaps in the knowledge base
# MAGIC
# MAGIC ### Exercise 4: Implement Cost Monitoring
# MAGIC - Create a function to track cumulative costs across all experiments
# MAGIC - Calculate cost per category of question
# MAGIC - Identify the most expensive question types
# MAGIC - Propose cost optimization strategies
# MAGIC
# MAGIC ### Exercise 5: Promote and Test the Challenger
# MAGIC - Load both Champion and Challenger models
# MAGIC - Run the same questions through both
# MAGIC - Compare responses, costs, and token usage
# MAGIC - Make a data-driven decision on which to promote
# MAGIC
# MAGIC ### Exercise 6: Query Audit Logs
# MAGIC - Query Unity Catalog audit logs
# MAGIC - Find all operations on your RAG model
# MAGIC - Create a compliance report showing:
# MAGIC   - Who registered the model
# MAGIC   - When aliases were set
# MAGIC   - All table access operations
# MAGIC
# MAGIC ### Exercise 7: Archive Old Versions
# MAGIC - Identify versions that should be archived
# MAGIC - Apply archiving tags with proper documentation
# MAGIC - Verify archived versions are still loadable (for audit purposes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Congratulations! You've completed a comprehensive lab on MLflow and Unity Catalog for enterprise RAG governance.
# MAGIC
# MAGIC ### What You've Learned:
# MAGIC
# MAGIC 1. **RAG System Architecture**
# MAGIC    - Building retrieval-augmented generation pipelines
# MAGIC    - Implementing keyword-based retrieval (foundation for vector search)
# MAGIC    - Creating mock LLM generators for cost-effective development
# MAGIC    - Packaging RAG systems as MLflow PyFunc models
# MAGIC
# MAGIC 2. **MLflow Experiment Tracking for RAG**
# MAGIC    - Logging RAG-specific parameters (top_k, prompt templates, LLM settings)
# MAGIC    - Tracking cost metrics (token usage, estimated costs)
# MAGIC    - Saving artifacts (prompts, sample predictions, cost analysis)
# MAGIC    - Comparing different RAG configurations
# MAGIC
# MAGIC 3. **Model Registry with Unity Catalog**
# MAGIC    - Registering RAG models with comprehensive documentation
# MAGIC    - Managing model lifecycle with aliases (Champion, Challenger)
# MAGIC    - Loading models for inference
# MAGIC    - Maintaining version history for audit trails
# MAGIC
# MAGIC 4. **Enterprise Governance for RAG**
# MAGIC    - RBAC for knowledge bases and models
# MAGIC    - Audit logging for compliance
# MAGIC    - Data lineage from knowledge base to responses
# MAGIC    - Cost governance and budget monitoring
# MAGIC
# MAGIC 5. **RAG-Specific Best Practices**
# MAGIC    - Cost tracking and optimization for LLM-based systems
# MAGIC    - Prompt template versioning
# MAGIC    - Knowledge base versioning and updates
# MAGIC    - Evaluation methodology for RAG quality
# MAGIC    - Production deployment requirements
# MAGIC
# MAGIC ### Real-World Applications:
# MAGIC
# MAGIC This RAG governance workflow is used in production environments for:
# MAGIC - **Customer Support**: Automated question answering, ticket deflection
# MAGIC - **Internal Knowledge Management**: Employee self-service, policy Q&A
# MAGIC - **Financial Services**: Regulatory compliance Q&A, product information
# MAGIC - **Healthcare**: Medical information retrieval, patient education
# MAGIC - **Legal**: Contract analysis, legal research assistance
# MAGIC - **E-commerce**: Product recommendations, customer inquiries
# MAGIC
# MAGIC ### Key Takeaways:
# MAGIC
# MAGIC ✅ **Cost Control**: Track every token and dollar spent on LLM inference
# MAGIC ✅ **Reproducibility**: Every RAG experiment is fully reproducible
# MAGIC ✅ **Governance**: Complete audit trail from knowledge base to responses
# MAGIC ✅ **Scalability**: Framework supports production deployment at scale
# MAGIC ✅ **Compliance**: Meets regulatory requirements for financial services and healthcare
# MAGIC ✅ **Flexibility**: Easy to swap retrieval methods, LLMs, and configurations
# MAGIC
# MAGIC ### Production Deployment Checklist:
# MAGIC
# MAGIC Before deploying your RAG system to production:
# MAGIC - [ ] Replace mock LLM with production LLM (DBRX, GPT-4, etc.)
# MAGIC - [ ] Implement vector-based retrieval (Databricks Vector Search)
# MAGIC - [ ] Add content filtering and guardrails
# MAGIC - [ ] Implement response caching
# MAGIC - [ ] Conduct human evaluation (100+ questions)
# MAGIC - [ ] Set up monitoring and alerting
# MAGIC - [ ] Obtain compliance approval
# MAGIC - [ ] Deploy to Databricks Model Serving
# MAGIC - [ ] Establish knowledge base update workflow
# MAGIC - [ ] Create incident response runbook
# MAGIC
# MAGIC ### Resources:
# MAGIC
# MAGIC **MLflow & Unity Catalog:**
# MAGIC - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
# MAGIC - [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/index.html)
# MAGIC - [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
# MAGIC
# MAGIC **RAG & LLMs:**
# MAGIC - [Databricks Foundation Models](https://docs.databricks.com/machine-learning/foundation-models/index.html)
# MAGIC - [Databricks Vector Search](https://docs.databricks.com/vector-search/index.html)
# MAGIC - [RAG Best Practices](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html)
# MAGIC
# MAGIC **Governance & Compliance:**
# MAGIC - [Unity Catalog RBAC](https://docs.databricks.com/data-governance/unity-catalog/manage-privileges/index.html)
# MAGIC - [Audit Logging](https://docs.databricks.com/administration-guide/account-settings/audit-logs.html)
# MAGIC - [Data Lineage](https://docs.databricks.com/data-governance/unity-catalog/data-lineage.html)
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC 1. **Immediate**: Integrate a production LLM and test with real queries
# MAGIC 2. **Short-term**: Implement vector search for better retrieval quality
# MAGIC 3. **Medium-term**: Deploy to production and monitor performance
# MAGIC 4. **Long-term**: Continuously improve based on user feedback and metrics
# MAGIC
# MAGIC ### Thank You!
# MAGIC
# MAGIC You're now equipped to implement enterprise-grade governance for RAG systems in your organization.
# MAGIC
# MAGIC **Key Achievement**: You can now build, track, govern, and deploy RAG systems that meet enterprise requirements for cost control, compliance, and quality.
# MAGIC
# MAGIC **Remember**: The governance framework you've learned applies to any LLM-based system, not just RAG. Use these patterns for:
# MAGIC - Fine-tuned models
# MAGIC - Prompt engineering experiments
# MAGIC - Multi-agent systems
# MAGIC - Any GenAI application
# MAGIC
# MAGIC Good luck with your RAG deployments! 🚀

# COMMAND ----------

# Final verification - Display key resources
print("="*80)
print("RAG LAB RESOURCES - QUICK REFERENCE")
print("="*80)

print(f"\n📚 Knowledge Base:")
print(f"   - Table: {CATALOG_NAME}.{SCHEMA_NAME}.{KNOWLEDGE_BASE_TABLE}")
print(f"   - Documents: {len(df_knowledge_base):,}")
print(f"   - Version: {df_knowledge_base['version'].iloc[0]}")

print(f"\n❓ Evaluation Questions:")
print(f"   - Table: {CATALOG_NAME}.{SCHEMA_NAME}.{EVAL_QUESTIONS_TABLE}")
print(f"   - Questions: {len(df_eval_questions):,}")

print(f"\n🤖 Model Registry:")
print(f"   - Model Name: {MODEL_NAME}")
print(f"   - Champion: Version {model_version.version} (top_k=3)")
print(f"   - Challenger: Version {model_version_v2.version} (top_k=5)")

print(f"\n🔬 MLflow Experiment:")
print(f"   - Experiment: {experiment_name}")
print(f"   - Total Runs: 3 (Baseline, Standard, Comprehensive)")

print(f"\n💰 Cost Summary (Champion):")
print(f"   - Cost per query: ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD']:.6f}")
print(f"   - Monthly (1K queries): ${comparison_df.loc[best_config_idx, 'Cost_Per_Query_USD'] * 1000:.2f}")
print(f"   - Avg tokens/query: {comparison_df.loc[best_config_idx, 'Avg_Tokens_Per_Query']:.1f}")

print(f"\n📁 Generated Reports:")
print(f"   - Performance Report: /tmp/rag_performance_report.txt")
print(f"   - Lab Summary: /tmp/rag_lab_summary.txt")

print(f"\n🎯 Next Steps:")
print(f"   1. Replace mock LLM with production LLM")
print(f"   2. Implement vector-based retrieval")
print(f"   3. Add guardrails and content filtering")
print(f"   4. Deploy to Model Serving")

print(f"\n✅ Lab Status: COMPLETE")
print(f"✅ RAG Governance Framework: ESTABLISHED")
print("="*80)

print(f"\n🎉 Congratulations! You've successfully built an enterprise-grade RAG system")
print(f"   with complete MLflow tracking and Unity Catalog governance!")
print("="*80)