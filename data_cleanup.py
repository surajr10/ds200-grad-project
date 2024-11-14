"""
EDA of Chatbot Arena Dataset 

This script explores the Chatbot Arena dataset to understand its structure, analyze distributions, and handle data cleaning.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# -------------------------------
# 1. Load Datasets
# -------------------------------

# Main Conversation Data
df = pd.read_json(
    "./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-conversations.jsonl.gz",
    lines=True,
    compression="gzip"
)

# Auxiliary Embeddings
prompt_embeddings = np.load("./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-prompts-embeddings.npy")
response_a_embeddings = np.load("./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-model_a_response-embeddings.npy")
response_b_embeddings = np.load("./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-model_b_response-embeddings.npy")

# Topic Modeling and Hardness Scores
topic_and_hardness = pd.read_json(
    "./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-gpt3-scores.jsonl.gz",
    lines=True,
    compression="gzip"
)

# -------------------------------
# 2. Explore Conversation Data
# -------------------------------

print("Dataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# Extract prompts
df["prompt"] = df["conversation_a"].apply(lambda x: x[0]["content"] if isinstance(x, list) and len(x) > 0 else "")
df["prompt_length"] = df["prompt"].str.len()

# Plot Prompt Length Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["prompt_length"], kde=True)
plt.title("Distribution of Prompt Lengths")
plt.xlabel("Length (characters)")
plt.ylabel("Count")
plt.show()

# Extract Model Responses
def extract_response(conversation):
    return next((turn["content"] for turn in conversation if turn.get("role") == "assistant"), "")

df["model_a_response"] = df["conversation_a"].apply(extract_response)
df["model_b_response"] = df["conversation_b"].apply(extract_response)

# Calculate Response Lengths
df["model_a_length"] = df["model_a_response"].str.len()
df["model_b_length"] = df["model_b_response"].str.len()

# Plot Response Length Distributions
plt.figure(figsize=(14, 6))
sns.histplot(df["model_a_length"], bins=50, kde=True, color='skyblue', label='Model A')
sns.histplot(df["model_b_length"], bins=50, kde=True, color='salmon', label='Model B')
plt.title("Distribution of Response Lengths")
plt.xlabel("Length (characters)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Handle Outliers by Capping at 99th Percentile
for col in ["model_a_length", "model_b_length", "prompt_length"]:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

# Plot Capped Prompt Length Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["prompt_length"], bins=50, kde=True, color='green')
plt.title("Capped Prompt Length Distribution")
plt.xlabel("Length (characters)")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 3. Embedding Analysis
# -------------------------------

# Sample Embeddings and Compute Similarity
sample_size = 1000
embeddings_sample = prompt_embeddings[:sample_size]
similarity = np.dot(embeddings_sample, embeddings_sample.T)

# Choose Source Prompt
source_idx = 23
print(f"Source Prompt (Index {source_idx}): {df.iloc[source_idx].prompt}")

# Find Top 5 Similar Prompts
top_k = 5
similar_indices = np.argsort(similarity[source_idx])[-top_k-1:-1][::-1]  # Exclude the source itself
similar_prompts = df.iloc[similar_indices].prompt.tolist()
print("Top 5 Similar Prompts:")
for i, prompt in enumerate(similar_prompts, 1):
    print(f"{i}. {prompt}")

# -------------------------------
# 4. Clean Topic and Hardness Data
# -------------------------------

# Drop rows with missing essential fields
essential_cols = [
    'topic_modeling_1', 'score_reason_1', 'score_value_1',
    'topic_modeling_2', 'score_reason_2', 'score_value_2',
    'topic_modeling_3', 'score_reason_3', 'score_value_3'
]
topic_clean = topic_and_hardness.dropna(subset=essential_cols).copy()

# Function to flatten and average scores
def clean_score(x):
    if isinstance(x, list):
        flat = [item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])]
        numeric = [s for s in flat if isinstance(s, (int, float))]
        return np.mean(numeric) if numeric else np.nan
    return float(x) if isinstance(x, (int, float)) else np.nan

# Clean score_value columns
score_cols = ['score_value_1', 'score_value_2', 'score_value_3']
for col in score_cols:
    topic_clean[col] = topic_clean[col].apply(clean_score)

# Impute any remaining NaNs with median
for col in score_cols:
    if topic_clean[col].isnull().sum() > 0:
        median = topic_clean[col].median()
        topic_clean[col].fillna(median, inplace=True)

# -------------------------------
# Summary
# -------------------------------

print("\nCleaned Topic and Hardness Data Info:")
print(topic_clean.info())

# Save cleaned data
df.to_csv("cleaned_chatbot_arena_conversations.csv", index=False)
topic_clean.to_csv("cleaned_topic_hardness_scores.csv", index=False)

print("\nEDA completed successfully.")
