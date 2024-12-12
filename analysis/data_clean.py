import pandas as pd
import numpy as np

def extract_assistant_response(conversation):
    """
    Extracts the assistant's response from a conversation list.
    
    Parameters:
    - conversation (list): A list of dictionaries representing the conversation turns.
    
    Returns:
    - str: The content of the assistant's response. Returns an empty string if not found.
    """
    if isinstance(conversation, list):
        for turn in conversation:
            if turn.get("role") == "assistant":
                return turn.get("content", "")
    return ""

def clean_data(
    conversation_file: str,
    topic_hardness_file: str,
    prompt_emb_file: str,
    model_a_emb_file: str,
    model_b_emb_file: str,
    upper_quantile: float = 0.99
) -> pd.DataFrame:
    """
    Load and clean the main dataset and auxiliary data.
    
    Steps:
    - Load conversation data.
    - Extract prompt and model responses.
    - Compute prompt and response lengths.
    - Cap outliers in length columns.
    - Load and clean topic/hardness data.
    - Merge cleaned auxiliary data if needed.
    
    Parameters:
    - conversation_file (str): Path to the main conversation dataset (JSONL GZIP).
    - topic_hardness_file (str): Path to the topic and hardness JSONL GZIP dataset.
    - prompt_emb_file (str): Path to prompt embeddings (Numpy file).
    - model_a_emb_file (str): Path to model A response embeddings (Numpy file).
    - model_b_emb_file (str): Path to model B response embeddings (Numpy file).
    - upper_quantile (float): Upper quantile threshold for capping outliers.
    
    Returns:
    - pd.DataFrame: A cleaned DataFrame ready for modeling or further analysis.
    """

    # Load main conversation data
    df = pd.read_json(
        conversation_file,
        lines=True,
        compression="gzip"
    )

    # Extract the user prompt
    df["prompt"] = df["conversation_a"].str[0].str["content"]

    # Extract assistant responses
    df["model_a_response"] = df["conversation_a"].apply(extract_assistant_response)
    df["model_b_response"] = df["conversation_b"].apply(extract_assistant_response)

    # Compute lengths
    df["prompt_length"] = df["prompt"].str.len()
    df["model_a_response_length"] = df["model_a_response"].str.len()
    df["model_b_response_length"] = df["model_b_response"].str.len()

    # Handle outliers in lengths by capping at the specified quantile
    cap_a = df["model_a_response_length"].quantile(upper_quantile)
    cap_b = df["model_b_response_length"].quantile(upper_quantile)
    cap_prompt = df["prompt_length"].quantile(upper_quantile)

    df["model_a_response_length"] = df["model_a_response_length"].clip(upper=cap_a)
    df["model_b_response_length"] = df["model_b_response_length"].clip(upper=cap_b)
    df["prompt_length"] = df["prompt_length"].clip(upper=cap_prompt)

    # Load embeddings (if needed for cleaning steps down the line)
    # Not actively cleaning embeddings here, just loading.
    prompt_embeddings = np.load(prompt_emb_file)
    response_a_embeddings = np.load(model_a_emb_file)
    response_b_embeddings = np.load(model_b_emb_file)

    # Load topic and hardness data
    topic_and_hardness = pd.read_json(
        topic_hardness_file,
        lines=True,
        compression="gzip"
    )

    # Drop rows with missing critical columns
    required_columns = [
        'topic_modeling_1', 'score_reason_1', 'score_value_1',
        'topic_modeling_2', 'score_reason_2', 'score_value_2',
        'topic_modeling_3', 'score_reason_3', 'score_value_3'
    ]
    topic_and_hardness = topic_and_hardness.dropna(subset=required_columns)

    # Example cleaning for score_value columns:
    score_value_columns = ['score_value_1', 'score_value_2', 'score_value_3']

    def clean_score_value(x):
        """
        Cleans the 'score_value' entries by flattening nested lists and extracting numeric values.
        """
        if isinstance(x, list):
            flat_list = []
            for item in x:
                if isinstance(item, list):
                    flat_list.extend(item)
                else:
                    flat_list.append(item)
            numeric_scores = [score for score in flat_list if isinstance(score, (int, float))]
            return np.mean(numeric_scores) if numeric_scores else np.nan
        elif isinstance(x, (int, float)):
            return float(x)
        else:
            return np.nan

    for col in score_value_columns:
        topic_and_hardness[col] = topic_and_hardness[col].apply(clean_score_value)

    # Impute missing values in hardness scores with median if any remain
    for col in score_value_columns:
        if topic_and_hardness[col].isnull().sum() > 0:
            median_val = topic_and_hardness[col].median()
            topic_and_hardness[col].fillna(median_val, inplace=True)

    # If merging topic/hardness data with main df is desired
    # Assuming 'question_id' is the join key
    df_clean = df.merge(topic_and_hardness, on="question_id", how="left")

    # Drop rows that still have missing critical fields after merge (if needed)
    df_clean = df_clean.dropna(subset=['winner', 'prompt', 'model_a_response', 'model_b_response'])

    # Return the cleaned DataFrame
    return df_clean

# Example usage:
"""
cleaned_df = clean_data(
    conversation_file="./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-conversations.jsonl.gz",
    topic_hardness_file="./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-gpt3-scores.jsonl.gz",
    prompt_emb_file="./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-prompts-embeddings.npy",
    model_a_emb_file="./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-model_a_response-embeddings.npy",
    model_b_emb_file="./data100-shared-readwrite/fa24_grad_project_data/nlp-chatbot-analysis_data/training-set/chatbot-arena-model_b_response-embeddings.npy"
)
 """

