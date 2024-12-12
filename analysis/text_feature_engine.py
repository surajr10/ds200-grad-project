# text_feature_engine.py

import pandas as pd
import numpy as np
import re
import textstat
from lexical_diversity import lex_div as ld
from collections import Counter

class TextFeatureEngine:
    def __init__(self):
        self.politeness_indicators = ["please", "thank you", "could you", "kindly", "would you mind"]
        self.negative_words = ["no", "not", "never", "can't", "won't", "don't", "hate", "bad"]
        
    def process_dataframe(self, df, conversation_a_col, conversation_b_col):
        """
        Main method to process the dataframe and add all features
        """
        df_processed = df.copy()
        
        # Extract prompt and responses from conversation structure
        df_processed["prompt"] = df_processed[conversation_a_col].str[0].str["content"]
        df_processed["response_a"] = df_processed[conversation_a_col].str[1].str["content"]
        df_processed["response_b"] = df_processed[conversation_b_col].str[1].str["content"]
        
        # Continue with feature processing...
        df_processed = self._add_tokenization_features(df_processed)
        df_processed = self._add_readability_scores(df_processed)
        df_processed = self._add_lexical_features(df_processed)
        df_processed = self._add_similarity_features(df_processed)
        df_processed = self._add_additional_features(df_processed)
        
        return df_processed
    
    def _extract_content(self, series):
        """Extract content from conversation format if needed"""
        try:
            return series.str["content"] if isinstance(series.iloc[0], dict) else series
        except:
            return series
    
    def _tokenize(self, text):
        """Tokenize text using regex"""
        pattern = r"\b\w+\b"
        return re.findall(pattern, str(text))
    
    def _add_tokenization_features(self, df):
        """Add tokenization-related features"""
        for col in ['prompt', 'response_a', 'response_b']:
            df[f'{col}_tokens'] = df[col].apply(self._tokenize)
            df[f'{col}_token_length'] = df[f'{col}_tokens'].apply(len)
        return df
    
    def _add_readability_scores(self, df):
        """Add readability scores"""
        for response in ['response_a', 'response_b']:
            df[f'{response}_flesch_kincaid'] = df[response].apply(textstat.flesch_kincaid_grade)
            df[f'{response}_gunning_fog'] = df[response].apply(textstat.gunning_fog)
            df[f'{response}_smog'] = df[response].apply(textstat.smog_index)
        return df
    
    def _add_lexical_features(self, df):
        """Add lexical richness features"""
        for response in ['response_a', 'response_b']:
            df[f'{response}_ttr'] = df[response].apply(self._type_token_ratio)
            df[f'{response}_lexical_diversity'] = df[response].apply(self._lexical_diversity)
            df[f'{response}_avg_syllable_count'] = df[response].apply(self._average_syllable_count)
            df[f'{response}_complex_word_count'] = df[response].apply(self._complex_word_count)
        return df
    
    def _add_similarity_features(self, df):
        """Add similarity features"""
        # Jaccard similarities between responses
        df['response_jaccard_similarity'] = df.apply(
            lambda row: self._jaccard_similarity(row['response_a_tokens'], row['response_b_tokens']), axis=1)
        
        # Jaccard similarities between prompt and each response
        df['prompt_a_jaccard_similarity'] = df.apply(
            lambda row: self._jaccard_similarity(row['prompt_tokens'], row['response_a_tokens']), axis=1)
        
        df['prompt_b_jaccard_similarity'] = df.apply(
            lambda row: self._jaccard_similarity(row['prompt_tokens'], row['response_b_tokens']), axis=1)
        
        # Keyword overlap
        df['prompt_a_keyword_overlap'] = df.apply(
            lambda row: self._keyword_overlap_count(row['prompt_tokens'], row['response_a_tokens']), axis=1)
        df['prompt_b_keyword_overlap'] = df.apply(
            lambda row: self._keyword_overlap_count(row['prompt_tokens'], row['response_b_tokens']), axis=1)
        df['response_ab_keyword_overlap'] = df.apply(
            lambda row: self._keyword_overlap_count(row['response_a_tokens'], row['response_b_tokens']), axis=1)
        
        # Add unique word counts
        df['prompt_unique_words'] = df['prompt_tokens'].apply(lambda x: len(set(x)))
        df['response_a_unique_words'] = df['response_a_tokens'].apply(lambda x: len(set(x)))
        df['response_b_unique_words'] = df['response_b_tokens'].apply(lambda x: len(set(x)))
        
        return df
    
    def _add_additional_features(self, df):
        """Add additional features"""
        # Question mark presence
        df['is_question'] = df['prompt'].apply(lambda x: 1 if '?' in str(x) else 0)
        
        # Politeness and negativity
        for col in ['prompt', 'response_a', 'response_b']:
            df[f'{col}_contains_politeness'] = df[f'{col}_tokens'].apply(self._contains_politeness)
            df[f'{col}_contains_negative'] = df[f'{col}_tokens'].apply(self._contains_negative)
        return df
    
    # Helper methods
    def _type_token_ratio(self, text):
        tokens = str(text).split()
        return len(set(tokens)) / len(tokens) if tokens else 0
    
    def _lexical_diversity(self, text):
        tokens = str(text).split()
        return ld.ttr(tokens) if tokens else 0
    
    def _average_syllable_count(self, text):
        tokens = str(text).split()
        if not tokens:
            return 0
        return sum(textstat.syllable_count(word) for word in tokens) / len(tokens)
    
    def _complex_word_count(self, text):
        tokens = str(text).split()
        return sum(1 for word in tokens if textstat.syllable_count(word) > 3)
    
    def _jaccard_similarity(self, tokens_a, tokens_b):
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / len(union) if union else 0
    
    def _keyword_overlap_count(self, tokens_a, tokens_b):
        return len(set(tokens_a).intersection(set(tokens_b)))
    
    def _contains_politeness(self, tokens):
        return 1 if any(word in tokens for word in self.politeness_indicators) else 0
    
    def _contains_negative(self, tokens):
        return 1 if any(word in tokens for word in self.negative_words) else 0