import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
import umap
import hdbscan
import os

class EmbeddingFeatureEngine:
    def __init__(self):
        """Initialize the EmbeddingFeatureEngine"""
        self.scaler = StandardScaler()

    def process_dataframe(self, train_prompt_embeddings_path, train_response_a_embeddings_path, train_response_b_embeddings_path, test_prompt_embeddings_path, test_response_a_embeddings_path, test_response_b_embeddings_path):
        """
        Main method to load and process embeddings from file paths

        Parameters:
        -----------
        prompt_embeddings_path : str
            Path to the .npy file containing prompt embeddings
        response_a_embeddings_path : str
            Path to the .npy file containing response A embeddings
        response_b_embeddings_path : str
            Path to the .npy file containing response B embeddings

        Returns:
        --------
        dict of pandas DataFrames
            Processed dataframes for each clustering method with embedding-based features
        """
        # Validate file paths
        for path in [train_prompt_embeddings_path, train_response_a_embeddings_path, train_response_b_embeddings_path, test_prompt_embeddings_path, test_response_a_embeddings_path, test_response_b_embeddings_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Embedding file not found: {path}")
            if not path.endswith('.npy'):
                raise ValueError(f"File must be a .npy file: {path}")

        # Load embeddings
        try:
            train_prompt_embeddings = np.load(train_prompt_embeddings_path)
            train_response_a_embeddings = np.load(train_response_a_embeddings_path)
            train_response_b_embeddings = np.load(train_response_b_embeddings_path)
            test_prompt_embeddings = np.load(test_prompt_embeddings_path)
            test_response_a_embeddings = np.load(test_response_a_embeddings_path)
            test_response_b_embeddings = np.load(test_response_b_embeddings_path)

            print(f"Loaded embeddings with shapes:")
            print(f"Train Prompt embeddings: {train_prompt_embeddings.shape}")
            print(f"Train Response A embeddings: {train_response_a_embeddings.shape}")
            print(f"Train Response B embeddings: {train_response_b_embeddings.shape}")
            print(f"Test Prompt embeddings: {test_prompt_embeddings.shape}")
            print(f"Test Response A embeddings: {test_response_a_embeddings.shape}")
            print(f"Test Response B embeddings: {test_response_b_embeddings.shape}")


        except Exception as e:
            raise Exception(f"Error loading embeddings: {str(e)}")
        
        cutoff = train_prompt_embeddings.shape[0]

        # Concatenate train and test embeddings
        prompt_embeddings = np.concatenate([train_prompt_embeddings, test_prompt_embeddings], axis=0)
        response_a_embeddings = np.concatenate([train_response_a_embeddings, test_response_a_embeddings], axis=0)
        response_b_embeddings = np.concatenate([train_response_b_embeddings, test_response_b_embeddings], axis=0)

        # Process embeddings
        features = {}

        # Add similarity features
        similarity_features = self._add_similarity_features(
            prompt_embeddings, 
            response_a_embeddings, 
            response_b_embeddings
        )
        features.update(similarity_features)

        # Add distance features
        distance_features = self._add_distance_features(
            prompt_embeddings, 
            response_a_embeddings, 
            response_b_embeddings
        )
        features.update(distance_features)

        # Add statistical features
        statistical_features = self._add_statistical_features(
            prompt_embeddings, 
            response_a_embeddings, 
            response_b_embeddings
        )
        features.update(statistical_features)

        # Add clustering features and split into separate DataFrames
        clustering_features = self._add_clustering_features(
            prompt_embeddings, 
            response_a_embeddings, 
            response_b_embeddings
        )

        general_features = pd.DataFrame(features)
        clustering_dataframes = {'train': {}, 'test': {}}
        for method, method_features in clustering_features.items():
            method_features_df = pd.concat([general_features, pd.DataFrame(method_features)], axis=1)
            clustering_dataframes['train'][method] = method_features_df[:cutoff]
            clustering_dataframes['test'][method] = method_features_df[cutoff:]
            # clustering_dataframes[method] = pd.concat([general_features, pd.DataFrame(method_features)], axis=1)

        return clustering_dataframes

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return 1 - cosine(vec1, vec2)

    def _euclidean_distance(self, vec1, vec2):
        """Calculate Euclidean distance between two vectors"""
        return euclidean(vec1, vec2)

    def _embedding_difference(self, vec1, vec2):
        """Calculate element-wise difference between embeddings"""
        return vec1 - vec2

    def _add_similarity_features(self, prompt_emb, response_a_emb, response_b_emb):
        """Add similarity-based features"""
        features = {}

        # Cosine similarities
        features['cosine_sim_prompt_response_a'] = [
            self._cosine_similarity(p, ra) 
            for p, ra in zip(prompt_emb, response_a_emb)
        ]
        features['cosine_sim_prompt_response_b'] = [
            self._cosine_similarity(p, rb) 
            for p, rb in zip(prompt_emb, response_b_emb)
        ]
        features['cosine_sim_response_a_b'] = [
            self._cosine_similarity(ra, rb) 
            for ra, rb in zip(response_a_emb, response_b_emb)
        ]

        return features

    def _add_distance_features(self, prompt_emb, response_a_emb, response_b_emb):
        """Add distance-based features"""
        features = {}

        # Euclidean distances
        features['euclidean_dist_prompt_response_a'] = [
            self._euclidean_distance(p, ra) 
            for p, ra in zip(prompt_emb, response_a_emb)
        ]
        features['euclidean_dist_prompt_response_b'] = [
            self._euclidean_distance(p, rb) 
            for p, rb in zip(prompt_emb, response_b_emb)
        ]
        features['euclidean_dist_response_a_b'] = [
            self._euclidean_distance(ra, rb) 
            for ra, rb in zip(response_a_emb, response_b_emb)
        ]

        return features

    def _add_statistical_features(self, prompt_emb, response_a_emb, response_b_emb):
        """Add statistical features from embeddings"""
        features = {}

        # Variances
        features['variance_prompt'] = [np.var(p) for p in prompt_emb]
        features['variance_response_a'] = [np.var(ra) for ra in response_a_emb]
        features['variance_response_b'] = [np.var(rb) for rb in response_b_emb]

        # Means
        features['mean_prompt'] = [np.mean(p) for p in prompt_emb]
        features['mean_response_a'] = [np.mean(ra) for ra in response_a_emb]
        features['mean_response_b'] = [np.mean(rb) for rb in response_b_emb]

        # Standard deviations
        features['std_prompt'] = [np.std(p) for p in prompt_emb]
        features['std_response_a'] = [np.std(ra) for ra in response_a_emb]
        features['std_response_b'] = [np.std(rb) for rb in response_b_emb]

        return features
    
    def _ohe_cluster_labels(self, df, columns):
        for column in columns:
            ohe = OneHotEncoder(drop='first')
            ohe.fit(df[[column]])
            encoded_column = ohe.transform(df[[column]]).toarray()
            encoded_column_df = pd.DataFrame(encoded_column, columns=ohe.get_feature_names_out())
            df = pd.concat([df, encoded_column_df], axis=1).drop(columns=column)

        return df

    def _add_clustering_features(self, prompt_emb, response_a_emb, response_b_emb):
        """Add clustering-based features from embeddings"""
        clustering_features = {}

        # UMAP reduction
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.1)
        prompt_umap = umap_reducer.fit_transform(prompt_emb)
        response_a_umap = umap_reducer.fit_transform(response_a_emb)
        response_b_umap = umap_reducer.fit_transform(response_b_emb)

        for method, clusterer in {
            'hdbscan': hdbscan.HDBSCAN(min_cluster_size=20, min_samples=2),
            'kmeans': KMeans(n_clusters=10, random_state=42),
            'agglo': AgglomerativeClustering(n_clusters=10)
        }.items():
            method_features = {}

            # Fit clusterers
            method_features['prompt_clusters'] = clusterer.fit_predict(prompt_umap)
            method_features['response_a_clusters'] = clusterer.fit_predict(response_a_umap)
            method_features['response_b_clusters'] = clusterer.fit_predict(response_b_umap)

            # Cluster Transition Type
            method_features['response_a_same_cluster_as_prompt'] = (
                method_features['response_a_clusters'] == method_features['prompt_clusters']
            ).astype(int)
            method_features['response_b_same_cluster_as_prompt'] = (
                method_features['response_b_clusters'] == method_features['prompt_clusters']
            ).astype(int)

            # Cluster Transition Distance (only for algorithms with centroids)
            if hasattr(clusterer, 'cluster_centers_'):
                centroids = clusterer.cluster_centers_

                def safe_euclidean(c1, c2):
                    if c1 == -1 or c2 == -1:  # Skip noise points
                        return float('nan')
                    return euclidean(centroids[c1], centroids[c2])

                method_features['centroid_distance_prompt_response_a'] = [
                    safe_euclidean(pc, ra) for pc, ra in zip(
                        method_features['prompt_clusters'], method_features['response_a_clusters']
                    )
                ]
                method_features['centroid_distance_prompt_response_b'] = [
                    safe_euclidean(pc, rb) for pc, rb in zip(
                        method_features['prompt_clusters'], method_features['response_b_clusters']
                    )
                ]

            # Cluster Independence
            method_features['response_a_different_from_response_b'] = (
                method_features['response_a_clusters'] != method_features['response_b_clusters']
            ).astype(int)

            # One-hot encode cluster labels
            method_features = self._ohe_cluster_labels(pd.DataFrame(method_features), columns=[
                'prompt_clusters', 'response_a_clusters', 'response_b_clusters'
            ])

            clustering_features[method] = method_features

        return clustering_features

