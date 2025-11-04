from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json

def perform_spatial_pca(df, n_components):
    
    feature_groups = {}
    
    for col in df.columns:
        try:
            parts = col.split(' ', 1)
            if len(parts) == 2:
                feature_type = parts[1]
                if feature_type not in feature_groups:
                    feature_groups[feature_type] = []
                feature_groups[feature_type].append(col)
        except Exception:
            continue

    n_components_map = {}

    if isinstance(n_components, int):
        n_components_map = {ftype: n_components for ftype in feature_groups.keys()}
    elif isinstance(n_components, dict):
        n_components_map = n_components
    else:
        raise ValueError("n_components must be an integer or a dictionary.")
            
    transformed_data = {}
    component_info = {}
    
    for feature_type, original_cols in feature_groups.items():
        if feature_type not in n_components_map:
            print(f"Skipping feature group '{feature_type}': Not in map and map was dictionary input.")
            continue
            
        N_channels = len(original_cols)
        k_components = n_components_map[feature_type]

        if k_components >= N_channels:
             k_components = N_channels - 1
             print(f"Warning: Components ({n_components_map[feature_type]}) >= Channels ({N_channels}) for '{feature_type}'. Setting k={k_components}.")

        X = df[original_cols]
        
        pca = PCA(n_components=k_components)
        pca.fit(X)
        X_pca = pca.transform(X)

        pca_features = [f"PCA_{feature_type}_PC{i+1}" for i in range(k_components)]
        
        for i, name in enumerate(pca_features):
            transformed_data[name] = X_pca[:, i]
        
        loadings = []
        channel_names = [col.split(' ', 1)[0] for col in original_cols]

        for i in range(k_components):
            
            pc_loading_data = {
                "Explained_Variance_Ratio": pca.explained_variance_ratio_[i],
                "Explained_Variance": pca.explained_variance_[i]
            }
            
            for j, channel in enumerate(channel_names):
                pc_loading_data[channel] = pca.components_[i, j]
                
            loadings.append(pc_loading_data)
            
        component_info[feature_type] = {
            "mean": pca.mean_.tolist(),
            "components": pca.components_.tolist(),
            "original_channels": channel_names,
            "loadings_metadata": loadings
        }

    return pd.DataFrame(transformed_data, index=df.index), component_info

def apply_spatial_pca_transformation(df, transform_data_json_path):
    with open(transform_data_json_path, 'r') as f:
        transform_data = json.load(f)
        
    transformed_data = {}
    
    for feature_type, params in transform_data.items():
        original_channels = params["original_channels"]
        original_cols = [f"{ch} {feature_type}" for ch in original_channels]
        
        try:
            X_new = df[original_cols]
        except KeyError as e:
            print(f"Error: Missing feature column in new data for type '{feature_type}'. {e}")
            continue

        mean_vector = np.array(params["mean"])
        components_matrix = np.array(params["components"])
        
        X_centered = X_new.values - mean_vector
        X_transformed = np.dot(X_centered, components_matrix.T)
        
        k_components = components_matrix.shape[0]
        pca_features = [f"PCA_{feature_type}_PC{i+1}" for i in range(k_components)]
        
        for i, name in enumerate(pca_features):
            transformed_data[name] = X_transformed[:, i]

    return pd.DataFrame(transformed_data, index=df.index)