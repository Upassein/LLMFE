"""
CART Decision Tree Feedback for LLM-FE
Based on OCTree paper: https://arxiv.org/abs/2406.08527

Trains shallow CART (Classification and Regression Tree) to analyze feature importance
and converts tree structure to human-readable text for LLM feedback.
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re


# Original ERA5 features (hardcoded)
# 7 weather features per grid + 6 time features = 13 columns in template
ORIGINAL_ERA5_FEATURES = [
    # Weather features (from grid1 template)
    'relative_humidity_2m',
    'wind_speed_10m',
    'wind_direction_10m',
    'wind_speed_100m',
    'wind_direction_100m',
    'pressure_msl',
    'surface_pressure',
    # Time features (global, shared across grids)
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos'
]


def train_cart_for_regression(X: pd.DataFrame, y: pd.Series, max_depth: int = 3,
                               random_state: int = 0) -> tree.DecisionTreeRegressor:
    """
    Train a shallow CART regression tree to analyze feature importance.

    Args:
        X: Feature matrix (with new engineered features)
        y: Target variable
        max_depth: Maximum depth of the tree (1-3 recommended for interpretability)
        random_state: Random seed

    Returns:
        Best CART model based on validation RMSE
    """
    # Split data for CART training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, shuffle=False
    )

    best_rmse = float('inf')
    best_cart = None

    # Search for best depth
    for depth in range(1, max_depth + 1):
        cart = tree.DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        cart.fit(X_train, y_train)

        val_pred = cart.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_cart = cart

    return best_cart


def tree_to_text(cart_model: tree.DecisionTreeRegressor, feature_names: list) -> str:
    """
    Convert CART decision tree to human-readable text.

    Args:
        cart_model: Trained CART regression tree
        feature_names: List of feature names

    Returns:
        Formatted text representation of the decision tree

    Example output:
        if wind_power_density_120m > 0.65:
          if direction_consistency > 0.80:
            power = 88.3 MW
          else:
            power = 61.2 MW
        else:
          power = 38.7 MW
    """
    tree_ = cart_model.tree_

    def recurse(node: int, depth: int) -> str:
        indent = "  " * depth
        result = ""

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Internal node (has a split)
            feature_name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            result += f"{indent}if {feature_name} > {threshold:.2f}:\n"
            result += recurse(tree_.children_right[node], depth + 1)
            result += f"{indent}else:\n"
            result += recurse(tree_.children_left[node], depth + 1)
        else:
            # Leaf node (prediction)
            value = float(tree_.value[node][0][0])  # Extract scalar from array
            result += f"{indent}power = {value:.2f} MW\n"

        return result

    return recurse(0, 0)


def get_cart_feedback(X: pd.DataFrame, y: pd.Series, max_depth: int = 3) -> str:
    """
    Train CART and generate text feedback for LLM.

    Args:
        X: Feature matrix with engineered features (with grid suffixes like _grid1, _grid2)
        y: Target variable
        max_depth: Maximum tree depth

    Returns:
        Formatted CART analysis text with:
        1. Top-5 engineered features by importance
        2. Complete decision tree
        3. Grid suffixes removed for LLM clarity
    """
    try:
        # Get original feature names and remove grid suffixes for identification
        original_feature_names = X.columns.tolist()

        # Map: Remove era5_grid1_, era5_grid2_, era5_grid3_, era5_grid4_ prefixes
        template_feature_names = []
        for name in original_feature_names:
            # Remove era5_gridX_ pattern (handles era5_grid1_wind_speed_10m -> wind_speed_10m)
            template_name = re.sub(r'era5_grid[1-4]_', '', name)
            template_feature_names.append(template_name)

        # Identify engineered feature columns (those NOT in ORIGINAL_ERA5_FEATURES)
        engineered_indices = []
        engineered_template_names = []
        for i, template_name in enumerate(template_feature_names):
            if template_name not in ORIGINAL_ERA5_FEATURES:
                engineered_indices.append(i)
                engineered_template_names.append(template_name)

        # Check if we have engineered features
        if len(engineered_indices) == 0:
            return "=== CART Analysis ===\n\nNo engineered features detected in this version.\n"

        # Extract only engineered features for CART training
        X_engineered = X.iloc[:, engineered_indices]

        # Train CART on ONLY engineered features
        cart_model = train_cart_for_regression(X_engineered, y, max_depth=max_depth)

        # Convert tree to text using engineered feature template names
        cart_text = tree_to_text(cart_model, engineered_template_names)

        # Aggregate feature importance by template name (sum across grids)
        importances = cart_model.feature_importances_
        aggregated_importances = {}
        for i, template_name in enumerate(engineered_template_names):
            if template_name in aggregated_importances:
                aggregated_importances[template_name] += importances[i]
            else:
                aggregated_importances[template_name] = importances[i]

        # Build feedback
        feedback = "=== CART Analysis (Engineered Features Only) ===\n\n"

        # Show Top-1 most important engineered feature
        sorted_eng = sorted(aggregated_importances.items(), key=lambda x: x[1], reverse=True)
        top_k = min(1, len(sorted_eng))

        feedback += "Most Important Engineered Feature:\n"
        for feat, imp in sorted_eng[:top_k]:
            feedback += f"  • {feat}: {imp*100:.1f}% importance\n"
        feedback += "\n"

        # Show complete decision tree
        feedback += "Decision Tree (using engineered features):\n"
        feedback += cart_text

        return feedback

    except Exception as e:
        return f"CART generation failed: {str(e)}"


def get_compressed_cart_feedback(X: pd.DataFrame, y: pd.Series, max_depth: int = 2) -> str:
    """
    Generate compressed CART feedback to save tokens.

    Returns:
        Compressed single-line summary of key decision paths
    """
    try:
        cart_model = train_cart_for_regression(X, y, max_depth=max_depth)
        importances = cart_model.feature_importances_
        feature_names = X.columns.tolist()

        # Get top 3 important features
        top_3_idx = np.argsort(importances)[-3:][::-1]

        summary = "Key Features: "
        for idx in top_3_idx:
            if importances[idx] > 0.01:  # Only show if importance > 1%
                summary += f"{feature_names[idx]} ({importances[idx]:.1%}), "

        return summary.rstrip(", ")

    except Exception as e:
        return f"CART analysis failed: {str(e)}"
