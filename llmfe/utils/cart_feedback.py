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
        X: Feature matrix with engineered features
        y: Target variable
        max_depth: Maximum tree depth

    Returns:
        Formatted CART analysis text
    """
    try:
        # Train CART
        cart_model = train_cart_for_regression(X, y, max_depth=max_depth)

        # Convert to text
        feature_names = X.columns.tolist()
        cart_text = tree_to_text(cart_model, feature_names)

        # Add feature importance summary
        importances = cart_model.feature_importances_
        if importances.max() > 0:
            top_feature_idx = importances.argmax()
            top_feature = feature_names[top_feature_idx]
            top_importance = importances[top_feature_idx]

            summary = f"Most Important Feature: {top_feature} (importance: {top_importance:.1%})\n\n"
            summary += "Decision Tree:\n" + cart_text
            return summary
        else:
            return "Decision Tree:\n" + cart_text

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
