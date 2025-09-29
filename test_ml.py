import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    # Test that train_model returns a RandomForestClassifier
    """
    # Create simple training data
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 1])
    
    model = train_model(X_train, y_train)
    
    # Check that it returns the expected model type
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, 'predict')


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # Test that compute_model_metrics returns expected values
    """
    # Perfect predictions
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    # Perfect predictions should give 1.0 for all metrics
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    
    # Check return types
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)


# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    # Test that inference returns predictions of correct shape
    """
    # Create simple model and data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 1])
    model = train_model(X_train, y_train)
    
    # Test inference
    X_test = np.array([[2, 3], [4, 5]])
    predictions = inference(model, X_test)
    
    # Check output shape and type
    assert len(predictions) == 2
    assert isinstance(predictions, np.ndarray)
    # Predictions should be 0 or 1
    assert all(pred in [0, 1] for pred in predictions)
