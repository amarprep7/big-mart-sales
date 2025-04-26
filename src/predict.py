import pandas as pd
import joblib
from .config import TEST_DATA_PATH, MODEL_PATH
from .data_preprocessing import preprocess_data

def run_inference():
    # Load test data
    test_data = pd.read_csv(TEST_DATA_PATH)
    test_data_orig = test_data.copy()
    # Preprocess test data
    test_data = preprocess_data(test_data)
    # Load trained model
    model = joblib.load(MODEL_PATH)
    # Load the features used for training
    features_path = MODEL_PATH.replace('.joblib', '_features.txt')
    with open(features_path) as f:
        features = [line.strip() for line in f]
    # Use only the features used in training
    predictions = model.predict(test_data[features])
    # Prepare submission
    test_data_orig['Item_Outlet_Sales'] = predictions
    submission = test_data_orig[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
    submission.to_csv('submission_inference.csv', index=False)
    print('Inference complete. Results saved to submission_inference.csv')

if __name__ == "__main__":
    run_inference()
