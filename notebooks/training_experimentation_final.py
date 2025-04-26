import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Use correct file paths for train and test
train_path = 'big_mart_sales/data/processed/train_v9rqX0R.csv'
test_path = 'big_mart_sales/data/processed/test_AbJTz2l.csv'

# Load training data
data = pd.read_csv(train_path)

def preprocess_data(data):
    
    # Handle missing values
    data['Item_Weight'] = data['Item_Weight'].interpolate(method="linear")
    data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan).interpolate(method='linear')
    
    # Fill missing Outlet_Size based on Outlet_Type
    outlet_size_mapping = {
        'Grocery Store': 'Small',
        'Supermarket Type1': 'Small',
        'Supermarket Type2': 'Medium',
        'Supermarket Type3': 'Medium'
    }
    
    for outlet_type, size in outlet_size_mapping.items():
        data.loc[(data['Outlet_Type'] == outlet_type) & data['Outlet_Size'].isnull(), 'Outlet_Size'] = size
    
    # Standardize Item_Fat_Content
    fat_content_mapping = {
        'Low Fat': 'Low Fat',
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'Regular': 'Regular',
        'reg': 'Regular'
    }
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(fat_content_mapping)
    
    # Extract first two characters from Item_Identifier
    data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[:2])
    
    # Calculate outlet age
    current_year = 2025
    data['Outlet_Establishment_Year'] = current_year - data['Outlet_Establishment_Year']
    
    data['Price_per_Weight'] = data['Item_MRP'] / data['Item_Weight']
    data['Store_Age_Size'] = data['Outlet_Establishment_Year'] * data['Outlet_Size']
    data['Visibility_MRP'] = data['Item_Visibility'] * data['Item_MRP']
    
    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        oe = OrdinalEncoder()
        data[col] = oe.fit_transform(data[[col]])
    
    return data

data = preprocess_data(data)

X = data.drop('Item_Outlet_Sales',axis=1)
y = data['Item_Outlet_Sales']

def train_top_features_model(X, y, n_features=20, cv_folds=5):
    
    xg_all = XGBRegressor(n_estimators=100, random_state=42)
    xg_all.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xg_all.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top n features
    top_features = feature_importance.head(n_features)
    X_selected = X[top_features['feature'].tolist()]
    
    # Hyperparameter tuning for XGBRegressor
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }
    model = XGBRegressor(random_state=42)
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=cv_folds, scoring='r2', n_jobs=-1, verbose=1, random_state=42)
    search.fit(X_selected, y)
    best_model = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")
    cv_scores = cross_val_score(
        best_model, 
        X_selected, 
        y, 
        cv=cv_folds, 
        scoring='r2',
        n_jobs=-1
    )
    
    # Train final model with selected features
    final_model = best_model
    
    # Calculate mean R2 score
    mean_r2_score = cv_scores.mean()
    
    print(f"\nCross-validation R2 scores: {cv_scores}")
    print(f"Mean R2 score: {mean_r2_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return final_model, top_features['feature'].tolist(), feature_importance, mean_r2_score

final_model, features, feature_importance, mean_r2_score= train_top_features_model(X, y, n_features=6)

test_data = pd.read_csv(test_path)
test_data2 = test_data.copy()

test_data2 = preprocess_data(test_data2)
test_data2 = test_data2[features]

test_result = final_model.predict(test_data2)

test_data['Item_Outlet_Sales'] = test_result

test_data = test_data[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]

test_data.to_csv('train_check.csv', index=False)

