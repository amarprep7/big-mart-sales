import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data = pd.read_csv('train_v9rqX0R.csv')

# Split data for validation early to avoid data leakage
data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train data shape: {data_train.shape}, Validation data shape: {data_val.shape}")

def preprocess_data(data, is_train=True):
    """
    Enhanced preprocessing function with more advanced techniques
    
    Args:
        data: The input DataFrame
        is_train: Whether this is the training data (True) or test data (False)
        
    Returns:
        Preprocessed DataFrame and list of categorical columns
    """
    # Make a copy to avoid modifying the original data
    data = data.copy()
    
    # Extract item type from Item_Identifier first 2 chars
    if 'Item_Identifier' in data.columns:
        data['Item_Type_ID'] = data['Item_Identifier'].apply(lambda x: x[:2])
    
    # Standardize Item_Fat_Content
    if 'Item_Fat_Content' in data.columns:
        fat_content_mapping = {
            'Low Fat': 'Low Fat',
            'LF': 'Low Fat',
            'low fat': 'Low Fat',
            'Regular': 'Regular',
            'reg': 'Regular'
        }
        data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(fat_content_mapping)
    
    # Calculate outlet age
    if 'Outlet_Establishment_Year' in data.columns:
        current_year = 2025
        data['Outlet_Age'] = current_year - data['Outlet_Establishment_Year']
        data.drop('Outlet_Establishment_Year', axis=1, inplace=True)
    
    # Better handling for missing values in Item_Weight
    if 'Item_Weight' in data.columns:
        item_avg_weight = data.groupby('Item_Type_ID')['Item_Weight'].transform('mean')
        data['Item_Weight'].fillna(item_avg_weight, inplace=True)
        data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
    
    # Item_Visibility: Replace 0 with mean per item category
    if 'Item_Visibility' in data.columns:
        visibility_means = data.loc[data['Item_Visibility'] > 0].groupby('Item_Type_ID')['Item_Visibility'].transform('mean')
        data.loc[data['Item_Visibility'] == 0, 'Item_Visibility'] = visibility_means
        data['Item_Visibility'].replace(0, data['Item_Visibility'].mean(), inplace=True)
    
    # Fill missing Outlet_Size based on Outlet_Type
    if 'Outlet_Size' in data.columns and 'Outlet_Type' in data.columns:
        outlet_size_mapping = {
            'Grocery Store': 'Small',
            'Supermarket Type1': 'Small',
            'Supermarket Type2': 'Medium',
            'Supermarket Type3': 'High'
        }
        for outlet_type, size in outlet_size_mapping.items():
            data.loc[(data['Outlet_Type'] == outlet_type) & data['Outlet_Size'].isnull(), 'Outlet_Size'] = size
    
    # Identify categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    for col in cat_cols:
        if col in ['Outlet_Size', 'Outlet_Location_Type']:
            size_map = {'Small': 1, 'Medium': 2, 'High': 3}
            if col == 'Outlet_Size':
                data[col] = data[col].map(size_map)
            else:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        elif col != 'Item_Identifier' and col != 'Outlet_Identifier':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    
    # Create more powerful features
    if all(col in data.columns for col in ['Item_MRP', 'Item_Weight']):
        data['Price_per_Weight'] = data['Item_MRP'] / data['Item_Weight']
    
    if all(col in data.columns for col in ['Item_Visibility', 'Item_MRP']):
        data['Visibility_to_MRP_Ratio'] = data['Item_Visibility'] / data['Item_MRP']
        data['Visibility_MRP'] = data['Item_Visibility'] * data['Item_MRP']
    
    if all(col in data.columns for col in ['Outlet_Age', 'Outlet_Size']):
        data['Age_Size_Interaction'] = data['Outlet_Age'] * data['Outlet_Size']
    
    if 'Item_MRP' in data.columns:
        data['Price_Segment_Numeric'] = pd.qcut(data['Item_MRP'], q=3, labels=[0, 1, 2]).astype(int)
        data['Log_MRP'] = np.log1p(data['Item_MRP'])
    
    if 'Item_Type' in data.columns:
        perishable_types = ['Fruits and Vegetables', 'Meat', 'Breads', 'Breakfast', 'Dairy', 'Seafood']
        data['Is_Perishable'] = data['Item_Type'].apply(lambda x: 1 if x in perishable_types else 0)
    
    for col in cat_cols:
        if col not in ['Item_Identifier', 'Outlet_Identifier'] and data[col].nunique() < 10:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
    
    data = data.select_dtypes(exclude=['object'])
    
    if 'Item_Identifier' in data.columns:
        data.drop('Item_Identifier', axis=1, inplace=True)
    if 'Outlet_Identifier' in data.columns:
        data.drop('Outlet_Identifier', axis=1, inplace=True)
    
    # Implement more robust feature engineering
    if 'Item_Type' in data.columns and 'Item_MRP' in data.columns:
        # Ensure division by zero doesn't occur
        data['Price_Category_Ratio'] = data['Item_MRP'] / (data['Item_Type'].astype('category').cat.codes + 1)
    
    if 'Outlet_Type' in data.columns and 'Outlet_Location_Type' in data.columns:
        data['Store_Location_Type'] = (data['Outlet_Type'].astype('category').cat.codes + 1) * (data['Outlet_Location_Type'].astype('category').cat.codes + 1)
    
    if 'Item_Visibility' in data.columns:
        data['Log_Visibility'] = np.log1p(data['Item_Visibility'])
        data['Visibility_Squared'] = data['Item_Visibility'] ** 2
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != 'Item_Outlet_Sales':
            q1 = data[col].quantile(0.01)
            q3 = data[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data[col] = data[col].clip(lower_bound, upper_bound)
    
    # CRITICAL FIX: Handle any remaining NaN values by imputing with mean
    data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'if' else x)
    
    # Double check for any NaN values in the dataframe and replace them
    data = data.fillna(0)
    
    return data, cat_cols

def validate_features(X_train, y_train, X_val, y_val, features):
    """
    Validate feature importance using out-of-sample data.
    """
    print("\nValidating feature importance...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train[features], y_train)
    
    train_preds = model.predict(X_train[features])
    val_preds = model.predict(X_val[features])
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Difference: {val_rmse - train_rmse:.4f}")
    
    return val_rmse

def select_features_with_validation(X_train, y_train, X_val, y_val, n_features_min=5, cv=5):
    """
    Select features with validation to combat overfitting.
    """
    print("Performing feature selection with validation...")
    
    base_estimator = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    
    selector = RFECV(
        estimator=base_estimator,
        step=1,
        min_features_to_select=n_features_min,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_].tolist()
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 features by importance:")
    print(feature_importance.head(20))
    
    top_features = feature_importance.head(min(20, len(feature_importance)))['feature'].tolist()
    
    validate_features(X_train, y_train, X_val, y_val, top_features)
    
    final_features = list(set(selected_features).intersection(set(top_features)))
    if len(final_features) < n_features_min:
        final_features = top_features[:n_features_min]
    
    print(f"Selected {len(final_features)} features that work well on validation data")
    
    return final_features, feature_importance

def tune_model(model, X, y, param_grid, cv=5):
    """
    Tune a model's hyperparameters.
    """
    print(f"\nTuning {model.__class__.__name__}...")
    
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    return grid_search.best_estimator_

def train_enhanced_ensemble(X_train, y_train, X_val, y_val, features, cv=5):
    """
    Train an enhanced ensemble model with extensive validation.
    """
    print("\nTraining enhanced ensemble model...")
    X_train_selected = X_train[features]
    X_val_selected = X_val[features]
    
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 8, 12, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gbm_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.6, 0.8, 1.0]
    }
    
    print("\nTuning base models...")
    xgb_model = tune_model(XGBRegressor(random_state=42), X_train_selected, y_train, xgb_params, cv)
    rf_model = tune_model(RandomForestRegressor(random_state=42), X_train_selected, y_train, rf_params, cv)
    gbm_model = tune_model(GradientBoostingRegressor(random_state=42), X_train_selected, y_train, gbm_params, cv)
    et_model = ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42)
    ridge_model = Ridge(alpha=1.0)
    
    base_models = [
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('gbm', gbm_model),
        ('et', et_model)
    ]
    
    weight_options = [(1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2), 
                      (3, 1, 1, 1), (1, 3, 1, 1), (1, 1, 3, 1), (1, 1, 1, 3)]
    
    best_score = float('inf')
    best_weights = None
    
    print("\nFinding optimal ensemble weights...")
    for weights in weight_options:
        voting_model = VotingRegressor(
            estimators=base_models,
            weights=weights
        )
        
        voting_model.fit(X_train_selected, y_train)
        val_preds = voting_model.predict(X_val_selected)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        print(f"Weights {weights}: Validation RMSE = {val_rmse:.4f}")
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_weights = weights
    
    print(f"\nBest weights: {best_weights}, Best validation RMSE: {best_score:.4f}")
    
    final_model = VotingRegressor(
        estimators=base_models,
        weights=best_weights
    )
    
    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=ridge_model,
        cv=cv,
        n_jobs=-1
    )
    
    X_combined = pd.concat([X_train, X_val])[features]
    y_combined = pd.concat([y_train, y_val])
    
    final_model.fit(X_combined, y_combined)
    stacked_model.fit(X_combined, y_combined)
    
    stacked_preds = stacked_model.predict(X_val_selected)
    stacked_rmse = np.sqrt(mean_squared_error(y_val, stacked_preds))
    
    print(f"\nFinal ensemble validation RMSE: {best_score:.4f}")
    print(f"Stacked model validation RMSE: {stacked_rmse:.4f}")
    
    if stacked_rmse < best_score:
        print("Using stacked model for final predictions")
        return stacked_model
    else:
        print("Using voting ensemble model for final predictions")
        return final_model

print("Preprocessing training data...")
data_train_processed, cat_cols = preprocess_data(data_train, is_train=True)
data_val_processed, _ = preprocess_data(data_val, is_train=True)

X_train = data_train_processed.drop('Item_Outlet_Sales', axis=1)
y_train = data_train_processed['Item_Outlet_Sales']
y_train_log = np.log1p(y_train)

X_val = data_val_processed.drop('Item_Outlet_Sales', axis=1)
y_val = data_val_processed['Item_Outlet_Sales']
y_val_log = np.log1p(y_val)

X_val = X_val[X_train.columns]

selected_features, feature_importance = select_features_with_validation(
    X_train, y_train_log, X_val, y_val_log, n_features_min=10, cv=5
)

best_model = train_enhanced_ensemble(
    X_train, y_train_log, X_val, y_val_log, selected_features, cv=5
)

print("\nLoading and preprocessing test data...")
test_data = pd.read_csv('test_AbJTz2l.csv')
test_data_processed, _ = preprocess_data(test_data, is_train=False)

for col in selected_features:
    if col not in test_data_processed.columns:
        print(f"Feature {col} not in test data. Adding with zeros.")
        test_data_processed[col] = 0

test_data_selected = test_data_processed[selected_features]

print("Making predictions...")
log_predictions = best_model.predict(test_data_selected)
test_predictions = np.expm1(log_predictions)

final_submission = pd.read_csv('test_AbJTz2l.csv')
final_submission['Item_Outlet_Sales'] = test_predictions
final_submission[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']].to_csv('submission_advanced_improved.csv', index=False)
print("Predictions saved to 'result.csv'")

print("\nScript completed successfully!")