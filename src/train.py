import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
import joblib
from .config import TRAIN_DATA_PATH, MODEL_PATH, RANDOM_STATE
from .data_preprocessing import preprocess_data
from .logger import logger

def train_top_features_model(X, y, n_features=20, cv_folds=5):
    logger.info('Training XGBRegressor for feature importance')
    xg_all = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE)
    xg_all.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xg_all.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = feature_importance.head(n_features)
    X_selected = X[top_features['feature'].tolist()]
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
    model = XGBRegressor(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=cv_folds, scoring='r2', n_jobs=-1, verbose=1, random_state=RANDOM_STATE)
    search.fit(X_selected, y)
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")
    cv_scores = cross_val_score(best_model, X_selected, y, cv=cv_folds, scoring='r2', n_jobs=-1)
    mean_r2_score = cv_scores.mean()
    logger.info(f"Cross-validation R2 scores: {cv_scores}")
    logger.info(f"Mean R2 score: {mean_r2_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
    return best_model, top_features['feature'].tolist(), feature_importance, mean_r2_score

def main():
    logger.info('Loading training data')
    data = pd.read_csv(TRAIN_DATA_PATH)
    data = preprocess_data(data)
    X = data.drop('Item_Outlet_Sales', axis=1)
    y = data['Item_Outlet_Sales']
    logger.info('Starting model training')
    final_model, features, feature_importance, mean_r2_score = train_top_features_model(X, y, n_features=6)
    joblib.dump(final_model, MODEL_PATH)
    # Save the features used for training
    features_path = MODEL_PATH.replace('.joblib', '_features.txt')
    with open(features_path, 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Top features: {features}")
    logger.info(f"Feature importance:\n{feature_importance}")
    logger.info(f"Mean R2 score: {mean_r2_score}")
main()
