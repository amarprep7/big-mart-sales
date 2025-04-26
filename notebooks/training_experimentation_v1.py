import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# --- Data Loading ---
train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')

# --- Preprocessing Function ---
def preprocess(df, is_train=True, le_dict=None):
    df = df.copy()
    # Fill missing values
    df['Item_Weight'] = df['Item_Weight'].interpolate()
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.nan).interpolate()
    outlet_size_map = {
        'Grocery Store': 'Small',
        'Supermarket Type1': 'Small',
        'Supermarket Type2': 'Medium',
        'Supermarket Type3': 'Medium'
    }
    for k, v in outlet_size_map.items():
        df.loc[(df['Outlet_Type'] == k) & df['Outlet_Size'].isnull(), 'Outlet_Size'] = v
    # Standardize Item_Fat_Content
    fat_map = {'Low Fat': 'Low Fat', 'LF': 'Low Fat', 'low fat': 'Low Fat', 'Regular': 'Regular', 'reg': 'Regular'}
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(fat_map)
    # Feature engineering
    df['Item_Identifier'] = df['Item_Identifier'].str[:2]
    df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Price_per_Weight'] = df['Item_MRP'] / (df['Item_Weight'] + 1e-3)
    df['Visibility_MRP'] = df['Item_Visibility'] * df['Item_MRP']
    # Log transform
    for col in ['Item_Visibility', 'Item_MRP', 'Item_Weight']:
        df[col] = np.log1p(df[col])
    # Encode categoricals
    cat_cols = df.select_dtypes(include='object').columns
    if is_train:
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        return df, le_dict
    else:
        for col in cat_cols:
            le = le_dict[col]
            df[col] = le.transform(df[col].astype(str))
        return df

# --- Preprocess Data ---
train, le_dict = preprocess(train, is_train=True)
test = preprocess(test, is_train=False, le_dict=le_dict)

# --- Outlier Capping ---
def cap_outliers(y, lower=0.01, upper=0.99):
    lower_val = y.quantile(lower)
    upper_val = y.quantile(upper)
    return y.clip(lower=lower_val, upper=upper_val)

# --- Prepare Features ---
X = train.drop(['Item_Outlet_Sales'], axis=1)
y = cap_outliers(train['Item_Outlet_Sales'])
X_test = test.copy()

# --- Polynomial Features ---
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)
poly_cols = poly.get_feature_names_out(X.columns)
X_poly = pd.DataFrame(X_poly, columns=poly_cols)
X_test_poly = pd.DataFrame(X_test_poly, columns=poly_cols)

# --- Feature Selection with XGBRegressor ---
xgb_fs = XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse')
xgb_fs.fit(X_poly, y)
feat_imp = pd.Series(xgb_fs.feature_importances_, index=X_poly.columns)
top_features = feat_imp.sort_values(ascending=False).head(15).index.tolist()
X_poly_sel = X_poly[top_features]
X_test_poly_sel = X_test_poly[top_features]

# --- Stacking Model ---
xgb = XGBRegressor(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='rmse')
rf = RandomForestRegressor(n_estimators=200, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
stack = StackingRegressor(
    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
    final_estimator=XGBRegressor(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='rmse'),
    n_jobs=-1
)

# --- Cross-validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(stack, X_poly_sel, y, cv=kf, scoring='r2', n_jobs=-1)
print(f"Stacked Model CV R2: {cv_scores}\nMean R2: {cv_scores.mean():.4f}")

# --- Train and Predict ---
stack.fit(X_poly_sel, y)
preds = stack.predict(X_test_poly_sel)

# --- Save Submission ---
sub = test[['Item_Identifier', 'Outlet_Identifier']].copy()
sub['Item_Outlet_Sales'] = preds
sub.to_csv('result.csv', index=False)
print("Submission saved to submissionv8.csv")