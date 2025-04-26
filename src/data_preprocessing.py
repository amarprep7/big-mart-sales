import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .logger import logger

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.info('Starting data preprocessing')
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
    data['Store_Age_Size'] = data['Outlet_Establishment_Year'] * data['Outlet_Size'].astype(str).map({'Small': 1, 'Medium': 2, 'High': 3})
    data['Visibility_MRP'] = data['Item_Visibility'] * data['Item_MRP']

    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        oe = OrdinalEncoder()
        data[col] = oe.fit_transform(data[[col]])

    logger.info('Data preprocessing completed')
    return data
