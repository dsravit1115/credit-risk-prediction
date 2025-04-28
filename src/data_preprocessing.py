import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Encode categorical columns (example: Gender, Employment_Status)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # Features and Target
    X = df.drop('Credit_Risk', axis=1)
    y = df['Credit_Risk']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
