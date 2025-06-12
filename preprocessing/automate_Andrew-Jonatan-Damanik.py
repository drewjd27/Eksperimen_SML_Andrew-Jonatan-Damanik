import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Drop missing values
    df = df.dropna()

    # Split features and target
    X = df.drop(columns=['species'])
    y = df['species']

    # Drop ID column
    X = X.drop(columns=['id'])

    # Define categorical and numerical columns
    categorical_cols = ['island', 'sex', 'year']
    numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Create ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    ohe = preprocessor.named_transformers_['cat']
    feature_names_cat = ohe.get_feature_names_out(categorical_cols)
    feature_names_all = list(feature_names_cat) + numerical_cols

    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names_all, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names_all, index=X_test.index)

    # Combine with targets
    train_df = X_train_df.copy()
    train_df['species'] = y_train.values

    test_df = X_test_df.copy()
    test_df['species'] = y_test.values

    return train_df, test_df


def save_preprocessed_data(train_df, test_df, output_dir='preprocessing'):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'penguins_train_preprocessing.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'penguins_test_preprocessing.csv'), index=False)
    print("Preprocessed datasets saved successfully.")

if __name__ == '__main__':
    # File input
    input_file = 'penguins_raw.csv'
    
    # Load and preprocess
    raw_df = load_data(input_file)
    train_df, test_df = preprocess_data(raw_df)

    # Save results
    save_preprocessed_data(train_df, test_df)