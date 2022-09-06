from sklearn import ensemble
from sklearn.model_selection import train_test_split

import mlrun
from mlrun.frameworks.sklearn import apply_mlrun

def train_model(context, dataset: mlrun.DataItem, label_column: str, test_size: float):
    
    # Initialize our dataframes
    df = dataset.as_df()
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # Train/Test split Iris data-set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier()
    
    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model, model_name='my_model', X_test=X_test, y_test=y_test)
    
    # Train our model
    model.fit(X_train, y_train)