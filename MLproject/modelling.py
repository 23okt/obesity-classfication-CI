import os
import sys
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    train_file = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), '..', 'dataset', 'obesity_data_train_preprocessing.csv')
    test_file = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), '..', 'dataset', 'obesity_data_test_preprocessing.csv')
    
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop("ObesityCategory", axis=1)
    y_train = train_df["ObesityCategory"]
    X_test = test_df.drop("ObesityCategory", axis=1)
    y_test = test_df["ObesityCategory"]
    
    input_example = X_train.iloc[:5]
    
with mlflow.start_run():
        # Log params manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Autolog untuk sklearn
        mlflow.sklearn.autolog()

        # Inisialisasi dan latih model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluasi
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        print(f"n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"Model accuracy: {accuracy:.4f}")