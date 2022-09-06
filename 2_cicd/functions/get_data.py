import mlrun
import pandas as pd
from sklearn.datasets import load_iris

def prep_data(context, label_column='label'):
    iris = load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_labels = pd.DataFrame(data=iris.target, columns=[label_column])
    iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)
    
    context.logger.info(f'saving iris dataframe to {context.artifact_path}')
    context.log_dataset('iris_dataset', df=iris_dataset, format="csv", index=False)
    context.log_result('num_rows', iris_dataset.shape[0])