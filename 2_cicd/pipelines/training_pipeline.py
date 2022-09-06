from kfp import dsl
import mlrun

@dsl.pipeline(
    name="batch-pipeline-academy",
    description="Example of batch pipeline for Iguazio Academy"
)
def pipeline(label_column: str, test_size=0.2):
    
    # Ingest the data set
    ingest = mlrun.run_function(
        'get-data',
        handler='prep_data',
        params={'label_column': label_column},
        outputs=["iris_dataset"]
    )
    
    # Train a model   
    train = mlrun.run_function(
        "train-model",
        handler="train_model",
        inputs={"dataset": ingest.outputs["iris_dataset"]},
        params={
            "label_column": label_column,
            "test_size" : test_size
        },
        outputs=['model']
    )
    
    # Deploy the model as a serverless function
    deploy = mlrun.deploy_function(
        "deploy-model",
        models=[{"key": "model", "model_path": train.outputs["model"]}]
    )
