# Module 4: Deploy Batch Inference Pipelines and Tune Hyperparameters with Azure Machine Learning

## 1. Overview
This module focuses on deploying machine learning models to generate predictions from large datasets in batch processes and tuning hyperparameters to optimize model performance. You will learn how to use Azure Machine Learning to publish a batch inference pipeline and leverage cloud-scale experiments to choose optimal hyperparameter values for model training.

## 2. Deploying Batch Inference Pipelines
Batch inference pipelines are used to process large volumes of data in batch mode, allowing for predictions on a large scale.

**Steps to Create and Deploy a Batch Inference Pipeline:**

1. **Define the Batch Scoring Script:**
   - Prepare a script to load the model and perform predictions on input data.
   ```python
   # batch_scoring.py
   import pandas as pd
   from azureml.core.model import Model

   def run(mini_batch):
       model = Model.get_model_path('my_model')
       df = pd.read_csv(mini_batch)
       # Perform inference
       predictions = model.predict(df)
       return predictions
   ```

2. **Create a Data Reference for Input Data:**
   ```python
   from azureml.core import Dataset

   datastore = ws.get_default_datastore()
   input_data = Dataset.File.from_files(path=(datastore, 'batch_data/*.csv'))
   ```

3. **Define the Batch Inference Step:**
   ```python
   from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
   from azureml.pipeline.core import PipelineData

   output = PipelineData(name='inferences', datastore=datastore)

   parallel_run_config = ParallelRunConfig(
       source_directory='.',
       entry_script='batch_scoring.py',
       mini_batch_size='5MB',
       error_threshold=10,
       output_action='append_row',
       environment=environment,
       compute_target=compute_target
   )

   batch_inference_step = ParallelRunStep(
       name='batch-inference',
       parallel_run_config=parallel_run_config,
       inputs=[input_data.as_named_input('input')],
       output=output,
       allow_reuse=True
   )
   ```

4. **Build and Run the Pipeline:**
   ```python
   from azureml.pipeline.core import Pipeline

   pipeline = Pipeline(workspace=ws, steps=[batch_inference_step])
   pipeline_run = pipeline.submit('batch-inference-pipeline')
   pipeline_run.wait_for_completion(show_output=True)
   ```

## 3. Hyperparameter Tuning
Hyperparameter tuning is essential to optimize the performance of machine learning models. Azure Machine Learning provides a robust mechanism to automate hyperparameter optimization at scale.

**Steps to Perform Hyperparameter Tuning:**

1. **Define the Training Script:**
   - Modify the training script to accept hyperparameters as arguments.
   ```python
   # train.py
   import argparse
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

   parser = argparse.ArgumentParser()
   parser.add_argument('--C', type=float, default=1.0)
   args = parser.parse_args()

   iris = load_iris()
   X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

   model = LogisticRegression(C=args.C)
   model.fit(X_train, y_train)
   accuracy = model.score(X_test, y_test)
   print(f'Accuracy: {accuracy}')
   ```

2. **Configure the Hyperparameter Tuning Experiment:**
   ```python
   from azureml.core import ScriptRunConfig, Environment
   from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice

   param_sampling = GridParameterSampling({
       '--C': choice(0.1, 1.0, 10.0)
   })

   script_config = ScriptRunConfig(source_directory='.',
                                   script='train.py',
                                   compute_target=compute_target,
                                   environment=environment)

   hyperdrive_config = HyperDriveConfig(run_config=script_config,
                                        hyperparameter_sampling=param_sampling,
                                        primary_metric_name='Accuracy',
                                        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                        max_total_runs=10)
   ```

3. **Submit the Hyperparameter Tuning Experiment:**
   ```python
   from azureml.core import Experiment

   experiment = Experiment(ws, 'hyperparameter-tuning')
   hyperdrive_run = experiment.submit(hyperdrive_config)
   hyperdrive_run.wait_for_completion(show_output=True)
   ```

4. **Analyze Hyperparameter Tuning Results:**
   ```python
   best_run = hyperdrive_run.get_best_run_by_primary_metric()
   best_run_metrics = best_run.get_metrics()
   best_run_params = best_run.get_details()['runDefinition']['arguments']
   print(f'Best Run Metrics: {best_run_metrics}')
   print(f'Best Run Parameters: {best_run_params}')
   ```

**Key Concepts:**
- **Parameter Sampling:** Defining the range of hyperparameters to be tested. Techniques include Grid Sampling, Random Sampling, and Bayesian Sampling.
- **Primary Metric:** The performance metric to be optimized, such as accuracy or loss.
- **Early Termination:** Mechanism to stop poorly performing runs early, conserving resources.

**Key Points to Remember:**
- Batch inference pipelines allow processing large datasets in batch mode, making predictions on a large scale.
- Hyperparameter tuning automates the optimization of model parameters, improving model performance.
- Azure Machine Learning provides robust tools to create, manage, and run batch inference pipelines and hyperparameter tuning experiments.

These notes cover the essential aspects of deploying batch inference pipelines and tuning hyperparameters with Azure Machine Learning, providing a detailed guide to managing large-scale predictions and optimizing model performance.
