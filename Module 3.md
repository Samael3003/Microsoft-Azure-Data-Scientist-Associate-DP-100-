# Module 3: Orchestrate Pipelines and Deploy Real-Time Machine Learning Services with Azure Machine Learning

## 1. Overview
Orchestrating machine learning training with pipelines is a key element of DevOps for machine learning. This module covers how to create, publish, and run pipelines to train models in Azure Machine Learning. Additionally, it details the process of registering and deploying ML models with the Azure Machine Learning service.

## 2. Creating and Managing Pipelines
A machine learning pipeline in Azure is a workflow of multiple steps executed sequentially or in parallel, enabling automation of the end-to-end machine learning lifecycle.

**Components of a Pipeline:**
- **Steps:** Individual operations in the pipeline (e.g., data preprocessing, model training, evaluation).
- **Datasets:** Input data required for the steps.
- **Compute Targets:** Compute resources where steps will be executed.
- **Pipeline Parameters:** Dynamic values that can be passed to the pipeline at runtime.

**Steps to Create a Pipeline:**
1. **Import Required Libraries:**
   ```python
   from azureml.core import Workspace, Experiment, Datastore
   from azureml.pipeline.core import Pipeline, PipelineData
   from azureml.pipeline.steps import PythonScriptStep
   ```

2. **Initialize Workspace and Datastore:**
   ```python
   ws = Workspace.from_config()
   datastore = Datastore.get(ws, 'workspaceblobstore')
   ```

3. **Create Pipeline Steps:**
   - Define the steps for data preprocessing, training, and evaluation.
   ```python
   output = PipelineData('output', datastore=datastore)

   preprocess_step = PythonScriptStep(
       name="Preprocess Data",
       script_name="preprocess.py",
       arguments=["--output", output],
       outputs=[output],
       compute_target='cpu-cluster',
       source_directory='scripts'
   )

   train_step = PythonScriptStep(
       name="Train Model",
       script_name="train.py",
       arguments=["--input", output],
       inputs=[output],
       compute_target='gpu-cluster',
       source_directory='scripts'
   )
   ```

4. **Build the Pipeline:**
   ```python
   pipeline = Pipeline(workspace=ws, steps=[preprocess_step, train_step])
   ```

5. **Submit the Pipeline:**
   ```python
   experiment = Experiment(ws, 'pipeline-experiment')
   pipeline_run = experiment.submit(pipeline)
   pipeline_run.wait_for_completion(show_output=True)
   ```

## 3. Publishing and Running Pipelines
Once created, pipelines can be published to make them reusable and shareable.

**Steps to Publish a Pipeline:**
1. **Publish the Pipeline:**
   ```python
   published_pipeline = pipeline.publish(name="My_Pipeline",
                                         description="Pipeline for training and evaluating model",
                                         version="1.0")
   print(f"Published Pipeline ID: {published_pipeline.id}")
   ```

2. **Run the Published Pipeline:**
   ```python
   from azureml.pipeline.core import PublishedPipeline

   published_pipeline = PublishedPipeline.get(workspace=ws, id='pipeline-id')
   pipeline_run = published_pipeline.submit(ws, experiment_name='pipeline-experiment')
   pipeline_run.wait_for_completion(show_output=True)
   ```

## 4. Registering and Deploying Models
Models trained using Azure Machine Learning can be registered in the workspace and deployed as web services for real-time predictions.

**Steps to Register a Model:**
1. **Register the Model:**
   ```python
   from azureml.core.model import Model

   model = Model.register(workspace=ws,
                          model_path='outputs/model.pkl',
                          model_name='my_model',
                          tags={'area': 'classification'},
                          description='A classification model')
   ```

**Steps to Deploy a Model as a Web Service:**
1. **Define Inference Configuration:**
   ```python
   from azureml.core.model import InferenceConfig
   from azureml.core.environment import Environment

   environment = Environment.from_conda_specification(name='env', file_path='environment.yml')
   inference_config = InferenceConfig(entry_script='score.py', environment=environment)
   ```

2. **Define Deployment Configuration:**
   ```python
   from azureml.core.webservice import AciWebservice

   deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
   ```

3. **Deploy the Model:**
   ```python
   service = Model.deploy(workspace=ws,
                          name='my-web-service',
                          models=[model],
                          inference_config=inference_config,
                          deployment_config=deployment_config)
   service.wait_for_deployment(show_output=True)
   print(f"Scoring URI: {service.scoring_uri}")
   ```

## 5. Testing and Consuming the Deployed Web Service
After deploying the model, you can test the web service using HTTP requests.

**Testing the Web Service:**
1. **Prepare the Input Data:**
   ```python
   import json

   input_data = json.dumps({
       'data': [[5.1, 3.5, 1.4, 0.2]]
   })
   ```

2. **Send a Request to the Web Service:**
   ```python
   import requests

   scoring_uri = 'http://<your-service-name>.azurewebsites.net/score'
   headers = {'Content-Type': 'application/json'}
   response = requests.post(scoring_uri, data=input_data, headers=headers)
   print(response.json())
   ```

**Key Points to Remember:**
- Pipelines automate and orchestrate machine learning workflows, consisting of multiple steps.
- Pipelines can be created, published, and run, facilitating reuse and sharing.
- Models can be registered in the Azure Machine Learning workspace.
- Models can be deployed as web services for real-time predictions.
- Deployed web services can be tested and consumed using standard HTTP requests.

These notes cover the key aspects of orchestrating pipelines and deploying real-time machine learning services with Azure Machine Learning, providing a comprehensive guide to creating, managing, and utilizing pipelines and deployed models.
