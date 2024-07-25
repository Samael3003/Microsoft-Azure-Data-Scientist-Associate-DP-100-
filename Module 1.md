# Module 1: Use the Azure Machine Learning SDK to Train a Model

## 1. Overview
Azure Machine Learning provides a cloud-based platform for training, deploying, and managing machine learning models. In this module, we will learn how to:
- Provision an Azure Machine Learning workspace.
- Use tools and interfaces to work with Azure Machine Learning.
- Run code-based experiments in an Azure Machine Learning workspace.
- Train and register a model in a workspace.

## 2. Provisioning an Azure Machine Learning Workspace
An Azure Machine Learning workspace is a foundational resource in the cloud that you use to experiment, train, and deploy machine learning models.

**Steps to Provision a Workspace:**
1. **Create an Azure Subscription:** Ensure you have an active Azure subscription.
2. **Access Azure Portal:** Navigate to the Azure portal (portal.azure.com).
3. **Create a Resource:** Click on "Create a resource" and search for "Machine Learning".
4. **Configure Workspace Settings:**
   - **Workspace Name:** Provide a unique name for your workspace.
   - **Subscription:** Select your Azure subscription.
   - **Resource Group:** Create a new resource group or select an existing one.
   - **Region:** Choose a region close to your data and resources.
   - **Workspace Edition:** Select the workspace edition (Basic or Enterprise).
5. **Review and Create:** Review the configurations and click "Create" to provision the workspace.

## 3. Using Tools and Interfaces to Work with Azure Machine Learning
Azure Machine Learning provides several tools and interfaces for managing machine learning workflows:
- **Azure Machine Learning Studio:** A web-based integrated development environment (IDE) for managing machine learning workflows.
- **Azure Machine Learning SDK for Python:** A powerful Python library to interact with the Azure Machine Learning service programmatically.

**Key Tools and Interfaces:**
- **Azure Machine Learning Studio:** Allows for creating and managing workspaces, experiments, datasets, models, and deployments.
- **Azure Machine Learning SDK:** Provides a rich set of functionalities for machine learning workflows, including data preparation, model training, deployment, and monitoring.

## 4. Running Code-Based Experiments in an Azure Machine Learning Workspace
Running experiments in Azure Machine Learning involves the following steps:

**Steps to Run an Experiment:**
1. **Install the Azure Machine Learning SDK:** Use pip to install the SDK.
   ```bash
   pip install azureml-sdk
   ```
2. **Import Required Libraries:**
   ```python
   from azureml.core import Workspace, Experiment
   ```
3. **Connect to the Workspace:**
   ```python
   ws = Workspace.from_config()
   ```
4. **Create an Experiment:**
   ```python
   experiment = Experiment(workspace=ws, name="my-experiment")
   ```
5. **Submit an Experiment:**
   ```python
   run = experiment.start_logging()
   # Your training code here
   run.complete()
   ```

## 5. Training and Registering a Model in a Workspace
Model training in Azure Machine Learning can be done using various frameworks like TensorFlow, PyTorch, and Scikit-learn. Once the model is trained, it can be registered in the workspace for easy access and deployment.

**Steps to Train and Register a Model:**
1. **Prepare Data:**
   - Upload data to the datastore.
   - Create a dataset from the data.
2. **Define the Training Script:**
   ```python
   # train.py
   from sklearn.datasets import load_iris
   from sklearn.linear_model import LogisticRegression
   from azureml.core.run import Run

   run = Run.get_context()
   iris = load_iris()
   X, y = iris.data, iris.target
   model = LogisticRegression()
   model.fit(X, y)
   run.log("accuracy", model.score(X, y))
   run.complete()
   ```
3. **Submit the Training Script:**
   ```python
   from azureml.core import ScriptRunConfig

   script_config = ScriptRunConfig(source_directory='.',
                                   script='train.py',
                                   compute_target='local')
   run = experiment.submit(script_config)
   run.wait_for_completion(show_output=True)
   ```
4. **Register the Model:**
   ```python
   from azureml.core.model import Model

   model = run.register_model(model_name='iris_model', model_path='outputs/model.pkl')
   print(model.name, model.id, model.version, sep='\t')
   ```

**Key Points to Remember:**
- A workspace is a centralized place to manage all the assets and activities related to machine learning projects.
- Azure Machine Learning Studio provides an easy-to-use interface for managing resources.
- The Azure Machine Learning SDK for Python allows for programmatic control over the machine learning workflow.
- Running experiments involves creating an experiment object, submitting the code, and logging the results.
- Models can be trained using various frameworks and registered for deployment and further use.

These notes cover the key aspects of using Azure Machine Learning SDK to train a model, providing a comprehensive guide to provisioning workspaces, running experiments, and registering trained models.
