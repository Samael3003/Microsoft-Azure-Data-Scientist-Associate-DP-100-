# Module 2: Work with Data and Compute in Azure Machine Learning

## 1. Overview
Data is the foundation of machine learning. This module covers how to work with datastores and datasets in Azure Machine Learning, enabling you to build scalable, cloud-based model training solutions. You'll also learn how to use cloud compute in Azure Machine Learning to run training experiments at scale.

## 2. Working with Datastores
Datastores are a mechanism to securely connect to storage services on Azure. They act as a central place to manage and access data required for training machine learning models.

**Types of Datastores:**
- **Azure Blob Storage:** Used for large amounts of unstructured data such as text or binary data.
- **Azure Data Lake Storage:** Provides scalable and secure storage for big data analytics.
- **Azure SQL Database:** Managed relational database service.

**Steps to Register a Datastore:**
1. **Access Azure Machine Learning Studio:** Navigate to the Azure Machine Learning workspace.
2. **Register a New Datastore:**
   - Go to the "Datastores" section.
   - Click "New datastore".
   - Select the appropriate storage service (e.g., Azure Blob Storage).
   - Provide necessary details (name, storage account name, container name, access key).
   - Click "Register".

**Example: Registering a Blob Datastore using SDK:**
```python
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()
blob_datastore = Datastore.register_azure_blob_container(workspace=ws,
                                                         datastore_name='my_blob_datastore',
                                                         container_name='my-container',
                                                         account_name='my-account',
                                                         account_key='my-key')
```

## 3. Working with Datasets
Datasets in Azure Machine Learning are a way to manage and version data for machine learning experiments. They provide a way to handle data efficiently and ensure reproducibility.

**Types of Datasets:**
- **TabularDataset:** Represents data in a tabular format, such as CSV, SQL, or Parquet files.
- **FileDataset:** Represents unstructured data stored in files.

**Creating Datasets:**
- **From Datastore:**
  ```python
  from azureml.core import Dataset

  datastore = Datastore.get(ws, 'my_blob_datastore')
  dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'data/my-data.csv'))
  ```
- **From Local Files:**
  ```python
  dataset = Dataset.File.from_files(path=['./data/my-file.txt'])
  ```

**Registering Datasets:**
```python
dataset = dataset.register(workspace=ws,
                           name='my_dataset',
                           description='My dataset description',
                           create_new_version=True)
```

## 4. Using Cloud Compute for Training
Azure Machine Learning supports various compute targets to run training experiments at scale. Compute targets are compute resources used to run training scripts or host services.

**Types of Compute Targets:**
- **Azure Machine Learning Compute:** A managed compute infrastructure that allows you to run training scripts at scale.
- **Azure Databricks:** An Apache Spark-based analytics platform.
- **Azure HDInsight:** A fully managed cloud Hadoop and Spark service.
- **Virtual Machines (VMs):** Custom VMs on which you can run training scripts.

**Creating a Compute Target:**
1. **Using Azure Machine Learning Studio:**
   - Navigate to the "Compute" section.
   - Click "New".
   - Select the compute type (e.g., Azure Machine Learning Compute).
   - Configure the settings (e.g., VM size, scale settings).
   - Click "Create".

2. **Using SDK:**
   ```python
   from azureml.core.compute import AmlCompute, ComputeTarget

   compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                          max_nodes=4)
   compute_target = ComputeTarget.create(ws, 'cpu-cluster', compute_config)
   compute_target.wait_for_completion(show_output=True)
   ```

## 5. Running Training Experiments at Scale
Once the datastores, datasets, and compute targets are set up, you can run training experiments at scale.

**Steps to Run a Training Experiment:**
1. **Define the Training Script:**
   - Prepare the script that includes data preprocessing, model training, and evaluation.
   ```python
   # train.py
   from azureml.core.run import Run
   from sklearn.datasets import load_iris
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split

   run = Run.get_context()

   # Load dataset
   iris = load_iris()
   X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

   # Train model
   model = LogisticRegression()
   model.fit(X_train, y_train)

   # Log accuracy
   accuracy = model.score(X_test, y_test)
   run.log('accuracy', accuracy)
   run.complete()
   ```

2. **Submit the Script to a Compute Target:**
   ```python
   from azureml.core import ScriptRunConfig

   script_config = ScriptRunConfig(source_directory='.',
                                   script='train.py',
                                   compute_target=compute_target)
   run = Experiment(ws, 'iris-experiment').submit(script_config)
   run.wait_for_completion(show_output=True)
   ```

3. **Monitor the Run:**
   - Use the Azure Machine Learning Studio or SDK to monitor the progress of the experiment.
   ```python
   from azureml.widgets import RunDetails

   RunDetails(run).show()
   ```

**Key Points to Remember:**
- Datastores provide a way to securely connect to and manage data storage services.
- Datasets enable efficient data management and ensure reproducibility in experiments.
- Various compute targets allow running experiments at scale, including managed compute, Databricks, HDInsight, and VMs.
- Running training experiments involves defining a training script, submitting it to a compute target, and monitoring the run.

These notes cover the essential aspects of working with data and compute in Azure Machine Learning, providing a detailed guide to managing datastores, datasets, and running experiments at scale.
