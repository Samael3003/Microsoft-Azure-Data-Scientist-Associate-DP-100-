# Build and Operate Machine Learning Solutions with Azure Machine Learning

Welcome to the repository for the "Build and Operate Machine Learning Solutions with Azure Machine Learning" course notes. This repository is part of my Open Source Learning initiative to share knowledge and resources related to machine learning and cloud computing.

## Course Overview

Azure Machine Learning is a cloud platform for training, deploying, managing, and monitoring machine learning models. This course covers using the Azure Machine Learning Python SDK to create and manage enterprise-ready ML solutions.

## Module 1: Use the Azure Machine Learning SDK to Train a Model

### Overview
This module introduces the Azure Machine Learning workspace and the tools and interfaces for running code-based experiments. You will learn to provision an Azure Machine Learning workspace, train a model, and register it in the workspace.

### Detailed Notes
1. **Provision an Azure Machine Learning Workspace**
   - Create a workspace using the Azure portal or SDK.
   - Set up necessary resources such as compute instances and storage accounts.

2. **Work with Azure Machine Learning Tools and Interfaces**
   - Utilize the Azure Machine Learning studio and Python SDK.
   - Explore the Azure Machine Learning CLI for managing resources and experiments.

3. **Train and Register a Model**
   - Prepare your dataset and define the training script.
   - Use `ScriptRunConfig` to configure and submit a training run.
   - Register the trained model in the workspace for future use.

## Module 2: Work with Data and Compute in Azure Machine Learning

### Overview
Data is the foundation of machine learning. This module covers working with datastores and datasets in Azure Machine Learning to build scalable, cloud-based model training solutions. You will also learn to use cloud compute in Azure Machine Learning to run training experiments at scale.

### Detailed Notes
1. **Working with Datastores**
   - Connect to datastores and upload data.
   - Use datastores to manage and access data for training experiments.

2. **Creating and Using Datasets**
   - Create tabular and file datasets from datastore files.
   - Register datasets in the workspace and use them in experiments.

3. **Leveraging Cloud Compute**
   - Create and configure compute targets (e.g., AmlCompute, ComputeInstance).
   - Use compute clusters for scalable training and inference.

## Module 3: Orchestrate Pipelines and Deploy Real-Time Machine Learning Services with Azure Machine Learning

### Overview
Orchestrating machine learning training with pipelines is a key element of DevOps for machine learning. This module covers creating, publishing, and running pipelines to train models in Azure Machine Learning. You will also learn to register and deploy ML models with the Azure Machine Learning service.

### Detailed Notes
1. **Creating Machine Learning Pipelines**
   - Define pipeline steps using `PipelineStep` and `PythonScriptStep`.
   - Build and run pipelines with the `Pipeline` class.

2. **Publishing and Running Pipelines**
   - Publish pipelines to make them reusable.
   - Trigger pipeline runs manually or via schedules.

3. **Deploying Real-Time Services**
   - Register trained models and create inference configurations.
   - Deploy models as real-time web services using Azure Kubernetes Service (AKS) or Azure Container Instances (ACI).

## Module 4: Deploy Batch Inference Pipelines and Tune Hyperparameters with Azure Machine Learning

### Overview
This module focuses on deploying machine learning models for batch inference and tuning hyperparameters to optimize model performance. You will learn to use Azure Machine Learning to publish a batch inference pipeline and leverage cloud-scale experiments for hyperparameter tuning.

### Detailed Notes
1. **Deploying Batch Inference Pipelines**
   - Define batch scoring scripts and create data references for input data.
   - Use `ParallelRunStep` to define and execute batch inference pipelines.

2. **Hyperparameter Tuning**
   - Configure and run hyperparameter tuning experiments using `HyperDriveConfig`.
   - Analyze and retrieve the best model configuration from the experiments.

## Module 5: Select Models and Protect Sensitive Data

### Overview
This module covers selecting the best machine learning models using automated machine learning (AutoML) and protecting sensitive data with differential privacy. You will learn to use AutoML to find the optimal model for your data and implement differential privacy to ensure data confidentiality.

### Detailed Notes
1. **Using Automated Machine Learning (AutoML)**
   - Set up and run AutoML experiments to automate model selection.
   - Retrieve and evaluate the best model from the AutoML run.

2. **Protecting Sensitive Data**
   - Understand and implement differential privacy techniques.
   - Use libraries like PySyft and TensorFlow Privacy for privacy-preserving machine learning.

3. **Factors Influencing Model Predictions**
   - Evaluate feature importance and interpret model predictions.
   - Use techniques like SHAP values to explain individual predictions.

## Module 6: Monitor Machine Learning Deployments

### Overview
Machine learning models can encapsulate unintentional bias, resulting in unfairness. This module covers using Fairlearn and Azure Machine Learning to detect and mitigate unfairness in models. You will also learn to use telemetry to monitor model usage and data drift to ensure ongoing prediction accuracy.

### Detailed Notes
1. **Detecting and Mitigating Unfairness**
   - Use Fairlearn to assess and reduce model bias.
   - Implement fairness-aware machine learning practices.

2. **Monitoring Model Usage and Data Drift**
   - Use telemetry to track model usage and performance.
   - Monitor data drift to maintain model accuracy over time.

## Contributing
Feel free to contribute to this repository by submitting issues or pull requests. Your contributions will help enhance the learning experience for everyone.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
This repository is part of an open-source education initiative to share knowledge and resources with the community. Thank you for your interest and support!

---

Happy Learning!

