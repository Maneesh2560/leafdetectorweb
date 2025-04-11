# Potato Disease Predictor
[Google Collab Link](https://colab.research.google.com/drive/1O4xAnlSvxASwV7lrR2mVTNOA-W73Dtvf?usp=sharing)
## Project Overview
The Potato Disease Predictor is a web application designed to detect diseases in potato plants based on images. The application leverages a machine learning model trained on a dataset of potato plant images. The frontend is built using React with Vite, and the backend is served using FastAPI. TensorFlow Serving is used for serving the trained models. Also the model was deployed on AWS. 

![Screenshot (63)](https://github.com/user-attachments/assets/4f283228-0413-436e-b1f5-b18d733e68bf)

![Screenshot (65)](https://github.com/user-attachments/assets/2633c439-5e02-4161-a524-d2c914047984)

![Screenshot (67)](https://github.com/user-attachments/assets/ebb32212-e67f-486f-96c6-eaa55fd025b3)

![Screenshot (68)](https://github.com/user-attachments/assets/f1abd2b1-8dd5-451e-866c-4d4efade20fb)

![Screenshot (70)](https://github.com/user-attachments/assets/1999eb8c-c11e-45bc-94c2-6e32c7e0dd86)

![Screenshot (69)](https://github.com/user-attachments/assets/0797a665-7f5c-465a-829e-f2d71c463d04)


## Running the Project
### Prerequisites
- Node.js and npm installed
- Python 3.8+ installed
- Jupyter Notebook
- TensorFlow and Keras
- FastAPI
- TensorFlow Serving
- AWS CLI

### Frontend: React (Vite)
To run the frontend of the Potato Disease Predictor:

1. Navigate to the frontend directory:
   ```bash
   cd Potato-disease-detector
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

The React app will be running at `http://localhost:5173`.

### Backend: FastAPI

#### Model Training
1. Before running the backend, you need to train the model. Run the `model_training.ipynb` Jupyter notebook to train the model on the dataset.

2. Download the dataset from the following Google Drive link:
   [Potato Disease Dataset](https://drive.google.com/drive/folders/1rYjnTLbCm__gy14ctpYm_RckX2gkZo_3?usp=sharing).

3. After training, save the model in both `SavedModel` format and Keras format in the following directories:
   - **SavedModel format:** `models/`
   - **Keras format:** `models_served/`

#### Running the Backend
1. Navigate to the backend directory where the API is located.

2. Ensure the models are correctly placed in the `models/` and `models_served/` directories.

3. Configure the `models.config` file for TensorFlow Serving. The configuration should specify the paths to the models in the `models_served/` directory.

4. Start the FastAPI backend:
   ```bash
   python api/main.py
   ```

The FastAPI backend will be running at `http://localhost:8000`.

## AWS Deployment

### Overview
The trained machine learning model has been deployed on AWS to allow for scalable inference. This section covers the steps to deploy the model using Amazon SageMaker and other AWS services.

### Steps for AWS Deployment

1. **Convert the Model to tar.gz Format:**
   - After training the model and saving it in the `SavedModel` format, compress the model files into a `tar.gz` archive.

   ```bash
   tar -czvf model.tar.gz -C path_to_savedmodel .
   ```

2. **Upload to S3 Bucket:**
   - Upload the `tar.gz` file to an S3 bucket in your AWS account. Ensure that the S3 bucket has the appropriate permissions for access by SageMaker.

3. **IAM User and Role Setup:**
   - Define a user on IAM with the necessary permissions to interact with SageMaker, S3, and other required AWS services.
   - Create a role with the required policies that SageMaker can assume to access the model in the S3 bucket.

4. **Deploy the Model Using SageMaker:**
   - Navigate to the `aws/` directory and run the deployment script using the following command:

   ```bash
   python deploy.py
   ```

   - The script uses the Boto3 library to interact with SageMaker and deploy the model. Ensure that AWS CLI is configured with the correct credentials and region.


### Requirements
- **AWS CLI:** Ensure AWS CLI is installed and configured with your AWS account credentials.
- **Boto3:** The Boto3 library is required for interacting with AWS services programmatically.
- **SageMaker Permissions:** Ensure the IAM user and roles have the necessary permissions for SageMaker, S3, and other AWS services involved in the deployment.

## Additional Notes
- Ensure TensorFlow Serving is correctly set up and running to serve the models.
- Adjust the `models.config` file according to the specific environment and TensorFlow Serving setup.
- The trained model must be placed in the correct format and directories for the backend to function properly.
