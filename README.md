 # FastAPI Text Classification Project

This project focuses on text classification using the DistilBERT model with a FastAPI backend and a simple HTML frontend.

## Note:
 A detailed project documentation is available in the [Documentation.pdf](Documentation.pdf) file, providing in-depth insights into the project and the thought process.

The project includes a pre-trained DistilBERT model (distilbert_model_low_memory.joblib), but the data is not published due to restrictions.


## Project Structure

- **Preprocessing:** The `preprocessing.py` file covers data cleaning and preprocessing steps to prepare the dataset for training.

- **Model Training:** The `BARTmodel.py` file contains the code for training the DistilBERT model on a dataset, and the resulting model and label encoder are saved.

- **API Code:** The FastAPI application is defined in the `main.py` file. It loads the pre-trained DistilBERT model and provides endpoints for predictions.

- **Testing:** The `testing.py` file performs testing on the test data and generates a classification report.

- **Test Cases:** The `test_cases.py` file contains test cases using FastAPI's TestClient for basic endpoint checks and handling of different input scenarios.

## Instructions to Run

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/AfaqJ/MuhJam364.git
    cd <repository-directory>
    ```

2. **Build and Run Docker Container:**
    ```bash
    docker build -t fastapi-app .
    docker run -p 8000:8000 fastapi-app
    ```

3. **Access the Frontend:**
   Open the HTML code in the `static` folder to interact with the frontend.


