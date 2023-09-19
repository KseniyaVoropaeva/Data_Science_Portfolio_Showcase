NLP - Positive and negative reviews classification

Project Description.

This project is aimed at analyzing user reviews using machine learning methods. The main task is to determine the mood or emotional state expressed in each review. To achieve this goal, we used an ensemble of Random Forest methods.

Task description.

1. Collect and clean the data.
I downloaded and cleaned a dataset containing user reviews using the pandas library. To process the text data, we used the clean_text function, which included removing punctuation and question marks, converting text to lowercase, and other operations.
2. Model analysis: The detailed data analysis and model selection process can be found in the file analysis_model_selection.ipynb. Please note that some libraries may be optional, so they may not be in the requirements.txt, as data analysis is not a major part of the project.
3.Text vectorization. I used TfidfVectorizer to convert the text to a numerical format that can be used by the machine learning model.
4. Model selection.The file analysis_model_selection.ipynb also contains the training of 6 models and the decision to select a specific model for further work.
5. Building the model. I used a Random Forest ensemble to analyze the sentiment of the text. I also used GridSearchCV to select the best hyperparameters for the model.
6. Validation and evaluation. To evaluate the accuracy of the model, we used various metrics such as balanced accuracy, precision, recall, and f1-score. The results were saved for further analysis.
7. Saving the model. The trained model was saved for further predictions on the test data


requirements.txt

The `requirements.txt` file contains a list of all the Python packages needed to execute scripts in this project. You can install them using the command: pip install -r requirements.txt

Instructions on how to run the project through inference.py.

You can run `inference.py` by passing the name of the input and output files through the command line arguments: python inference.py input_file.csv output_file.csv

Opportunities for improvement

It is important to note that the accuracy of the model is not perfect. This may be due to limited resources, limited training data, or other factors. I would like to emphasize that this model is an initial one and can be improved in the future.
There are several opportunities to improve this project:

1. Increase the amount of available data to train the model.
2. Use more sophisticated models or improve the current model.
3. Use third-party datasets for pre-training


Project structure

labels.csv: File with mood labels.
reviews.csv: File with text reviews.
data_preprocessing.py: Module for data pre-processing.
training.py: A script to train the model and find optimal hyperparameters.
inference.py: A script for making predictions on validation data.
analysis_model_selection.ipynb: Script for analyzing data and selecting a model for further work
requirements.txt: A file with a list of required libraries and dependencies.
README.md: This file contains a general description of the project and instructions for use.