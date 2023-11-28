# Task 1. Natural Language Processing. Named entity recognition

## Section 0: Preface

You can find the solution explanation and potential improvement in file Mountain_NER.ipynb.

Link to my fine-tuned NER model: https://huggingface.co/ruba12/mountain_ner_test_quantum

## Section 1: Introduction

In this section, we provide a brief overview of our project, which is focused on solving a Named Entity Recognition (NER) problem related to recognizing mountain names in texts.

### Problem Statement

Our goal was to address the challenge of NER by:

1. **Dataset Preparation**:
    - We prepared a dataset by scraping a portion of data from a website that contained the names of mountains in Ukraine.
    - Utilizing the GPT-API, we created a dataset in the format of 'mountain name' - 'text containing that mountain name'.

2. **Solution Approaches**:
    - We proposed several pipelines for solutions, four in total.
        - **First Pipeline**: Based on the knowledge of the region in which the mountains are located, we aimed to identify all mountains in that region. We then scanned the entire text to find all mountain names present in this region.
        - **Second Pipeline**: We employed a pre-trained NER model from Hugging Face.
        - **Third Pipeline**: Similarly, we used a Question-Answering model to address this task.
        - **Fourth Pipeline**: We fine-tuned a NER model from Hugging Face on our dataset.

3. **Comparison of results**:
   - Compared each of the pipelines using test Data Set and described the advantages and problems of each pipeline.

## Section 2: Project Structure

The project is organized into several directories and files that serve distinct purposes within the workflow:

- **Data_collection**:
   - `parse_ukr_peaks.py`: A parser script for extracting Ukrainian mountain names from https://mountain.land.kiev.ua/list.html.
   - `create_dataset_by_gpt.py`: A script used to generate the dataset of pair (mountain_name - text with mountain_name) from mountain_names_ukr.csv with the help of OpenAI-API.
   - `dataset_NER_mountain.csv`: The dataset containing pair (mountain_name - text with mountain_name) for NER.
   - `mountain_names_ukr.csv`: A list of mountain names in Ukrainian from parser.
   - `mountain_names_eng.csv`: A list of mountain names in English. I translated `mountain_names_ukr.csv` using the web version of the translator.

- **Fine_tune_NER**:
   - `my_model`: A directory for the custom model.
   - `fine_tune_huggingface_NER_model.py`: The script for fine-tuning the Hugging Face NER model.
   - `dataset_NER_mountain.csv`: The dataset containing pair (mountain_name - text with mountain_name) for NER.

- **Solving_pipeline**:
   - `mountain_names_eng.csv`: A list of mountain names in English. I translated `mountain_names_ukr.csv` using the web version of the translator.
   - `pipeline1_search_name_from_given_set.py`: Script for searching names based on `mountain_names_ukr.csv` set, the first solution approach.
   - `pipeline2_NER_from_huggingface.py`: Script for the NER model from Hugging Face, the second solution approach.
   - `pipeline3_QA_from_huggingface.py`: Script utilizing a Question-Answering model from Hugging Face for extracting mountain names, the third solution approach.
   - `pipeline4_fine_tuned_NER.py`: script for using our fine-tuned model, the fourth solution approach.

- **Root Directory**:
   - `readme.md`: The Markdown file that provides an overview and instructions for the project.
   - `requirements.txt`: A list of dependencies required to run the project.
   - `Mountain_NER.ipynb`: Jupyter Notebook with a detailed solving problem explanation.


## Section 3: Environment Setup

To run the project, an isolated environment is recommended to manage dependencies and ensure consistency. Here's how to set up the Anaconda environment for this project:

1. **Create a New Conda Environment**:
   - Open the Anaconda Command Prompt.
   - Use the command `conda create --name myenv` (replace `myenv` with your desired environment name).
   - Activate the new environment using `conda activate myenv`.

2. **Install Dependencies**:
   - Navigate to the root project directory.
   - Run `pip install -r requirements.txt` to install the necessary packages.

Once the installation is complete, the environment is ready. You can now execute any script from the project within this Conda environment.