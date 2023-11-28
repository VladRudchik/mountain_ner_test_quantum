import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
from huggingface_hub import notebook_login

from typing import List, Tuple


def separate_words_punctuation(text: str) -> List[str]:
    """
        Splits a given text into words and punctuation marks.

        Parameters:
        text (str): The text to be split.

        Returns:
        List[str]: A list of words and punctuation marks.
    """
    # Using regular expression to separate words and punctuation
    return re.findall(r"[\w']+|[.,!?;]", text)


def add_word_starts(text: str, text_list: List[str]) -> List[Tuple[str, int]]:
    """
        Finds the starting index of each word in the text.

        Parameters:
        text (str): The original text.
        text_list (List[str]): A list of words to find in the text.

        Returns:
        List[Tuple[str, int]]: A list of tuples containing words and their starting indices.
    """
    result = []
    start = 0
    for word in text_list:
        start = text.find(word, start)
        result.append((word, start))
    return result


def add_target(peak_loc: Tuple[int], text_set: List[Tuple[str, int]]) -> List[int]:
    """
        Marks words in the text set as target entities based on their location.

        Parameters:
        peak_loc (Tuple[int]): The start and end indices of the target entity in the text.
        text_set (List[Tuple[str, int]]): A list of tuples containing words and their starting indices.

        Returns:
        List[int]: A list indicating whether each word in the text set is part of the target entity.
    """
    result = [0] * len(text_set)
    for i, word in enumerate(text_set):
        if peak_loc[0] <= word[1] < peak_loc[1]:
            result[i] = 1
    return result


def markup_data_to_ner_model(data: pd.DataFrame) -> pd.DataFrame:
    """
        Processes a DataFrame to make it suitable for training a NER model.
        Added text_list and target columns:
            text_list - A list of words and punctuation marks.
            target - A list indicating whether each word in the text set is part of the target entity.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the text data.

        Returns:
        pd.DataFrame: The processed DataFrame with added columns for NER model training.
    """
    # Create a column containing text divided into individual words and punctuation murks
    data["text_list"] = data.text.apply(separate_words_punctuation)
    # Find start and end index of target sentence
    data["peak_loc"] = data.apply(lambda x: (x[1].find(x[0]), x[1].find(x[0]) + len(x[0])), axis=1)
    # Find the starting index of each word in the text.
    data["text_set"] = data.apply(lambda x: add_word_starts(x[1], x[2]), axis=1)
    # Create target markup for NER model
    data["target"] = data.apply(lambda x: add_target(x[3], x[4]), axis=1)
    # Fix mistake in one sentence
    data["target"] = data.target.apply(
        lambda x: [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if x == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else x
    )
    return data


def transform_df(data: pd.DataFrame) -> List[dict]:
    """
        Transforms a DataFrame into a format suitable for the Hugging Face datasets.

        Parameters:
        data (pd.DataFrame): The DataFrame to be transformed.

        Returns:
        List[dict]: A list of dictionaries with 'id', 'tokens', and 'ner_tags'.
    """
    result = []
    for id_, row in enumerate(data[["text_list", "target"]].values):
        result.append(
            {
                "id": id_,
                "tokens": row[0],
                "ner_tags": row[1]
            }
        )
    return result


def prepare_dataset(data_path: str = "dataset_NER_mountain.csv") -> DatasetDict:
    """
        Prepares a dataset for NER model training.

        Parameters:
        data_path (str): The path to the dataset file.

        Returns:
        DatasetDict: A dictionary of datasets for training, validation, and testing.
    """
    # Load and process the data
    data = pd.read_csv(data_path)
    data = markup_data_to_ner_model(data)

    # Split the data into training, validation, and testing sets
    data_train, data_test = train_test_split(data, test_size=0.2)
    data_train, data_valid = train_test_split(data_train, test_size=0.25)

    # Transform the DataFrames to dicts
    data_train_dict = transform_df(data_train)
    data_test_dict = transform_df(data_test)
    data_valid_dict = transform_df(data_valid)

    # Convert the dictionaries into HuggingFace datasets
    train_dataset = Dataset.from_list(data_train_dict)
    test_dataset = Dataset.from_list(data_test_dict)
    valid_dataset = Dataset.from_list(data_valid_dict)

    # Create a DatasetDict to hold the datasets
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset,
    })
    return dataset_dict


def tokenize_and_align_tags(records, tokenizer):
    """
    Transfer word splitting and markup to new tokenizer

    source: https://github.com/anyuanay/medium/blob/main/src/working_huggingface/Working_with_HuggingFace_ch2_Fine_Tuning_NER_Model.ipynb
    """
    # Tokenize the input words. This will break words into subtokens if necessary.
    # For instance, "ChatGPT" might become ["Chat", "##G", "##PT"].
    tokenized_results = tokenizer(records["tokens"], truncation=True, is_split_into_words=True)

    input_tags_list = []

    # Iterate through each set of tags in the records.
    for i, given_tags in enumerate(records["ner_tags"]):
        # Get the word IDs corresponding to each token. This tells us to which original word each token corresponds.
        word_ids = tokenized_results.word_ids(batch_index=i)

        previous_word_id = None
        input_tags = []

        # For each token, determine which tag it should get.
        for wid in word_ids:
            # If the token does not correspond to any word (e.g., it's a special token), set its tag to -100.
            if wid is None:
                input_tags.append(-100)
            # If the token corresponds to a new word, use the tag for that word.
            elif wid != previous_word_id:
                input_tags.append(given_tags[wid])
            # If the token is a subtoken (i.e., part of a word we've already tagged), set its tag to -100.
            else:
                input_tags.append(-100)
            previous_word_id = wid

        input_tags_list.append(input_tags)

    # Add the assigned tags to the tokenized results.
    # Hagging Face trasformers use 'labels' parameter in a dataset to compute losses.
    tokenized_results["labels"] = input_tags_list

    return tokenized_results


def compute_metrics(p, tag_names, seqeval):
    """
    Calculate metrics on the validation dataset during model training

    source: https://github.com/anyuanay/medium/blob/main/src/working_huggingface/Working_with_HuggingFace_ch2_Fine_Tuning_NER_Model.ipynb
    """
    # p is the results containing a list of predictions and a list of labels
    # Unpack the predictions and true labels from the input tuple 'p'.
    predictions_list, labels_list = p

    # Convert the raw prediction scores into tag indices by selecting the tag with the highest score for each token.
    predictions_list = np.argmax(predictions_list, axis=2)

    # Filter out the '-100' labels that were used to ignore certain tokens (like sub-tokens or special tokens).
    # Convert the numeric tags in 'predictions' and 'labels' back to their string representation using 'tag_names'.
    # Only consider tokens that have tags different from '-100'.
    true_predictions = [
        [tag_names[p] for (p, l) in zip(predictions, labels) if l != -100]
        for predictions, labels in zip(predictions_list, labels_list)
    ]
    true_tags = [
        [tag_names[l] for (p, l) in zip(predictions, labels) if l != -100]
        for predictions, labels in zip(predictions_list, labels_list)
    ]

    # Evaluate the predictions using the 'seqeval' library, which is commonly used for sequence labeling tasks like NER.
    # This provides metrics like precision, recall, and F1 score for sequence labeling tasks.
    results = seqeval.compute(predictions=true_predictions, references=true_tags)

    # Return the evaluated metrics as a dictionary.
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train_pipeline(output_dir: str = "my_model", push_to_hub: bool = False):
    """
        Sets up and executes the training pipeline for a Named Entity Recognition (NER) model.

        This function prepares the dataset, configures the tokenizer, model, and training parameters,
        and then trains the model using the Trainer API of Hugging Face Transformers. It also supports
        pushing the trained model to the Hugging Face Hub if desired.

        Parameters:
        output_dir (str): The directory to save the trained model.
        push_to_hub (bool): Flag to determine whether to push the trained model to the Hugging Face Hub.
    """
    # Prepare the dataset for training
    dataset_dict = prepare_dataset("dataset_NER_mountain.csv")
    # Define tag names for NER
    tag_names = ["O", "Peak"]

    # Initialize the tokenizer with a pretrained model
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    # Tokenize the dataset and align the tags
    tokenized_dataset_dict = dataset_dict.map(
        lambda records: tokenize_and_align_tags(records, tokenizer=tokenizer),
        batched=True
    )

    # Create a data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create dictionaries to map IDs to labels and vice versa
    id2label = dict(enumerate(tag_names))
    label2id = dict(zip(id2label.values(), id2label.keys()))

    # Initialize the model for token classification
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Load the seqeval metric for evaluation
    seqeval = evaluate.load("seqeval")

    # Login to Hugging Face Hub if pushing the model to the hub
    if push_to_hub:
        notebook_login()

    # Set training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=push_to_hub,
    )

    # Initialize the Trainer with the model, training arguments, dataset, tokenizer, and metrics computation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_dict["train"],
        eval_dataset=tokenized_dataset_dict["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tag_names, seqeval),
    )
    # Start the training process
    trainer.train()

    # Push the model to Hugging Face Hub if enabled
    if push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    user_output_dir = "my_model"
    user_push_to_hub = False
    # Run train pipeline
    train_pipeline(user_output_dir, user_push_to_hub)
