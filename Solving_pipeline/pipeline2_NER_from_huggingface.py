from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from typing import List


def convert_answer_to_words_p2(sentence: str, ner_results: List[dict]):
    """
       Converts NER results to a list of locations word.

       Parameters:
       sentence (str): The original sentence.
       ner_results (List[dict]): The results from NER model, containing identified entities.

       Returns:
       List[str]: A list of location names extracted from the sentence.
   """
    result = []
    current_entity = ""
    for entity in ner_results:
        # Check for the beginning of a location word
        if entity["entity"] == "B-LOC":
            # If a word already exists, add it to the result, and stand new word
            if current_entity:
                result.append(current_entity)
            current_entity = sentence[entity["start"]:entity["end"]]
        # If this is a continuation of the current word then add it to current word
        elif entity["entity"] == "I-LOC" and current_entity:
            if "#" not in entity["word"]:
                current_entity += " "
            current_entity += sentence[entity["start"]:entity["end"]]
    # Add the last found word to the result list
    if current_entity:
        result.append(current_entity)
    return result


def extract_entity_name_p2(sentence: str) -> List[str]:
    """
       Extracts location names from a sentence using a pretrained NER model.

       Parameters:
       sentence (str): The sentence from which to extract location names.

       Returns:
       List[str]: A list of extracted location names.
   """
    # Load a pretrained tokenizer and model for token classification
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    # Create a NER pipeline using the loaded model and tokenizer
    ner_cls = pipeline("ner", model=model, tokenizer=tokenizer)
    # Get NER results from the sentence
    ner_results = ner_cls(sentence)
    # Convert NER results to a list of words
    result = convert_answer_to_words_p2(sentence, ner_results)
    return result


def main_p2():
    # Prompt the user to input a sentence
    sentence = input("Enter sentence with name of the Ukrainian Carpathians peak: ")
    # Extract mountain names from the sentence
    result = extract_entity_name_p2(sentence)
    # Print results
    print(f"Found {len(result)} locations:")
    for mount_name in result:
        print(mount_name)


if __name__ == "__main__":
    main_p2()
