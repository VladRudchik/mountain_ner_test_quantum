from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from typing import List


def convert_answer_to_words_p4(sentence: str, ner_results: List[dict]) -> List[str]:
    """
      Reconstructs words from token information provided by NER results.

      This function iterates through tokens identified by the NER model and reconstructs
      the original words based on their start and end positions in the sentence.

      Parameters:
      sentence (str): The original sentence.
      ner_results (List[dict]): The results from NER model, containing identified tokens with their positions.

      Returns:
      List[str]: A list of reconstructed words from the sentence.
    """
    words = []
    current_word = ""
    last_end = -1
    # Iterate through each token information provided by NER
    for token_info in ner_results:
        start, end = token_info['start'], token_info['end']

        # Check if the current token follows immediately after the previous token
        if start == last_end:
            current_word += sentence[start:end]
        elif start == last_end + 1:
            # If the difference in indices is one, it represents a space
            current_word += ' ' + sentence[start:end]
        else:
            # If there is a separate word, add it to the list
            if current_word:
                words.append(current_word.strip())
            current_word = sentence[start:end]

        last_end = end

    # Add the last collected word if it exists
    if current_word:
        words.append(current_word.strip())

    return words


def extract_entity_name_p4(sentence: str) -> List[str]:
    """
        Extracts entity names from a sentence using a pretrained NER model.

        Parameters:
        sentence (str): The sentence from which to extract entity names.

        Returns:
        List[str]: A list of extracted entity names.
    """
    # Load a pretrained tokenizer and model for token classification
    tokenizer = AutoTokenizer.from_pretrained("ruba12/mountain_ner_test_quantum")
    model = AutoModelForTokenClassification.from_pretrained("ruba12/mountain_ner_test_quantum")
    # Create a NER pipeline using the loaded model and tokenizer
    ner_cls = pipeline("ner", model=model, tokenizer=tokenizer)
    # Get NER results from the sentence
    ner_results = ner_cls(sentence)
    # Convert NER results to a list of words
    result = convert_answer_to_words_p4(sentence, ner_results)
    return result


def main_p4():
    # Prompt the user to input a sentence
    sentence = input("Enter sentence with name of the Ukrainian Carpathians peak: ")
    # Extract mountain names from the sentence
    result = extract_entity_name_p4(sentence)
    # Print results
    print(f"Found {len(result)} locations:")
    for mount_name in result:
        print(mount_name)


if __name__ == "__main__":
    main_p4()
