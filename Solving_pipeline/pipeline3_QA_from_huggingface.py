from transformers import pipeline


def extract_entity_name_p3(sentence: str) -> str:
    """
        Extracts peak names from a sentence using a question-answering model.

        Parameters:
        sentence (str): The sentence from which to extract peak names.

        Returns:
        str: The extracted peak name(s) from the sentence.
    """
    model_name = "deepset/roberta-base-squad2"
    # Initialize a pipeline for question-answering
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    # Formulate the question-answering input with the question and the provided sentence as context
    qa_input = {
        'question': 'What peak names were used in this text?',
        'context': sentence,
    }
    # Get the answer from the model
    res = nlp(qa_input)
    # Return model answer
    return res["answer"]


def main_p3():
    # Prompt the user to input a sentence
    sentence = input("Enter sentence with name of the Ukrainian Carpathians peak: ")
    # Extract mountain names from the sentence
    result = extract_entity_name_p3(sentence)
    # Print results
    print(result)


if __name__ == "__main__":
    main_p3()
