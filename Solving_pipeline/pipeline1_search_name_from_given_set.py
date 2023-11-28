import pandas as pd

from typing import List


def find_mount_name_p1(sentence: str, mount_name_list: List[str]) -> List[str]:
    """
    This function searches for mountain names within a given sentence.

    Parameters:
    sentence (str): The sentence in which to search for mountain names.
    mount_name_list (List[str]): A list of mountain names to search for.

    Returns:
    List[str]: A list of mountain names found in the sentence.
    """
    result_mount_name = []
    # Iterate through each mountain name in the list
    for mount_name in mount_name_list:
        # If the mountain name is found in the sentence, add it to the result list
        if mount_name in sentence:
            result_mount_name.append(mount_name)
    return result_mount_name


def extract_entity_name_p1(sentence: str) -> List[str]:
    """
    Extracts mountain names from a sentence using a predefined list of names.

    Parameters:
    sentence (str): The sentence from which to extract mountain names.

    Returns:
    List[str]: A list of extracted mountain names.
    """
    mount_name_list = pd.read_csv("mountain_names_eng.csv")
    mount_name_list = mount_name_list.mountain_name.values
    result = find_mount_name_p1(sentence, mount_name_list)
    return result


def main_p1():
    # Prompt the user to input a sentence
    sentence = input("Enter sentence with name of the Ukrainian Carpathians peak: ")
    # Extract mountain names from the sentence
    result = extract_entity_name_p1(sentence)
    # Print results
    print(f"Found {len(result)} mountain peaks:")
    for mount_name in result:
        print(mount_name)


if __name__ == "__main__":
    main_p1()
