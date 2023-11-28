from openai import OpenAI

import json
import pandas as pd

from typing import List


def create_prompt(names: List[str]) -> str:
    """Add our data to the instructions for generating a dataset"""

    prompt = """create a text of 2-3 sentences with meaningful context in English which should contain the mountain names.
    create such text for each mountain name that is in this list:
    """ + f"{names}" + """
    Return the results as a Json file: {
    "mountain name 1": "generated text 1",
    "mountain name 2": "generated text 2",
    "mountain name 3": "generated text 3",
    ....
    }
    don't use the word mountains in every sentence you make!"""

    return prompt


def main():
    """
        Creates the dataset in mountain name - generated text format
        containing the names of the mountains and the text that contains these names
        from list of the mountains names
    """
    # Get users api_key
    api_key = input("Enter OpenAI api_key: ")
    # Initialize OpenAI client
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key,
    )

    # Get set of mountain name
    name_set = pd.read_csv("mountain_names_eng.csv")
    name_set = name_set.mountain_name.values

    # Initialize lists for results
    name_list = []
    text_list = []

    for bucket_id in range(0, 1301, 50):
        # Extract a subset of 20 mountain names based on the current bucket_id every 50 name
        name_subset = name_set[bucket_id:bucket_id + 20]
        # Create prompt for GPT
        prompt = create_prompt(name_subset)
        # Request a chat completion from the OpenAI API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=0.6
        )
        # Extract the content of the response
        chat_result = chat_completion.choices[0].message.content
        # Parse the response from JSON
        data_json = json.loads(chat_result)
        for name in data_json:
            name_list.append(name)
            text_list.append(data_json[name])

    # Write the collected data
    data_df = pd.DataFrame(
        {
            "peak_name": name_list,
            "text": text_list,
        }
    )
    data_df.to_csv("dataset_NER_mountain.csv", index=False)


if __name__ == "__main__":
    main()
