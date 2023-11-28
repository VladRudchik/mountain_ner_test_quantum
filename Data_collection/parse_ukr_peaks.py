import requests
from bs4 import BeautifulSoup

import pandas as pd


def main():
    """
        Parse the site with the names of most Ukrainian peaks and mountains
        Site link: https://mountain.land.kiev.ua/list.html
    """
    # URL of the webpage to scrape
    url = 'https://mountain.land.kiev.ua/list.html'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "lxml")

        # Find table rows in page content
        table = soup.find("table", class_="w100")
        table_rows = table.find("tbody").find_all("tr")

        # Extract the mountain name text from table
        mountain_names = []
        for row in table_rows:
            mount_name = row.find_all("td")[1].text
            mountain_names.append(mount_name)

        # Create a DataFrame from the collected mountain names
        result_dataset = pd.DataFrame(mountain_names, columns=['mountain_name'])
        result_dataset = result_dataset[result_dataset.mountain_name.apply(lambda x: "висота" not in x.lower())]
        result_dataset.to_csv("mountain_names_ukr.csv", index=False)
    else:
        print(f"Failed to retrieve content: Status code {response.status_code}")


if __name__ == "__main__":
    main()
