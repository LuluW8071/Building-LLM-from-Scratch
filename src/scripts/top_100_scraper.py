""" Scrape and download top 100 books from www.gutenberg.org """

import requests
from bs4 import BeautifulSoup
import re
import os


def ebooks_list_scraper(url="https://www.gutenberg.org/browse/scores/top"):
    response = requests.get(url)

    # Raise an exception for bad status codes
    response.raise_for_status()

    # Get the HTML contents from url response
    soup = BeautifulSoup(response.content, "html.parser")

    # ==== View HTML Content if you to know how it works ====
    # with open("url_reponse.html", "w", encoding="utf-8") as f:
    #     f.write(str(soup))

    # ==== Finding: <h2 id="books-last1"> ====
    top_books_heading = soup.find("h2", id="books-last1")
    if not top_books_heading:
        print("Couldn't find the 'books-last1' section in HTML")
        return []

    # ==== Getting <ol> tag element inside <h2 id="books-last1"> ====
    ol_element = top_books_heading.find_next_sibling("ol")
    if not ol_element:
        print("Could not find the ordered list after 'books-last1'")
        return []

    # ==== Find all <li> listed items of top book and fetch ids ==== 
    list_items = ol_element.find_all("li")
    return list_items


def process_listed_items(list_item):
    """ Process the first 100 books """
    # ==== Get the <a href> contents ====
    a_tag = list_item.find("a", href=True)
    # print(a_tag)
    if a_tag and "/ebooks/" in a_tag["href"]:
        ebook_id_match = re.search(r'/ebooks/(\d+)', a_tag['href'])
        title = a_tag.get_text(strip=True)

        if ebook_id_match:
            ebook_id = ebook_id_match.group(1)
            return ebook_id, title
    return None


def download_top_100_books():
    try:
        list_items = ebooks_list_scraper()
        downloaded_count = 0

        output_dir = "gutenberg_corpus"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, item in enumerate(list_items[:100], start=1):
            result = process_listed_items(item)
            if not result:
                continue

            ebook_id, title = result
            download_url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
            filename = f"{''.join(c if c.isalnum() or c.isspace() else '_' for c in title)}.txt"
            filepath = os.path.join(output_dir, filename)
            try:
                # print(f"Downloading: {title} (ID: {ebook_id}) from {download_url}")
                download_response = requests.get(download_url, stream=True)
                download_response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # ==== Print Download progress ====
                print(f"\r[{downloaded_count + 1:>3}/100] Downloaded: {title[:50]:.<60}", end='', flush=True)
                downloaded_count += 1

            except requests.exceptions.RequestException as e:
                print(f"\nError downloading {title} (ID: {ebook_id}): {e}")
            except Exception as e:
                print(f"\nAn unexpected error occurred while processing {title} (ID: {ebook_id}): {e}")

        print(f"\n\nSuccessfully downloaded {downloaded_count} books to the '{output_dir}' folder.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the Gutenberg top books page: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    download_top_100_books()
