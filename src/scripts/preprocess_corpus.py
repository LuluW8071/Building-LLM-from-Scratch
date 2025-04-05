import os
import argparse
import re 


def clean_text(text):
    """ Fix paragraph breaks """
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)    # Remove newlines within a paragraph (make it a single line)
    text = re.sub(r'\n{2,}', '\n\n', text)          # Keep double newlines for paragraph separation
    text = re.sub(r'[ \t]+', ' ', text)             # Remove extra spaces
    return text


def preprocess_books(input_folder, output_folder):
    """ Process all .txt files in input_folder, clean text, and save in output_folder. """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_files = len([f for f in os.listdir(input_folder) if f.endswith(".txt")])
    processed_files = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", encoding="utf-8") as infile:
                text = infile.read()

            cleaned_text = clean_text(text)

            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(cleaned_text)

            processed_files += 1
            print(f"Processed {processed_files}/{total_files}: {filename}", end='\r', flush=True)

    print("\nPreprocess Completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Books")

    parser.add_argument('-i', '--input_folder', required=True, type=str, help='Folder of .txt books corpus')
    args = parser.parse_args()
    preprocess_books(args.input_folder, output_folder="preprocessed_books")