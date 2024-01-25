import pandas as pd


data = pd.read_csv(r"Data/sample_data.csv")

# Function to check if the text contains Chinese characters
def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

# Removing rows with empty text, empty label,numbers in the text or Chinese words
cleaned_data = data[(data['text'].apply(lambda x: not (str(x).strip() == '' or x.isnumeric()))) &
                    (data['label'].notna()) & (data['label'].apply(lambda x: str(x).strip() != '')) &
                    (~data['text'].apply(contains_chinese))]


cleaned_file_path = 'Data/cleaned_data.csv'
cleaned_data.to_csv(cleaned_file_path, index=False)


