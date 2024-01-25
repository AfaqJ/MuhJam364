import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel

# Load BERT tokenizer and model for German
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-uncased")

# Load the dataset
data = pd.read_csv(r"Data/sample_data.csv")

# Remove rows with numbers or empty text
data = data[(data['text'].apply(lambda x: not (x.isnumeric() or x.strip() == ''))) & (data['label'].notna())]

# Variables to keep track of zero vectors
total_vectors = 0
zero_vectors = 0

# Tokenization using BERT tokenizer and handling OOV words
def tokenize_with_vectors(text):
    global total_vectors, zero_vectors
    tokens = []
    # Tokenize the text using BERT tokenizer
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # Extract the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    for embedding in embeddings:
        if embedding.all() != 0.0:
            total_vectors += 1
            tokens.append(embedding)
        else:
            # Handling OOV words: Use a zero vector for OOV words
            zero_vectors += 1
            tokens.append([0.0] * len(embedding))
    return tokens

# Apply tokenization with vectors to the text column
data['word_vectors'] = data['text'].apply(tokenize_with_vectors)

# Label Encoding
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['label'])

# Data Splitting
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the preprocessed data
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Print statistics about zero vectors
print("Total vectors:", total_vectors)
print("Zero vectors:", zero_vectors)
print("Percentage of zero vectors:", (zero_vectors / total_vectors) * 100)
