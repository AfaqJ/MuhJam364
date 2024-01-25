import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Loading the cleaned dataset
cleaned_data = pd.read_csv('Data/cleaned_data.csv')


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
cleaned_data['encoded_label'] = label_encoder.fit_transform(cleaned_data['label'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_data['text'],
    cleaned_data['encoded_label'],
    test_size=0.2,
    random_state=42
)
# Saving data files for future use (Although train data file isn't being used anywhere i saved it just in case)
train_data = pd.DataFrame({'text': X_train, 'encoded_label': y_train})
train_data.to_csv('train_data.csv', index=False)

test_data = pd.DataFrame({'text': X_test, 'encoded_label': y_test})
test_data.to_csv('test_data.csv', index=False)

# loading the pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-german-cased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# setting the number of classes according to our task
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=6)

max_seq_length = 64  # giving sequence length to avoid extra load on computation
X_train_encoded = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt', max_length=max_seq_length)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)



batch_size = 4  # Adjust based on available memory
train_dataset = TensorDataset(X_train_encoded['input_ids'], X_train_encoded['attention_mask'], y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

accumulation_steps = 2  # Accumulate gradients over 2 small batches before performing a backward pass


optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 1  #epochs can be set here

for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}')):
        optimizer.zero_grad()
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
            optimizer.step()

# Save the trained model and label encoder
joblib.dump(model, 'distilbert_model_low_memory.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')


