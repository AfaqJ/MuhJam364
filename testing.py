import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


test_data = pd.read_csv('Data/test_data.csv')  # Replace with the actual filename

# Loading the fine-tuned saved model and label encoder for testing
loaded_model = joblib.load('distilbert_model_low_memory.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Tokenize and encode the new test data iteratively
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
max_seq_length = 64
batch_size = 4

predictions = []
actual_labels = []
for i in range(0, len(test_data), batch_size):
    batch_text = list(test_data['text'][i:i + batch_size])
    new_test_encoded = tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_length)

    # Inference using the loaded model
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(**new_test_encoded)
        batch_predictions = torch.argmax(outputs.logits, dim=1).numpy()
        predictions.extend(batch_predictions)
        actual_labels.extend(list(test_data['encoded_label'][i:i + batch_size]))

# Decoding predictions
decoded_predictions = loaded_label_encoder.inverse_transform(predictions)
decoded_actual_labels = loaded_label_encoder.inverse_transform(actual_labels)

# Adding predictions and actual labels to the test data
test_data['predicted_label'] = decoded_predictions
test_data['actual_label'] = decoded_actual_labels

# Saving the test data with predictions and actual labels to a separate file
test_data.to_csv('predictions.csv', index=False)

# Calculating metrics
accuracy = accuracy_score(test_data['actual_label'], test_data['predicted_label'])
precision = precision_score(test_data['actual_label'], test_data['predicted_label'], average='weighted')
recall = recall_score(test_data['actual_label'], test_data['predicted_label'], average='weighted')
f1 = f1_score(test_data['actual_label'], test_data['predicted_label'], average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# classification report
print('Classification Report:')
print(classification_report(test_data['actual_label'], test_data['predicted_label']))
