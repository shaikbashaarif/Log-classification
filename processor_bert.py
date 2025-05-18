
from sentence_transformers import SentenceTransformer
import joblib

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier_model = joblib.load(r'D:\project\log_classification\training\dataset\models\logistic_regression_model.pkl')

def classify_with_bert(log_message):
    
    message_embedding = transformer_model.encode(log_message)
    probabilities = classifier_model.predict_proba([message_embedding])[0]
    
    if max(probabilities[0]) < 0.5:
        return "Unclassified"
        predicted_class = classifier_model.predict([message_embedding])[0]
    predicted_label = classifier_model.predict([message_embedding])[0]

    return predicted_label


if __name__ == "__main__":
    logs =[
        "alpha.osapi_compute.1: 2023-10-01 12:00:00 - User User123 logged in",
    ]