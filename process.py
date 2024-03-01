import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import BertTokenizer, TFBertForSequenceClassification, TextClassificationPipeline
import tensorflow as tf

# Download NLTK resources (one-time step)
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

# Function to preprocess text
def preprocess_text(text):
    text = str(text)
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in punctuations]
    # Reconstruct the text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

bert_tokenizer = BertTokenizer.from_pretrained('mainakhf/bert-base-uncased-sentiment-analysis')
 
# Load model
bert_model = TFBertForSequenceClassification.from_pretrained('mainakhf/bert-base-uncased-sentiment-analysis')


def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model):
	# Convert Review to a list if it's not already a list
	if not isinstance(Review, list):
		Review = [Review]
	model = bert_model
	model.config.id2label = {0: "Negative", 1: "Positive"} 
	tokenizer = bert_tokenizer
	pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
	pred_labels=pipe(Review)
	return [pred_labels[0]['label']]