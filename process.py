import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import BertTokenizer, TFBertForSequenceClassification
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

bert_tokenizer = BertTokenizer.from_pretrained('E:\jupyter\internship assesment\Techdome\Tokenizer')
 
# Load model
bert_model = TFBertForSequenceClassification.from_pretrained('E:\jupyter\internship assesment\Techdome\Model')
label = {
	1: 'positive',
	0: 'Negative'
}

def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model):
	# Convert Review to a list if it's not already a list
	if not isinstance(Review, list):
		Review = [Review]

	Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
																			padding=True,
																			truncation=True,
																			max_length=128,
																			return_tensors='tf').values()
	prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])

	# Use argmax along the appropriate axis to get the predicted labels
	pred_labels = tf.argmax(prediction.logits, axis=1)

	# Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
	pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
	return pred_labels