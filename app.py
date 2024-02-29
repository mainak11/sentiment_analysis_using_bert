import streamlit as st
from process import preprocess_text, Get_sentiment

def analyze_sentiment(text):
    text = preprocess_text(text)
# print(Review)
    result = Get_sentiment(text)
    return result


def main():
    st.title("Sentiment Analysis App")
    st.write("Enter text below for sentiment analysis:")

    # Text input
    text_input = st.text_area("Input Text")

    # Button to trigger sentiment analysis
    if st.button("Analyze"):
        if text_input:
            sentiment = analyze_sentiment(text_input)
            st.write("Sentiment:", sentiment[0])
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
