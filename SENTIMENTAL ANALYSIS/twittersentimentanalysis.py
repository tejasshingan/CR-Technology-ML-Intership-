import re
import pandas as pd
import numpy as np
import streamlit as st
import nltk
from PIL import Image
import requests
from wordcloud import ImageColorGenerator, WordCloud
import seaborn as sns
import matplotlib.pyplot as plt


# Function to remove patterns from text
def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text


# Function to perform sentiment analysis and generate visualizations
def perform_sentiment_analysis(data, title):
    st.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
    if "sentiment" in data.columns:
        positive_tweets = data[data["sentiment"] == "positive"]["text"]
        negative_tweets = data[data["sentiment"] == "negative"]["text"]
    else:
        st.warning(
            "No 'sentiment' column found in the uploaded data. Unable to perform sentiment analysis."
        )
        return
    data["text"] = data["text"].astype(str)
    data["Cleaned_Tweets"] = np.vectorize(remove_pattern)(data["text"], "@[\w]*")
    data["Cleaned_Tweets"] = data["Cleaned_Tweets"].str.replace(
        "[^a-zA-Z#]", " ", regex=True
    )
    data["Cleaned_Tweets"] = data["Cleaned_Tweets"].apply(
        lambda x: " ".join([w for w in x.split() if len(w) > 3])
    )
    tokenized_tweets = data["Cleaned_Tweets"].apply(lambda x: x.split())
    ps = nltk.PorterStemmer()
    tokenized_tweets = tokenized_tweets.apply(lambda x: [ps.stem(i) for i in x])
    for i in range(len(tokenized_tweets)):
        tokenized_tweets[i] = " ".join(tokenized_tweets[i])
    data["Clean_Tweets"] = tokenized_tweets

    # Generate Word Clouds (same steps as before)
    positive_words = " ".join(
        text for text in data["Cleaned_Tweets"][data["sentiment"] == "positive"]
    )
    Mask = np.array(
        Image.open(
            requests.get(
                "http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png",
                stream=True,
            ).raw
        )
    )
    image_color = ImageColorGenerator(Mask)
    wc = WordCloud(
        background_color="black", height=1500, width=4000, mask=Mask
    ).generate(positive_words)
    positive_wordcloud_file = "positive_wordcloud.png"
    wc.to_file(positive_wordcloud_file)
    negative_words = " ".join(
        text for text in data["Clean_Tweets"][data["sentiment"] == "negative"]
    )
    Mask = np.array(
        Image.open(
            requests.get(
                "http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png",
                stream=True,
            ).raw
        )
    )
    image_colors = ImageColorGenerator(Mask)
    wc = WordCloud(
        background_color="black", height=1500, width=4000, mask=Mask
    ).generate(negative_words)
    negative_wordcloud_file = "negative_wordcloud.png"
    wc.to_file(negative_wordcloud_file)

    # Analyze hashtags (same steps as before)
    def extractHashtags(x):
        hashtags = []
        for i in x:
            ht = re.findall(r"#(\w+)", i)
            hashtags.append(ht)
        return hashtags

    positive_hashTags = extractHashtags(
        data["Cleaned_Tweets"][data["sentiment"] == "positive"]
    )
    positive_hastags_unnested = sum(positive_hashTags, [])
    negative_hashtags = extractHashtags(
        data["Cleaned_Tweets"][data["sentiment"] == "negative"]
    )
    negative_hashtags_unnest = sum(negative_hashtags, [])
    positive_word_freq = nltk.FreqDist(positive_hastags_unnested)
    positive_df = pd.DataFrame(
        {
            "Hashtags": list(positive_word_freq.keys()),
            "Count": list(positive_word_freq.values()),
        }
    )
    negative_word_freq = nltk.FreqDist(negative_hashtags_unnest)
    negative_df = pd.DataFrame(
        {
            "Hashtags": list(negative_word_freq.keys()),
            "Count": list(negative_word_freq.values()),
        }
    )

    # Calculate sentiment distribution (same steps as before)
    total_count = len(data)
    positive_count = data[data["sentiment"] == "positive"]["Cleaned_Tweets"].count()
    negative_count = data[data["sentiment"] == "negative"]["Clean_Tweets"].count()
    neutral_count = total_count - (positive_count + negative_count)

    # Create visualizations (same steps as before)
    st.markdown(
        "<h3 style='text-align: center;'>Positive Word Cloud</h3>",
        unsafe_allow_html=True,
    )
    st.write(
        "A word cloud visualizing words from positive tweets. Each word's size represents its frequency in positive tweets."
    )
    st.image(positive_wordcloud_file, use_column_width=True)

    st.markdown(
        "<h3 style='text-align: center;'>Negative Word Cloud</h3>",
        unsafe_allow_html=True,
    )
    st.write(
        "A word cloud visualizing words from negative tweets. Each word's size represents its frequency in negative tweets."
    )
    st.image(negative_wordcloud_file, use_column_width=True)

    st.markdown(
        "<h3 style='text-align: center;'>Positive Hashtags Count</h3>",
        unsafe_allow_html=True,
    )
    st.write("Top 20 hashtags in positive tweets along with their respective counts.")
    positive_df_plot = positive_df.nlargest(20, columns="Count")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=positive_df_plot, y="Hashtags", x="Count", ax=ax)
    sns.despine()
    ax.set_title("Top 20 Positive Hashtags")
    st.pyplot(fig)

    st.markdown(
        "<h3 style='text-align: center;'>Negative Hashtags Count</h3>",
        unsafe_allow_html=True,
    )
    st.write("Top 20 hashtags in negative tweets along with their respective counts.")
    negative_df_plot = negative_df.nlargest(20, columns="Count")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=negative_df_plot, y="Hashtags", x="Count", ax=ax)
    sns.despine()
    ax.set_title("Top 20 Negative Hashtags")
    st.pyplot(fig)

    st.markdown(
        "<h3 style='text-align: center;'>Sentiment Distribution</h3>",
        unsafe_allow_html=True,
    )
    st.write(
        "A pie chart illustrating the distribution of sentiments (positive, negative, neutral) in the analyzed data."
    )
    st.write(
        "Each slice of the pie represents a sentiment category, and the size of each slice corresponds to the percentage of tweets belonging to that sentiment."
    )
    counts = [positive_count, negative_count, neutral_count]
    labels = ["Positive", "Negative", "Neutral"]
    colors = ["#66b3ff", "#ff9999", "#99ff99"]
    explode = (0.1, 0, 0)
    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        explode=explode,
    )
    st.pyplot(fig)


# Task 1: Ask the user to upload a CSV or Excel file containing tweets'
st.markdown(
    f"<h1 style='text-align: center;'>Twitter Sentimental Analysis</h1>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file containing tweets:", type=["csv", "xlsx"]
)
if uploaded_file:
    st.success("File uploaded successfully. Analyzing the data...")

    # Determine file type and read into a DataFrame
    if uploaded_file.name.endswith(".csv"):
        uploaded_data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        uploaded_data = pd.read_excel(uploaded_file)

    # Display sentiment analysis and visualizations for uploaded data
    perform_sentiment_analysis(uploaded_data, "Uploaded Data Sentiment Analysis")
