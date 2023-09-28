import re
import pandas as pd
import numpy as np
import streamlit as st
import string
import nltk
import warnings
from PIL import Image
import requests
from wordcloud import ImageColorGenerator, WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# Set the page title as an H1 heading
st.markdown(
    "<h1 style='text-align: center;'>Twitter Sentiment Analysis</h1>",
    unsafe_allow_html=True,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv(
    "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
)
train_orignal = train.copy()

test = pd.read_csv(
    "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv"
)
test_original = test.copy()

combined_data = pd.concat([train, test], ignore_index=True, sort=True)


def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, "", text)
    return text


combined_data["Cleaned_Tweets"] = np.vectorize(remove_pattern)(
    combined_data["tweet"], "@[\w]*"
)

combined_data["Cleaned_Tweets"] = combined_data["Cleaned_Tweets"].str.replace(
    "[^a-zA-Z#]", " ", regex=True
)

combined_data["Cleaned_Tweets"] = combined_data["Cleaned_Tweets"].apply(
    lambda x: " ".join([w for w in x.split() if len(w) > 3])
)

tokenized_tweets = combined_data["Cleaned_Tweets"].apply(lambda x: x.split())

from nltk import PorterStemmer

ps = PorterStemmer()
tokenized_tweets = tokenized_tweets.apply(lambda x: [ps.stem(i) for i in x])

for i in range(len(tokenized_tweets)):
    tokenized_tweets[i] = " ".join(tokenized_tweets[i])

combined_data["Clean_Tweets"] = tokenized_tweets

positive_words = " ".join(
    text for text in combined_data["Cleaned_Tweets"][combined_data["label"] == 0]
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
wc = WordCloud(background_color="black", height=1500, width=4000, mask=Mask).generate(
    positive_words
)

# Save the positive WordCloud as an image file
positive_wordcloud_file = "positive_wordcloud.png"
wc.to_file(positive_wordcloud_file)

# Display the positive WordCloud using st.image
st.markdown(
    "<h3 style='text-align: center;'>Positive Word Cloud</h1>",
    unsafe_allow_html=True,
)
st.write(
    "Explore the Positive Word Cloud: Delve into the most frequent and uplifting words used in the tweets with positive sentiment. Discover the words that bring joy and positivity to the Twitterverse"
)
st.image(positive_wordcloud_file, use_column_width=True)

negative_words = " ".join(
    text for text in combined_data["Clean_Tweets"][combined_data["label"] == 1]
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
wc = WordCloud(background_color="black", height=1500, width=4000, mask=Mask).generate(
    negative_words
)

# Save the negative WordCloud as an image file
negative_wordcloud_file = "negative_wordcloud.png"
wc.to_file(negative_wordcloud_file)

# Display the negative WordCloud using st.image
st.markdown(
    "<h3 style='text-align: center;'>Negative Word Cloud</h1>",
    unsafe_allow_html=True,
)
st.write(
    "Unveil the Negative Word Cloud: Dive into the most common words found in tweets with negative sentiment. Gain insights into the words that express dissatisfaction and disappointment in the Twitter world."
)
st.image(negative_wordcloud_file, use_column_width=True)


def extractHashtags(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


positive_hashTags = extractHashtags(
    combined_data["Cleaned_Tweets"][combined_data["label"] == 0]
)
positive_hastags_unnested = sum(positive_hashTags, [])

negative_hashtags = extractHashtags(
    combined_data["Cleaned_Tweets"][combined_data["label"] == 1]
)
negative_hashtags_unnest = sum(negative_hashtags, [])

positive_word_freq = nltk.FreqDist(positive_hastags_unnested)
positive_df = pd.DataFrame(
    {
        "Hashtags": list(positive_word_freq.keys()),
        "Count": list(positive_word_freq.values()),
    }
)

st.markdown(
    "<h3 style='text-align: center;'>Positive Hashtags Count</h1>",
    unsafe_allow_html=True,
)
positive_df_plot = positive_df.nlargest(20, columns="Count")
# Create a bar plot of positive hashtags
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=positive_df_plot, y="Hashtags", x="Count", ax=ax)
sns.despine()
ax.set_title("Top 20 Positive Hashtags")
st.pyplot(fig)

negative_word_freq = nltk.FreqDist(negative_hashtags_unnest)
negative_df = pd.DataFrame(
    {
        "Hashtags": list(negative_word_freq.keys()),
        "Count": list(negative_word_freq.values()),
    }
)

st.markdown(
    "<h3 style='text-align: center;'>Negative Hashtags Count</h1>",
    unsafe_allow_html=True,
)
negative_df_plot = negative_df.nlargest(20, columns="Count")
# Create a bar plot of negative hashtags
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=negative_df_plot, y="Hashtags", x="Count", ax=ax)
sns.despine()
ax.set_title("Top 20 Negative Hashtags")
st.pyplot(fig)

# Calculate the total number of comments
total_count = len(combined_data)

# Calculate the number of positive, negative, and neutral comments
positive_count = combined_data[combined_data["label"] == 0]["Cleaned_Tweets"].count()
negative_count = combined_data[combined_data["label"] == 1]["Clean_Tweets"].count()
neutral_count = total_count - (positive_count + negative_count)

# Calculate the percentages
positive_percentage = (positive_count / total_count) * 100
negative_percentage = (negative_count / total_count) * 100
neutral_percentage = (neutral_count / total_count) * 100

# Calculate the number of positive and negative comments
positive_count = combined_data[combined_data["label"] == 0]["Cleaned_Tweets"].count()
negative_count = combined_data[combined_data["label"] == 1]["Clean_Tweets"].count()

# Calculate the total number of comments
total_count = len(combined_data)

# Create a list of counts
counts = [positive_count, negative_count, neutral_count]

# Labels for the sections
labels = ["Positive", "Negative", "Neutral"]

# Colors for each section
colors = ["#66b3ff", "#ff9999", "#99ff99"]

# Explode the 2nd slice (i.e., 'Negative')
explode = (0.1, 0, 0)

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(
    counts,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    explode=explode,
)

# Add a title
st.markdown(
    "<h3 style='text-align: center;'>Twitter Sentiment Analysis</h1>",
    unsafe_allow_html=True,
)

# Display the pie chart
st.pyplot(fig)
