'''
1. enter location & restaurant 
2. with cache

'''
from flask import Flask, render_template, request
from flask_caching import Cache
import googlemaps
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import os
import nltk
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import openai
import csv

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': 'simple'})

gmaps = googlemaps.Client(key='AIzaSyCS38TI7v43Q9bLhMbEMRlfccMfX-Vg3gg')
openai.api_key = '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'

@app.route('/')
def home():
    return render_template('home03.html')

@cache.memoize(50)
def fetch_reviews(place_id):
    place_details = gmaps.place(place_id=place_id)
    return place_details['result'].get('reviews', [])

@app.route('/result', methods=['POST'])
def result():
    restaurant_name = request.form['restaurant_name']
    location = request.form['location']
    shape = request.form['shape']
    
    places_result = gmaps.places(query=restaurant_name, location=location, language='zh-TW')
    place_id = places_result['results'][0]['place_id']

    reviews = fetch_reviews(place_id)

    with open('static/reviews.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Review'])
        for review in reviews:
            writer.writerow([review['text']])

    review_texts = [review['text'] for review in reviews]
    text = ' '.join(review_texts)

    ## show the translated score only
    '''
    sentiment_scores = [TextBlob(review).sentiment.polarity for review in review_texts]
    sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Convert sentiment from [-1, 1] to [0, 100]
    sentiment = 50 * (sentiment + 1)
    '''
    # show both the sentiment score and translated score
    sentiment_scores = [TextBlob(review).sentiment.polarity for review in review_texts]
    sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Calculate translated sentiment score
    #sentiment_score = 50 * (sentiment + 1)
    sentiment_score = 50 * (sentiment + 1.2)
    
    # Calculate star rating (0-5 scale)
    star_rating = round(sentiment_score / 20 * 2) / 2  # Rounds to nearest half
    
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text,
        max_tokens=100,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        n=1
    )

    summary = response.choices[0].text.strip()

    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(lemmatized_tokens)

    stopwords = set(STOPWORDS)

    mask_image_path = None
    if shape in ['news', 'twitter', 'github', 'message']:
        mask_image_path = f'static/{shape}.png'

    if mask_image_path:
        mask_image = Image.open(mask_image_path)
        resized_mask_image = mask_image.resize((1200, 800), Image.ANTIALIAS)
        mask = np.array(resized_mask_image)
    else:
        mask = None

    wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=1200, height=800, mask=mask).generate(text)

    if shape in ['news', 'twitter', 'github', 'message']:
        output_filename = f'static/{shape.capitalize()}Cloud.png'
        wordcloud.to_file(output_filename)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        timestamp = None

    #return render_template('result03.html', shape=shape, os=os, timestamp=timestamp, summary=summary, sentiment=sentiment, sentiment_score=sentiment_score, star_rating=star_rating)
    sentiment_integer = int(star_rating)
    sentiment_decimal = star_rating - sentiment_integer

    return render_template(
        'result03.html', 
        shape=shape, 
        os=os, 
        timestamp=timestamp, 
        summary=summary, 
        sentiment=sentiment, 
        sentiment_score=sentiment_score, 
        star_rating=star_rating, 
        sentiment_integer=sentiment_integer, 
        sentiment_decimal=sentiment_decimal
    )


if __name__ == '__main__':
    app.run(debug=True)
