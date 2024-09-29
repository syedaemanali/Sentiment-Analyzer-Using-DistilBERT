import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import base64
from transformers import pipeline
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'sentiment-analyzer'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Find review column in the DataFrame
def find_review_column(df):
    for column in df.columns:
        if 'review' in column.lower() or 'text' in column.lower():
            return column
    return None

# Find rating column in the DataFrame
def find_rating_column(df):
    for column in df.columns:
        if 'rating' in column.lower() or 'overall' in column.lower():
            return column
    return None

# Classify sentiment using the pipeline
def classify_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):
        return 'unknown'
    result = sentiment_pipeline(text, truncation=True, padding=True, max_length=512)[0]
    return 'good' if result['label'].lower() == 'positive' else 'bad'

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for the sentiment analyzer page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Route to handle file upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.filename.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    df = pd.DataFrame(data) if isinstance(data, list) else None
            else:
                flash('File type not allowed. Please upload a CSV or JSON file.')
                return redirect(request.url)

            if df is None:
                flash('Invalid file format. Could not read the data.')
                return redirect(request.url)

            review_column = find_review_column(df)
            if review_column:
                df['sentiment'] = df[review_column].apply(classify_sentiment)
            else:
                flash('No review column found in the file.')
                return redirect(url_for('upload'))

            good_reviews = df[df['sentiment'] == 'good'][review_column]
            bad_reviews = df[df['sentiment'] == 'bad'][review_column]
            good_count = len(good_reviews)
            bad_count = len(bad_reviews)

            # Generate pie chart for sentiment distribution
            pie_chart = px.pie(values=[good_count, bad_count], names=['Positive', 'Negative'],
                               title='Sentiment Distribution', color_discrete_sequence=['green', 'red']).to_html(full_html=False)

            # Rating chart
            rating_column = find_rating_column(df)
            if rating_column:
                df['rounded_rating'] = df[rating_column].round()
                rating_counts = df['rounded_rating'].value_counts().sort_index()
                bar_chart = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values,
                                   labels={'x': 'Star Rating', 'y': 'Count of Reviews'},
                                   title='Count of Reviews by Star Rating').to_html(full_html=False)
            else:
                bar_chart = None

            # Sentiment by Review Length
            df['review_length'] = df[review_column].str.len()
            length_sentiment_chart = df.groupby('review_length')['sentiment'].value_counts().unstack().fillna(0)
            length_sentiment_chart = length_sentiment_chart.reset_index()
            length_sentiment_chart = length_sentiment_chart.melt(id_vars='review_length', value_vars=['good', 'bad'], var_name='Sentiment', value_name='Count')

            # Specify custom colors for the bar chart
            length_chart = px.bar(
                length_sentiment_chart,
                x='review_length',
                y='Count',
                color='Sentiment',
                color_discrete_sequence=['#003f5c', '#bc5090'],  # Dark blue for good, Dark pink for bad
                title='Sentiment by Review Length',
                labels={'review_length': 'Length of Review', 'Count': 'Number of Reviews'}
            )

            # Adjust layout for better readability
            length_chart.update_traces(marker=dict(opacity=0.85))  # Set bar opacity
            length_chart.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background for the plot area
                paper_bgcolor='white',  # White background for the entire chart
                title_font=dict(size=20),  # Increase title font size
                xaxis_title_font=dict(size=14),  # Increase x-axis title font size
                yaxis_title_font=dict(size=14),  # Increase y-axis title font size
                legend=dict(title_font=dict(size=12), font=dict(size=10))  # Adjust legend font size
            )

            # Convert to HTML for display
            length_chart_html = length_chart.to_html(full_html=False)

            # Word Frequency for Negative Reviews
            neg_words = ' '.join(bad_reviews)
            neg_wordcloud = WordCloud(width=400, height=200, background_color='white').generate(neg_words)
            img_pos = BytesIO()
            plt.figure(figsize=(5, 3))
            plt.imshow(neg_wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(img_pos, format='png')
            img_pos.seek(0)
            neg_wordcloud_base64 = base64.b64encode(img_pos.getvalue()).decode()

            return render_template('results.html',
                                   pie_chart=pie_chart,
                                   bar_chart=bar_chart,
                                   neg_wordcloud_image=neg_wordcloud_base64,
                                   length_chart=length_chart_html,
                                   good_count=good_count,
                                   bad_count=bad_count,
                                   total_reviews=good_count + bad_count)

        else:
            flash('File type not allowed. Please upload a CSV or JSON file.')
            return redirect(request.url)

    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(debug=True)
