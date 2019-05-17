from flask import Flask, render_template
app = Flask(__name__)

posts = [
    {
        'author': 'Corey Schafer',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'Jane Doe',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 19, 2018'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home_landing.html', posts=posts)


@app.route("/review")
def about():
    return render_template('review.html')


@app.route("/classify")
def about():
    return render_template('full_detection.html')


@app.route("/classify-tweet")
def about():
    return render_template('tweet_detection.html')


if __name__ == '__main__':
    app.run(debug=True)