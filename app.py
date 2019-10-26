# ''' Deploy user recomender model using flask framework'''

# Import Relevant Modules
try:
    import pickle
    import joblib
    from flask import Flask, request, render_template
    import pandas as pd
    from difflib import get_close_matches
    import requests
    from bs4 import BeautifulSoup
    from monkeylearn import MonkeyLearn
except ImportError as i_error:
    print(i_error)

# Load the datasets needed for deployment
cac_db = pd.read_csv('data/cac_db.csv')
# cac_db['name'] = cac_db.name.str.lower()

# load the model from disk
classifier = joblib.load('data/classifier.sav')


def search(word):
    word = word.upper()

    def start(x):
        if x.startswith(word):
            return True
        else:
            return False

    if len(cac_db[cac_db['COMPANY NAME'].apply(start)]) > 0:
        return cac_db[cac_db['COMPANY NAME'].apply(start)].head(5)
    elif len(get_close_matches(word, cac_db['COMPANY NAME'], 5, cutoff=0.7)) > 0:
        return get_close_matches(word, cac_db['COMPANY NAME'], 5, cutoff=0.7)
    else:
        return "Not in our CAC database"


def scrape(company_name):
    cn = company_name.split()
    cn = '+'.join(cn)
    url = f"https://www.nairaland.com/search?q={cn}&board=0"
    links = [url]
    posts = []
    header = {'User-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36"}
    scraped = requests.get(url, headers=header)
    page = scraped.content
    soup = BeautifulSoup(page, 'html.parser')
    p = soup.find('p')
    a = p.find_all('a')
    for link in a:
        a_tag = link.get('href')
        if str(a_tag).startswith("https://www.nairaland.com/search"):
            links.append(a_tag)
    for link in links:
        scraped = requests.get(link, headers=header)
        page = scraped.content
        soup = BeautifulSoup(page, 'html.parser')
        nar = soup.find_all('div', {'class': "narrow"})
        for i in range(len(nar)):
            post = nar[i].text
            posts.append(post)
    return posts


def get_prediction(post, loaded_class):
    pred = []
    for text in post:
        features = dict([(word, True) for word in text.split()])
        probdist = loaded_class.prob_classify(features)
        pred_sentiment = probdist.max()
        if pred_sentiment == 'Positive':
            pred.append(1)
        else:
            pred.append(0)
    avg_pred = round(sum(pred)/len(pred))
    if avg_pred == 1:
        return "Positive"
    else:
        return 'Negative'


def parse(data):
    company_name = None
    address = None
    contact = None
    ml = MonkeyLearn('7a46e749c957dd41ff0f5fe1365598120f52b76a')
    data = [data]
    model_id = 'ex_P7wZeSxj'
    result = ml.extractors.extract(model_id, data)
    text = result.body[0]['text']
    extractions = result.body[0]['extractions']
    for i in range(len(extractions)):
        if extractions[i]['tag_name'] == 'Company Name':
            company_name = extractions[i]['extracted_text']
        elif extractions[i]['tag_name'] == 'Address':
            address = extractions[i]['extracted_text']
        elif extractions[i]['tag_name'] == 'Contact':
            contact = extractions[i]['extracted_text']
    return company_name, address, contact, text


# Initialize the app
APP = Flask(__name__)

'''HTML GUI '''
# Render the home page
@APP.route('/')
def home():
    """Display of web app homepage"""
    return render_template('index.html')

# Render the parser page
@APP.route('/parser')
def parser():
    """Display of web app parser"""
    return render_template('parser.html')

# render the company search page
@APP.route('/company_search')
def company_find():
    """Display of result of search"""
    return render_template('company_find.html')

# render the company select page
@APP.route('/company_select')
def company_select():
    """Display of result of search"""
    return render_template('company_find.html')

# render the search page
@APP.route('/search', methods=['POST'])
def search_company():
    """Function that accepts the company name displays search result
    for Web App Testing"""
    # get the values from the form
    try:
        search_query = [x for x in request.form.values()]
        result = search(search_query[0])
        if isinstance(result, pd.DataFrame):
            result = result.values.tolist()
            result_list = []
            if len(result) > 1:
                for row in result:
                    company_name = row[1]
                    result_list.append(company_name)
                return render_template('company_select.html', prediction_text=result_list)
            result_list.append(f'Company name: {result[0][1]}')
            result_list.append(f'Address: {result[0][2]}')
            result_list.append(f'RC Number: {result[0][0]}')

        else:
            result_list = [result]
        post = scrape(search_query[0])
        if len(post) > 0:
            prediction = get_prediction(post, classifier)
            prediction_text = f'Nairaland search for {search_query[0]} returned {str(len(post))} posts that are mostly {prediction}'
            result_list.append(prediction_text)
        else:
            prediction_text = "A search for this company on Nairaland didn't return any results"
            result_list.append(prediction_text)
        return render_template('search.html', prediction_text=result_list)
    except KeyError:
        return render_template('search.html', prediction_text=[["company does not exist"]])


# render the search page
@APP.route('/search2', methods=['POST'])
def parse_iv():
    # get the values from the form
    try:
        result_list = []
        search_query = [x for x in request.form.values()]
        company_name, address, contact, text = parse(search_query[0])
        message1 = f"Your input: {text}"
        if company_name is not None:
            message2 = f"Company Name: {company_name}"
            result = search(company_name)
            if isinstance(result, pd.DataFrame):
                result = result.values.tolist()
                result_list = []
                if len(result) > 1:
                    for row in result:
                        company_name = row[1]
                        result_list.append(company_name)
                    # return render_template('company_select.html', prediction_text=result_list)
                m1 = 'comparing with CAC data'
                m11 = f'Company name in CAC Database: {result[1][1]}'
                m12 = f'Address in CAC database: {result[1][2]}'
                m13 = f'RC Number: {result[1][0]}'
                m5 = None

            else:
                m1, m11, m12, m13 = None, None, None, None
                m5 = result
        else:
            message2, m1, m11, m12, m13, m5 = None, None, None, None, None, None
        if address is not None:
            message3 = f'Address : {address}'
        else:
            message3 = None
        if contact is not None:
            message4 = f'Contact: {contact}'
        else:
            message4 = None

        m_list = [message1, message2, message3, message4, m1, m11, m12, m13, m5]

        for message in m_list:
            if message is not None:
                result_list.append(message)
        return render_template('search.html', prediction_text=result_list)
    except:
        return render_template('search.html', prediction_text=[["Wahala dey ohhh"]])


# run the app
if __name__ == "__main__":
    APP.run(debug=True)
