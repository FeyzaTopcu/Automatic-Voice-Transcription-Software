from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np





app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])


def predict(features):
    path_df = r"C:\Users\Feyza\Desktop\denemetez\latest-dneme\Dataset Creation\News_dataset.pickle"

    with open(path_df, 'rb') as data:
    df = pickle.load(data)
    

    df.head()

    df.loc[1]['Content']

  
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")

    text = "Mr Greenspan\'s"
    text

# " when quoting text
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')

# Lowercasing the text
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()

    punctuation_signs = list("...?:!.,;")
    df['Content_Parsed_3'] = df['Content_Parsed_2']

    for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    

    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    

    # Downloading punkt and wordnet from NLTK
    nltk.download('punkt')
    print("------------------------------------------------------------")
    nltk.download('wordnet')

    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()

    nrows = len(df)
    lemmatized_text_list = []

    for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = df.loc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
        lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    
    
    df['Content_Parsed_5'] = lemmatized_text_list

# Downloading the stop words list
    nltk.download('stopwords')


# Loading the stop words in english
    stop_words = list(stopwords.words('english'))

    stop_words[0:10]

    example = "me eating a meal"
    word = "me"

# The regular expression is:
    regex = r"\b" + word + r"\b"  # we need to build it like that to work properly

    re.sub(regex, "StopWord", example)


    df['Content_Parsed_6'] = df['Content_Parsed_5']

    for stop_word in stop_words:

        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
    
    df.loc[5]['Content']

    df.loc[5]['Content_Parsed_1']

    df.loc[5]['Content_Parsed_2']


    df.loc[5]['Content_Parsed_3']

    df.loc[5]['Content_Parsed_4']

    df.loc[5]['Content_Parsed_5']

    df.loc[5]['Content_Parsed_6']

    df.head(1)  

    list_columns = ["File_Name", "Category", "Complete_Filename", "Content", "Content_Parsed_6"]
    df = df[list_columns]

    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})

    df.head()

    category_codes = {
            'business': 0,
            'entertainment': 1,
            'politics': 2,
            'sport': 3,
            'tech': 4
            }

# Category mapping
    df['Category_Code'] = df['Category']
    df = df.replace({'Category_Code':category_codes})

    df.head()

    X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'], 
                                                    df['Category_Code'], 
                                                    test_size=0.15, 
                                                    random_state=8)

    # Parameter election
    ngram_range = (1,2)
    min_df = 10
    max_df = 1.
    max_features = 300

    tfidf = TfidfVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words=None,
                            lowercase=False,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features,
                            norm='l2',
                            sublinear_tf=True)
                        
    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    print(features_train.shape)

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test
    print(features_test.shape)

    from sklearn.feature_selection import chi2
    import numpy as np

    for Product, category_id in sorted(category_codes.items()):
        features_chi2 = chi2(features_train, labels_train == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}' category:".format(Product))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
        print("")


    bigrams

# X_train
    with open('X_train.pickle', 'wb') as output:
        pickle.dump(X_train, output)
    
# X_test    
    with open('X_test.pickle', 'wb') as output:
        pickle.dump(X_test, output)
    
# y_train
    with open('y_train.pickle', 'wb') as output:
        pickle.dump(y_train, output)
    
# y_test
    with open('y_test.pickle', 'wb') as output:
        pickle.dump(y_test, output)
    
# df
    with open('df.pickle', 'wb') as output:
        pickle.dump(df, output)
    
# features_train
    with open('features_train.pickle', 'wb') as output:
        pickle.dump(features_train, output)

# labels_train
    with open('labels_train.pickle', 'wb') as output:
        pickle.dump(labels_train, output)

# features_test
    with open('features_test.pickle', 'wb') as output:
        pickle.dump(features_test, output)

# labels_test
with open('labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
    with open('tfidf.pickle', 'wb') as output:
        pickle.dump(tfidf, output)
    # Obtain the highest probability of the predictions for each article
    predictions_proba = svc_model.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = svc_model.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .65:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = get_category_name(prediction_svc)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)