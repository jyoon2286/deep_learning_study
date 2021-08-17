import csv
import json
from tqdm import tqdm
import re
import pandas as pd
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegression
from spacy.lang.en import English
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from empath import Empath
lexicon = Empath()


def read_csv_file(filename, essay = False):

    df = pd.read_csv (filename, sep=",", encoding='cp1252')
    if essay:
        df = df.groupby('#AUTHID').agg({'cNEU':'first', 
                                 'TEXT': ', '.join }).reset_index()
    else:
        df = df.groupby('#AUTHID').agg({'cNEU':'first', 
                                 'STATUS': ', '.join }).reset_index()

    data = df.values.tolist()
    return data

def read_and_clean_lines(infile):

    NeuScores = []
    Statuses = []
    for line in tqdm(data):
        NeuScore      = line[1]
        Status      = line[2]
        clean_text = re.sub(r"\s+"," ",Status)
        NeuScores.append(clean_text)
        Statuses.append(NeuScore)
    print("Read {} documents".format(len(NeuScores)))
    print("Read {} labels".format(len(Statuses)))
    return NeuScores, Statuses

def load_stopwords(filename):
    stopwords = []
    with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        stopwords = fp.read().split('\n')
    return set(stopwords)

def split_training_set(lines, labels, test_size=0.3, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, random_state=random_seed)
    # print("Training set label counts: {}".format(Counter(y_train)))
    # print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test



def convert_lines_to_feature_strings(lines, stopwords, remove_stopword_bigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print("Initializing")
    nlp          = English(parser=False)
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):

        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams          = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]

        # Collect string bigram tokens as features
        bigrams           = ngrams(normalized_tokens, 2) 
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'

        # TO DO: replace this line with your code
        feature_list   = unigrams + bigram_tokens
        feature_string = " ".join(feature_list)

        # Add this feature string to the output
        all_features.append(feature_string)


    # print(" Feature string for first document: '{}'".format(all_features[0]))
        
    return all_features

def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):

    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)

    # training_vectorizer = TfidfVectorizer(min_df=2, max_df=0.5,
    #                                       stop_words=stopwords_arg,
    #                                       analyzer=analyzefn,
    #                                       lowercase=True,
    #                                       ngram_range=range)

    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

def laxicon_extract(data):
    columns = []
    features = []
    for i in data:
        lex = lexicon.analyze(i, normalize=True)
        columns = list(lex.keys())
        features.append(lex)
    return pd.DataFrame(features)

def make_classifications(data,essay_data, stopwords, num_folds = 5, stratify = False,  classifier = "NB", features = "BG"):
    
    X, y                              = read_and_clean_lines(data)
    X_train, X_test, y_train, y_test  = split_training_set(X, y)
    
    essay_X, essay_y                  = read_and_clean_lines(essay_data)
    
    if features == "BG":
        X_features_train, training_vectorizer = convert_text_into_features(X_train, stop_words, "word", range=(1,2))
        X_test_features =  training_vectorizer.transform(X_test)
        essay_features = training_vectorizer.transform(essay_X)
        
    if features == "lexicon":
        X_features_train = laxicon_extract(X_train)
        X_test_features = laxicon_extract(X_test)
        essay_features = laxicon_extract(essay_X)
    
    
    if stratify == False:
        cv = KFold(n_splits=num_folds, random_state=42, shuffle=True)
    else:
        cv = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True)
        
        
    if classifier == "NB":
         clf = MultinomialNB()
            
    if classifier == "SVM":
        clf = SVC(kernel='rbf')
        
    if classifier == "RF":
        clf = RandomForestClassifier(max_depth=3, random_state=0)
        
    if classifier == "LR":
        clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')

  
    clf.fit(X_features_train, y_train)
    y_pred = clf.predict(X_test_features)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy For the prediction: {accuracy * 100}")
    
        
    y_essay_predict = clf.predict(essay_features)
    
    accuracy = accuracy_score(essay_y, y_essay_predict)
    print(f"Accuracy For the Essay prediction: {accuracy * 100}")
    
 

ip_file = "data/wcpr_mypersonality.csv"
essay_file = "data/wcpr_essays.csv"
stopword_file = "data/mallet_en_stoplist.txt"

data = read_csv_file(ip_file)
essay_data = read_csv_file(essay_file, essay = True)
stop_words = load_stopwords(stopword_file)

# make_classifications(data,essay_data, stop_words, classifier = "LR" )
# make_classifications(data,essay_data, stop_words, classifier = "RF" )
# make_classifications(data,essay_data, stop_words, classifier = "SVM" )
make_classifications(data,essay_data, stop_words, classifier = "NB" )