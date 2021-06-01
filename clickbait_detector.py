"""Determining if an article is clickbait based on its title"""
import ast
import nltk
from nltk import pos_tag, word_tokenize
import statistics
import sklearn.feature_extraction  # type: ignore
import sklearn.linear_model  # type: ignore



def find_token(token: str,lis: list) -> int: #casefolds words/tokens and finds specified word in list of tokens 
    token = token.casefold()
    for word in lis:
        word = word.casefold()
        if word == token: return "True"
        else: return "False"

def find_POS(POS: str,lis: list) -> int: 
    if POS in lis: return "True"
    else: return "False"

def list_POS(lis: list) -> list:    #creates list of POS tags given list of tokens
    POS_list = []
    for word in lis:
        POS_list.append(pos_tag(word.casefold())[0][1])
    return POS_list

def count_POS(POS: str,lis: list) -> int:
    count = 0
    for pos in lis:
        if pos == POS: count +=1
    return count


def extract_article_features(title: str):
    features = {}
    tokens = nltk.word_tokenize(title)
    
    # an attempt to only count words and not punctuation; 
    # though this would count letters after an apostrophe as separate words..
    words = []
    for token in tokens:
        if token.isalnum():
            words.append(token)     
    features["title token count"] = str(len(words))
    
    POS_list = list_POS(tokens)

    features["no. of adj"] = str(count_POS("JJ",POS_list))
    
    for pos in ["RB","RBR","RBS"]:
        if find_POS(pos,POS_list) == "True":
            if POS_list.index(pos) == len(POS_list) - 1:
              features["contains adv-adj phrase"] = "False"  
            elif POS_list[POS_list.index(pos)+1] == "JJ":
                features["contains adv-adj phrase"] = "True"
            else: features["contains adv-adj phrase"] = "False"
            break
        else: features["contains adv-adj phrase"] = "False"

    
    if find_POS("WDT",POS_list) == "True" or find_POS("WP",POS_list) == "True" or find_POS("WPS",POS_list) == "True":
        features["contains wh-words"] = "True"
    else: features["contains wh-words"] = "False"

    for pos in ["JJS","RBS"]:
        if find_POS(pos,tokens) == "True":
            features["superlative adj./adv."] = "True"
            break
        else: features["superlative adj./adv."] = "False"
    
    for word in ["I","My","Me","We","Our"]:
        if find_token(word,tokens) == "True":
            features["first person"] = "True"
            break
        else: features["first person"] = "False"
    
    for word in ["You","Your","Yours","Yourself"]:
        if find_token(word,tokens) == "True":
            features["second person"] = "True"
            break
        else: features["second person"] = "False"
    
    if find_POS("Should",tokens) == "True":
        features["contains 'should'"] = "True"
    else: features["contains 'should'"] = "False"

    
    if find_token("?",tokens) == "True":
        features["has question mark"] = "True"
    else: features["has question mark"] = "False"
    
    if find_token("!",tokens) == "True":
        features["has exclamation mark"] = "True"
    else: features["has exclamation mark"] = "False"
    
    if find_token("Tweet",tokens) == "True" or find_token("Tweets",tokens) == "True":
        features["about tweets"] = "True"
    else: features["about tweets"] = "False"

    for word in ["President","Government","Governor","Congress", "Congressional","Pope","Minister","Politics","Political","Politician"]:
        if find_token(word,tokens) == "True":
            features["about politics"] = "True"
            break
        else: features["about politics"] = "False"


    return features

def extract_features_labels(path:str):
    features = []
    labels = []
    with open(path,"r") as source:
        for line in source:
            x = ast.literal_eval(line)
            article = list(x)[0]
            labels.append(x[article])
            features.append(extract_article_features(article))
    return features, labels

def main():
    train_path = "data/train"
    train_features, train_labels = extract_features_labels(train_path)

    vectorizer = sklearn.feature_extraction.DictVectorizer()

    train_feature_vect = vectorizer.fit_transform(train_features)

    model = sklearn.linear_model.LogisticRegression(penalty="l1",C=10,solver="liblinear")

    model.fit(train_feature_vect,train_labels)

    test_path = "data/test"

    test_features, test_labels = extract_features_labels(test_path)

    test_feature_vect = vectorizer.transform(test_features)
            
    prediction = model.predict(test_feature_vect)

    correct = []
    size = 0
    correct_count = 0
    
    for pred, label in zip(prediction,test_labels):
        if pred == label:
            correct_count += 1
        size += 1
        
    correct_sum = 0

    acc = (correct_count/size) * 100
    
    print(f'accuracy: {acc}')

main()