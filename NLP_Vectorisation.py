# useful packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class vectorisation(object):

    # import dataset
    data = pd.read_csv("Cleaned_Data.csv", sep=',')

    ## Vectorizing Data ##
    ## process of encoding text as integers ##

    ## standardise numerical data ##

    def additional_data(self) -> object:

        data = pd.read_csv("Cleaned_Data.csv", sep=',')

        # expanatory variable
        X_addition = data[["word_count", "char_count"]]
        # response variable
        Y = data["dinosaur_count"]
        data.head()

        x_train, x_test, y_train, y_test = train_test_split(X_addition, Y, test_size=0.2, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)

        list1 = X_addition.columns.tolist()
        ## standardize the data ##
        for i in list1:
            x_train[i] = (x_train[i] - min(x_train[i])) / (max(x_train[i]) - min(x_train[i]))
            x_val[i] = (x_val[i] - min(x_val[i])) / (max(x_val[i]) - min(x_val[i]))
            x_test[i] = (x_test[i] - min(x_test[i])) / (max(x_test[i]) - min(x_test[i]))
        x_train = x_train.reset_index(drop=True)

        x_val = x_val.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)

        return x_train, x_val, x_test

    for i in range(len(data["Documents"])):
        data["Documents"][i] = str(' '.join(list(data["Documents"][i])))

        ### TFIDF VECTORIZER ###

        def Tfidf_Vectorizer2(self, min_ngram: object, max_ngram: object, max_features: object) -> object:

            data = pd.read_csv("Cleaned_Data.csv", sep=',')
            # expanatory variable
            X = data["Documents"]
            # response variable
            Y = data["dinosaur_count"]

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
            vectorizer = TfidfVectorizer(min_df=0.02, max_features=max_features, max_df=0.95,
                                         ngram_range=(min_ngram, max_ngram))  # much less features
            X_train = vectorizer.fit_transform(x_train)
            X_validate = vectorizer.transform(x_val)
            X_test = vectorizer.transform(x_test)
            return X_train, X_validate, X_test, y_train, y_val, y_test
