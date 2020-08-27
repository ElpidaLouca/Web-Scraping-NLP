# useful packages
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from tqdm import tqdm
import itertools

from NLP_Vectorisation import vectorisation


# Train model on training data

class model(vectorisation):

    def trainLasso(self, min_ngram, max_ngram, max_features,
                   alpha):  # Define a function to train lasso regression model on training data
        X_Train, X_validate, X_Test, y_train, y_val, y_test = vectorisation().Tfidf_Vectorizer2(min_ngram, max_ngram,
                                                                                                max_features)
        X_Train = pd.DataFrame(list(X_Train.toarray()))
        x_validate: object
        x_train, x_validate, x_test = vectorisation().additional_data()
        full_x_train = pd.concat([x_train, X_Train], axis=1)
        y_train = y_train.tolist()
        lassoreg = Lasso(alpha=alpha)  # Initialise lasso model
        fit = lassoreg.fit(X=full_x_train, y=y_train)
        return fit

    # Test model on Validation set (parameter optimisation) - RMSE

    def validateLassoMSE(self, min_ngram, max_ngram, max_features, alpha):
        X_Train, X_validate, X_Test, y_train, y_val, y_test = vectorisation().Tfidf_Vectorizer2(min_ngram, max_ngram,
                                                                                                max_features)
        X_validate = pd.DataFrame(list(X_validate.toarray()))
        x_train, x_validate, x_test = vectorisation().additional_data()
        full_x_validate = pd.concat([x_validate, X_validate], axis=1)
        lassoreg = self.trainLasso(min_ngram, max_ngram, max_features, alpha)  # Train model using specific alpha
        y_pred = lassoreg.predict(full_x_validate)  # Use model to predict on validation set
        y_pred[y_pred < 0] = 0
        rmse = np.sqrt(np.square(np.subtract(y_pred, y_val)).mean())
        return rmse  # Calculate and return mean-squared-error

    # Test model on Validation set (parameter optimisation) - R2

    def validateLassoR2(self, min_ngram, max_ngram, max_features, alpha):
        X_Train, X_validate, X_Test, y_train, y_val, y_test = vectorisation().Tfidf_Vectorizer2(min_ngram, max_ngram,
                                                                                                max_features)
        X_validate = pd.DataFrame(list(X_validate.toarray()))
        x_train, x_validate, x_test = vectorisation().additional_data()
        full_x_validate = pd.concat([x_validate, X_validate], axis=1)
        lassoreg = self.trainLasso(min_ngram, max_ngram, max_features, alpha)  # Train model using specific alpha
        y_pred = lassoreg.predict(full_x_validate)  # Use model to predict on validation set
        y_pred[y_pred < 0] = 0
        y_val = y_val.tolist()
        return r2_score(y_val, y_pred)  # Calculate and r-square

    def testmodel(self) -> object:
        min_ngram = [1]
        max_ngram = [2, 3, 4, 5]
        max_features = [100, 300, 500, 600, 700]
        alpha = [0.005, 0.008, 0.01]

        results_Lasso = pd.DataFrame()

        for i in tqdm(itertools.product(min_ngram, max_ngram, max_features, alpha)):
            r2 = self.validateLassoR2(i[0], i[1], i[2], i[3])
            mean_result = self.validateLassoMSE(i[0], i[1], i[2], i[3])
            results_Lasso = results_Lasso.append(pd.DataFrame({'max_ngram': [i[1]],
                                                               'max_features': i[2],
                                                               "alpha": i[3],
                                                               'r2': [r2],
                                                               'rmse': [mean_result]}), ignore_index=True)

        ## CREATE EXCEL FILE OF MAIN BODY ##
        results_Lasso_xls = pd.DataFrame(results_Lasso)
        results_Lasso_xls.to_csv("Performance.csv",index=False)

        return results_Lasso_xls

def main():
    x = model()
    x.testmodel()

