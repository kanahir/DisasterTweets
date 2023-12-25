import pandas as pd
from sklearn import model_selection
import Model
import Visualization
import PreProcessing


def predict_test(model, words_location, words_keyword):
    test_df = pd.read_csv("test.csv")
    # test_df = test_df.dropna()
    id = test_df['id']
    test_df = test_df.drop(['id'], axis=1)
    # extract features
    test_df,  _, _, _ = PreProcessing.pre_process(test_df, "Test", words_location, words_keyword)

    test_df.columns = test_df.columns.astype(str)
    y_pred = model.predict(test_df)
    # write to csv
    predictions_df = pd.DataFrame({'id': id, 'target': y_pred})
    predictions_df.to_csv('predictions.csv', index=False)
    return y_pred


def compare_models(X, y, file_name="Models Comparison"):
    df = pd.DataFrame(columns=["Model", "Score", "Value"])
    for model_name in Model.models_list:
        clf = Model.get_model(model_name)
        if model_name == "NN":
            scores = clf.cross_validation(X.values, y.values)
        else:
            scores = cross_validation(clf, X.values, y.values)
        print(model_name, scores)
        lst = []
        for score in scores:
            if score in ['fit_time', 'score_time']:
                continue
            for value in scores[score]:
                lst.append([model_name, score, value])
        df_ext = pd.DataFrame(lst, columns=["Model", "Score", "Value"])
        df = pd.concat([df, df_ext])
    df.to_csv(f"Results/{file_name}.csv")
    return df


def cross_validation(clf, X, y):
    scoring = ['f1', 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro', 'f1_weighted']
    scores = model_selection.cross_validate(clf, X, y, scoring=scoring, cv=5)
    return scores


def compare_features(X, y):
    features_options = ["Both", "Location", "Keyword", "Text"]
    df = pd.DataFrame(columns=["Score", "Value", "Features"])
    clf = Model.get_model("NN")
    for features_option in features_options:
        print(features_option)
        if features_option != "Both":
            columns = [col for col in X.columns if features_option.lower() in col]
            X_to_test = X[columns]
        else:
            X_to_test = X
        scores = clf.cross_validation(X_to_test.values.astype(float), y.values)
        lst = []
        for score in scores:
            if score in ['fit_time', 'score_time']:
                continue
            for value in scores[score]:
                lst.append([score, value, features_option])
        df_ext = pd.DataFrame(lst, columns=["Score", "Value", "Features"])
        df = pd.concat([df, df_ext])
        df.to_csv(f"Results/Features Comparison.csv")
    return df

if __name__ == '__main__':
    train_df = pd.read_csv("train.csv")
    # pre-process
    train_df, y, words_location, words_keyword = PreProcessing.pre_process(train_df)
    # Compare the models
    df = compare_models(train_df, y, file_name="Models Comparison Both")
    # Compare the features
    df = compare_features(train_df, y)
    # plot the results
    Visualization.plot_models_comparison(df, file_name="Both")
    Visualization.plot_models_comparison(df, compare_models_or_scores="scores", file_name="Both")
    # choose the best model
    clf = Model.get_model("SVM").fit(train_df.values, y.values)
    predict_test(clf, words_location, words_keyword)
