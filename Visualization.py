import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import os
import pandas as pd
plot(1)

def plot_models_comparison(df, compare_models_or_scores="models", file_name=""):
    # plt.figure(figsize=(10, 5))
    # The x-axis is the score, the bars are the models and the y-axis is the value
    df_stats = df.groupby(["Model", "Score"]).agg(["mean","std"])
    # extract the scores to a column
    df_stats.reset_index(inplace=True)
    if compare_models_or_scores == "models":
        df_stats = df_stats.pivot(index="Model", columns="Score")
    else:
        df_stats = df_stats.pivot(index="Score", columns="Model")
    yerr = df_stats[('Value', 'std')]
    y= df_stats[('Value', 'mean')]
    ax = y.plot(kind="barh")
    flatten_errors = yerr.values.flatten()

    # add the labels
    labels = list(y.columns)
    labels = [" ".join(label.replace("test_", "").split('_')).capitalize() for label in labels]
    plt.legend(labels)

    # Change y labels
    y_labels = ax.get_yticklabels()
    y_labels = [label.get_text() for label in y_labels]
    y_labels = [" ".join(label.replace("test_", "").split('_')).capitalize() for label in y_labels]
    ax.set_yticklabels(y_labels)

    # add the error bars
    for i, p in enumerate(ax.patches):
        value = p.get_width()  # get width of bar
        h = p.get_y()
        l = p.get_height()
        error = flatten_errors[i]  # use h to get min from dict z
        plt.hlines(h+l/2, value-error/2, value+error/2, color='k')  # draw a vertical line

    plt.title("Models Comparison")
    if not os.path.exists("Results"):
        os.mkdir("Results")
    plt.savefig(f"Results/Models Comparison {compare_models_or_scores} {file_name}.png")
    plt.show(block=False)


def compare_features():
    features_types = ["Basic", "just Text", "Both"]
    df = pd.DataFrame(columns=["Score", "Value", "Features"])
    for features_type in features_types:
        df_ext = pd.read_csv(f"Results/Models Comparison {features_type}.csv", index_col=0)
        df_ext = df_ext[df_ext["Model"] == "NN"]
        df_ext.drop("Model", axis=1, inplace=True)
        df_ext["Features"] = features_type
        df = pd.concat([df, df_ext])
    average_df = df.groupby(["Features", "Score"]).agg(["mean", "std"])
    average_df.reset_index(inplace=True)
    average_df = average_df.pivot(index="Score", columns="Features")
    y = average_df[('Value', 'mean')]
    ax = y.plot(kind="barh")
    # add the labels
    labels = list(y.columns)
    labels = [" ".join(label.replace("test_", "").split('_')).capitalize() for label in labels]
    plt.legend(labels)
    # Change y labels
    y_labels = ax.get_yticklabels()
    y_labels = [label.get_text() for label in y_labels]
    y_labels = [" ".join(label.replace("test_", "").split('_')).capitalize() for label in y_labels]
    ax.set_yticklabels(y_labels)
    plt.title("Models Comparison")
    plt.savefig(f"Results/Models Comparison Features.png")
    plt.show(block=False)



def compare_features2():
    df = pd.read_csv("Results/Features Comparison.csv", index_col=0)
    average_df = df.groupby(["Features", "Score"]).agg(["mean", "std"])
    average_df.reset_index(inplace=True)
    average_df = average_df.pivot(index="Score", columns="Features")
    y = average_df[('Value', 'mean')]
    ax = y.plot(kind="barh")
    # add the labels
    labels = list(y.columns)
    labels = [" ".join(label.replace("test_", "").split('_')).capitalize() for label in labels]
    plt.legend(labels)
    # Change y labels
    y_labels = ax.get_yticklabels()
    y_labels = [label.get_text() for label in y_labels]
    y_labels = [" ".join(label.replace("test_", "").split('_')).capitalize() for label in y_labels]
    ax.set_yticklabels(y_labels)
    plt.title("Models Comparison")
    plt.savefig(f"Results/Features Comparison MLP.png")
    plt.show(block=False)


if __name__ == '__main__':
    compare_features2()

