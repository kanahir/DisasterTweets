from sklearn import linear_model
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = ["NN", "Linear_Regression", "SVM", "NuSVM", "Random_Forest", "XGBoost"]

LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
def get_model(model_name):
    if model_name == "Linear_Regression":
        return linear_model.LogisticRegression(C=1e5)
    elif model_name == "SVM":
        return SVC()
    elif model_name == "NuSVM":
        return NuSVC()
    elif model_name == "Random_Forest":
        return RandomForestClassifier()
    elif model_name == "XGBoost":
        return XGBClassifier()
    elif model_name == "NN":
        return NN_Model()
    else:
        print("Model name is not valid")
        exit()

# Define MLP model
class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()

    def initialize(self, input_size, output_size, class_weights=None):
        self.layer_1 = nn.Linear(input_size, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.dropout = nn.Dropout(p=0.25)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)


    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

    def predict(self, x):
        x = torch.Tensor(x)
        self.eval()
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

    def train_model(self, X_train, y_train, X_val, y_val):
        X_train, y_train = torch.Tensor(X_train), torch.LongTensor(y_train)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        X_val, y_val = torch.Tensor(X_val), torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        train_loss_vector = []
        val_loss_vector = []
        train_accuracy_vector = []
        val_accuracy_vector = []
        for iteration in range(self.epochs):
            if self.is_converged(train_loss_vector):
                break
            self.epoch(train_loader, train_accuracy_vector, train_loss_vector, train=True)
            self.epoch(val_loader, val_accuracy_vector, val_loss_vector, train=False)
            # print(f"Epoch: {iteration+1}/{self.epochs}, Train Loss: {train_loss_vector[-1]:.4f}, Train Accuracy: {train_accuracy_vector[-1]:.4f}, Val Loss: {val_loss_vector[-1]:.4f}, Val Accuracy: {val_accuracy_vector[-1]:.4f}")
        # self.plot(train_loss_vector, val_loss_vector, train_accuracy_vector, val_accuracy_vector)
        return


    def epoch(self, loader, accuracy_vector, loss_vector, train=True):
        running_loss = 0.0
        running_accuracy = 0.0
        if train:
            self.train()
        else:
            self.eval()

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = self.forward(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            running_accuracy += accuracy_score(y_batch, torch.argmax(y_pred, dim=1))
        running_loss /= len(loader)
        running_accuracy /= len(loader)
        loss_vector.append(running_loss)
        accuracy_vector.append(running_accuracy)


    def plot(self, train_loss_vector, val_loss_vector, train_accuracy_vector, val_accuracy_vector):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_vector, label="Train Loss")
        plt.plot(val_loss_vector, label="Val Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy_vector, label="Train Accuracy")
        plt.plot(val_accuracy_vector, label="Val Accuracy")
        plt.legend()
        plt.show(block=False)
        plt.savefig("Results/NN_training.png")

    def cross_validation(self, X, y):
        self.initialize(X.shape[1], len(np.unique(y)))
        scoring = ['f1', 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro', 'f1_weighted']
        k_cross = 5
        indexes = np.arange(len(X))
        np.random.shuffle(indexes)
        indexes = np.array_split(indexes, k_cross)
        scores = {"test_" + score: [] for score in scoring}
        for i in range(k_cross):
            print(f"Cross Validation: {i+1}/{k_cross}")
            train_indexes = np.concatenate(indexes[:i] + indexes[i+1:])
            test_indexes = indexes[i]
            X_train, y_train = X[train_indexes], y[train_indexes]
            X_test, y_test = X[test_indexes], y[test_indexes]
            self.train_model(X_train, y_train, X_test, y_test)
            y_pred = self.predict(X_test)
            scores["test_f1"].append(f1_score(y_test, y_pred))
            scores["test_accuracy"].append(accuracy_score(y_test, y_pred))
            scores["test_precision"].append(precision_score(y_test, y_pred, average="macro"))
            scores["test_recall"].append(recall_score(y_test, y_pred, average="macro"))
            scores["test_f1_macro"].append(f1_score(y_test, y_pred, average="macro"))
            scores["test_f1_micro"].append(f1_score(y_test, y_pred, average="micro"))
            scores["test_f1_weighted"].append(f1_score(y_test, y_pred, average="weighted"))
            print(scores)
        return scores

    def is_converged(self, loss_vector):
        if len(loss_vector) < 11:
            return False
        return np.min(loss_vector[-10:]) > np.min(loss_vector[:-10])


def set_fit(text_train_df, text_val_df):
    # rename target column to label
    text_train_df.rename(columns={"target": "label"}, inplace=True)
    text_val_df.rename(columns={"target": "label"}, inplace=True)
    train_ds = Dataset.from_pandas(text_train_df)
    test_ds = Dataset.from_pandas(text_val_df)
    # Load SetFit model from Hub
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        batch_size=16,
        num_iterations=20,  # Number of text pairs to generate for contrastive learning
        num_epochs=1  # Number of epochs to use for contrastive learning
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)



if __name__ == '__main__':
    pass

