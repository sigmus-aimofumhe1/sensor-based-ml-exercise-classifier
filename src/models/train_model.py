import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/03_data_features.pkl")

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# CLEAN DATA (CRITICAL FIX - prevents ALL sklearn errors)
# --------------------------------------------------------------
df_model = df.copy()

# Remove metadata columns
df_model = df_model.drop(
    columns=["participant", "category", "set"],
    errors="ignore"
)

# Remove any epoch / datetime columns
df_model = df_model.loc[:, ~df_model.columns.str.contains("epoch")]

# Separate label
y = df_model["label"]
X = df_model.drop("label", axis=1)

# Keep only numeric features (VERY IMPORTANT)
X = X.select_dtypes(include=[np.number])

print("Final feature shape:", X.shape)

# --------------------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------------------
# CLASS DISTRIBUTION CHECK
# --------------------------------------------------------------
fig, ax = plt.subplots()
y.value_counts().plot(kind="bar", color="lightgray", label="Total", ax=ax)
y_train.value_counts().plot(kind="bar", color="dodgerblue", label="Train", ax=ax)
y_test.value_counts().plot(kind="bar", color="royalblue", label="Test", ax=ax)
plt.legend()
plt.title("Label Distribution")
plt.show()

# --------------------------------------------------------------
# FEATURE GROUPS (SAFE - based on cleaned X)
# --------------------------------------------------------------
basic_features = [c for c in X.columns if c in [
    "acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"
]]

squared_features = [c for c in X.columns if "acc_r" in c or "gyr_r" in c]
pca_features = [c for c in X.columns if "pca_" in c]
time_features = [c for c in X.columns if "_temp_" in c]
freq_features = [c for c in X.columns if "_freq_" in c or "_pse" in c]
cluster_features = [c for c in X.columns if "cluster" in c]

feature_set_1 = basic_features
feature_set_2 = basic_features + squared_features + pca_features
feature_set_3 = feature_set_2 + time_features
feature_set_4 = feature_set_3 + freq_features + cluster_features

feature_sets = {
    "Basic": feature_set_1,
    "Extended": feature_set_2,
    "Temporal": feature_set_3,
    "Full": feature_set_4,
}

# --------------------------------------------------------------
# MODEL CLASS
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

# --------------------------------------------------------------
# FORWARD FEATURE SELECTION (SAFE VERSION)
# --------------------------------------------------------------
max_features = 10

selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features,
    X_train,
    y_train
)

plt.figure()
plt.plot(range(1, len(ordered_scores) + 1), ordered_scores, marker="o")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Forward Feature Selection")
plt.show()

# --------------------------------------------------------------
# MODEL EVALUATION FUNCTION
# --------------------------------------------------------------
def run_model(model_name, X_tr, y_tr, X_te, y_te):
    if model_name == "RF":
        return learner.random_forest(X_tr, y_tr, X_te, gridsearch=True)
    if model_name == "NN":
        return learner.feedforward_neural_network(X_tr, y_tr, X_te, gridsearch=False)
    if model_name == "KNN":
        return learner.k_nearest_neighbor(X_tr, y_tr, X_te, gridsearch=True)
    if model_name == "DT":
        return learner.decision_tree(X_tr, y_tr, X_te, gridsearch=True)
    if model_name == "NB":
        return learner.naive_bayes(X_tr, y_tr, X_te)

# --------------------------------------------------------------
# FEATURE SET COMPARISON
# --------------------------------------------------------------
models = ["NN", "RF", "KNN", "DT", "NB"]
results = []

for fname, fset in feature_sets.items():

    print(f"\nEvaluating feature set: {fname}")

    # ensure only valid columns exist
    fset = [f for f in fset if f in X.columns]

    X_train_sel = X_train[fset]
    X_test_sel = X_test[fset]

    for model in models:

        print("  Model:", model)

        _, y_pred, _, _ = run_model(
            model,
            X_train_sel,
            y_train,
            X_test_sel,
            y_test
        )

        acc = accuracy_score(y_test, y_pred)

        results.append({
            "feature_set": fname,
            "model": model,
            "accuracy": acc
        })

# --------------------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------------------
results_df = pd.DataFrame(results)

plt.figure()
sns.barplot(
    data=results_df,
    x="model",
    y="accuracy",
    hue="feature_set"
)
plt.ylim(0.7, 1.0)
plt.title("Model Comparison")
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# BEST MODEL (Random Forest)
# --------------------------------------------------------------
best_features = feature_set_4

_, y_pred, y_prob_train, y_prob_test = learner.random_forest(
    X_train[best_features],
    y_train,
    X_test[best_features],
    gridsearch=True
)

print("Final Accuracy:", accuracy_score(y_test, y_pred))

# --------------------------------------------------------------
# CONFUSION MATRIX (FIXED)
# --------------------------------------------------------------
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

ticks = np.arange(len(labels))
plt.xticks(ticks, labels, rotation=45)
plt.yticks(ticks, labels)

threshold = cm.max() / 2

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j, i, cm[i, j],
        ha="center",
        color="white" if cm[i, j] > threshold else "black"
    )

plt.xlabel("Predicted")
plt.ylabel("True")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# PARTICIPANT-BASED VALIDATION (NO DATA LEAKAGE)
# --------------------------------------------------------------
df_part = df.copy()

df_part = df_part.drop(columns=["set", "category"], errors="ignore")
df_part = df_part.loc[:, ~df_part.columns.str.contains("epoch")]

y = df_part["label"]
X = df_part.drop(["label", "participant"], axis=1)
X = X.select_dtypes(include=[np.number])

X_train = X[df_part["participant"] != "A"]
X_test = X[df_part["participant"] == "A"]
y_train = y[df_part["participant"] != "A"]
y_test = y[df_part["participant"] == "A"]

_, y_pred, _, _ = learner.random_forest(
    X_train[best_features],
    y_train,
    X_test[best_features],
    gridsearch=True
)

print("Participant-based accuracy:", accuracy_score(y_test, y_pred))






## PREVIOUS WORK (OUTDATED, SEE ABOVE FOR FIXED PIPELINE)
# # Plot settings
# plt.style.use("fivethirtyeight")
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.rcParams["figure.dpi"] = 100
# plt.rcParams["lines.linewidth"] = 2

# df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# # --------------------------------------------------------------
# # Create a training and test set
# # --------------------------------------------------------------

# df_train = df.drop(["participant", "category", "set"], axis=1)

# X = df_train.drop("label", axis=1)
# y = df_train["label"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

# fig, ax = plt.subplots(figsize=(10, 5))
# df_train["label"].value_counts().plot(
#     kind="bar", ax=ax, color="lightblue", label="Total"
# )
# y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
# y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
# plt.legend()
# plt.show()


# # --------------------------------------------------------------
# # Split feature subsets
# # --------------------------------------------------------------

# basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
# squared_features = ["acc_r", "gyr_r"]
# pca_features = ["pca_1", "pca_2", "pca_3"]
# time_features = [f for f in df_train.columns if "_temp_" in f]
# freq_features = [f for f in df_train.columns if (("_freq_" in f) or ("_pse" in f))]
# cluster_features = ["cluster"]

# print("Basic features:", len(basic_features))
# print("Squared features:", len(squared_features))
# print("PCA features:", len(pca_features))
# print("Time features:", len(time_features))
# print("Frequency features:", len(freq_features))
# print("Cluster features:", len(cluster_features))

# feature_set_1 = list(set(basic_features))
# feature_set_2 = list(set(basic_features + squared_features + pca_features))
# feature_set_3 = list(set(feature_set_2 + time_features))
# feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))


# # --------------------------------------------------------------
# # Perform forward feature selection using simple decision tree
# # --------------------------------------------------------------

# learner = ClassificationAlgorithms()

# max_features = 10
# selected_features, ordered_features, ordered_scores = learner.forward_selection(
#     max_features, X_train, y_train
# )

# selected_features = [
#     "acc_z_freq_0.0_Hz_ws_14",
#     "acc_x_freq_0.0_Hz_ws_14",
#     "gyr_r_pse",
#     "acc_y_freq_0.0_Hz_ws_14",
#     "acc_z_freq_0.714_Hz_ws_14",
#     "acc_r_freq_1.071_Hz_ws_14",
#     "acc_z_freq_0.357_Hz_ws_14",
#     "acc_x_freq_1.071_Hz_ws_14",
#     "acc_x_max_freq",
#     "acc_z_max_freq"
# ]

# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
# plt.xlabel("Number of Features")
# plt.ylabel("Accuracy")
# plt.xticks(np.arange(1, max_features + 1, 1))
# plt.show()

# # --------------------------------------------------------------
# # Grid search for best hyperparameters and model selection
# # --------------------------------------------------------------

# possible_feature_sets = [
#     feature_set_1,
#     feature_set_2,
#     feature_set_3,
#     feature_set_4,
#     selected_features,
# ]

# feature_names = [
#     "Feature Set 1",
#     "Feature Set 2",
#     "Feature Set 3",
#     "Feature Set 4",
#     "Selected Features"
# ]

# iterations = 1
# score_df = pd.DataFrame()


# for i, f in zip(range(len(possible_feature_sets)), feature_names):
#     print("Feature set:", i)
#     selected_train_X = X_train[possible_feature_sets[i]]
#     selected_test_X = X_test[possible_feature_sets[i]]

#     # First run non deterministic classifiers to average their score.
#     performance_test_nn = 0
#     performance_test_rf = 0

#     for it in range(0, iterations):
#         print("\tTraining neural network,", it)
#         (
#             class_train_y,
#             class_test_y,
#             class_train_prob_y,
#             class_test_prob_y,
#         ) = learner.feedforward_neural_network(
#             selected_train_X,
#             y_train,
#             selected_test_X,
#             gridsearch=False,
#         )
#         performance_test_nn += accuracy_score(y_test, class_test_y)

#         print("\tTraining random forest,", it)
#         (
#             class_train_y,
#             class_test_y,
#             class_train_prob_y,
#             class_test_prob_y,
#         ) = learner.random_forest(
#             selected_train_X, y_train, selected_test_X, gridsearch=True
#         )
#         performance_test_rf += accuracy_score(y_test, class_test_y)

#     performance_test_nn = performance_test_nn / iterations
#     performance_test_rf = performance_test_rf / iterations

#     # And we run our deterministic classifiers:
#     print("\tTraining KNN")
#     (
#         class_train_y,
#         class_test_y,
#         class_train_prob_y,
#         class_test_prob_y,
#     ) = learner.k_nearest_neighbor(
#         selected_train_X, y_train, selected_test_X, gridsearch=True
#     )
#     performance_test_knn = accuracy_score(y_test, class_test_y)

#     print("\tTraining decision tree")
#     (
#         class_train_y,
#         class_test_y,
#         class_train_prob_y,
#         class_test_prob_y,
#     ) = learner.decision_tree(
#         selected_train_X, y_train, selected_test_X, gridsearch=True
#     )
#     performance_test_dt = accuracy_score(y_test, class_test_y)

#     print("\tTraining naive bayes")
#     (
#         class_train_y,
#         class_test_y,
#         class_train_prob_y,
#         class_test_prob_y,
#     ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

#     performance_test_nb = accuracy_score(y_test, class_test_y)

#     # Save results to dataframe
#     models = ["NN", "RF", "KNN", "DT", "NB"]
#     new_scores = pd.DataFrame(
#         {
#             "model": models,
#             "feature_set": f,
#             "accuracy": [
#                 performance_test_nn,
#                 performance_test_rf,
#                 performance_test_knn,
#                 performance_test_dt,
#                 performance_test_nb,
#             ],
#         }
#     )
#     score_df = pd.concat([score_df, new_scores])


# # --------------------------------------------------------------
# # Create a grouped bar plot to compare the results
# # --------------------------------------------------------------

# score_df.sort_values(by="accuracy", ascending=False)

# plt.figure(figsize=(10, 5))
# sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# plt.ylim(0.7, 1)
# plt.legend(loc="lower right")
# plt.show()

# # --------------------------------------------------------------
# # Select best model and evaluate results
# # --------------------------------------------------------------

# (
#     class_train_y,
#     class_test_y,
#     class_train_prob_y,
#     class_test_prob_y,
# ) = learner.random_forest(
#     X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
# )

# accuracy = accuracy_score(y_test, class_test_y)
# print("Test accuracy:", accuracy)

# classes = class_train_prob_y.columns
# cm = confusion_matrix(y_test, class_test_y, labels=classes)

# # create confusion matrix for cm
# plt.figure(figsize=(10, 10))
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.title("Confusion matrix")
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# thresh = cm.max() / 2.0
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(
#         j,
#         i,
#         format(cm[i, j]),
#         horizontalalignment="center",
#         color="white" if cm[i, j] > thresh else "black",
#     )
# plt.ylabel("True label")
# plt.xlabel("Predicted label")
# plt.grid(False)
# plt.show()

# # --------------------------------------------------------------
# # Select train and test data based on participant
# # --------------------------------------------------------------

# participant_df = df.drop(["set", "category"], axis=1)

# X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
# y_train = participant_df[participant_df["participant"] != "A"]["label"]

# X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
# y_test = participant_df[participant_df["participant"] == "A"]["label"]

# X_train = X_train.drop(["participant"], axis=1)
# X_test = X_test.drop(["participant"], axis=1)

# fig, ax = plt.subplots(figsize=(10, 5))
# df_train["label"].value_counts().plot(
#     kind="bar", ax=ax, color="lightblue", label="Total"
# )
# y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
# y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
# plt.legend()
# plt.show()

# # --------------------------------------------------------------
# # Use best model again and evaluate results
# # --------------------------------------------------------------

# (
#     class_train_y,
#     class_test_y,
#     class_train_prob_y,
#     class_test_prob_y,
# ) = learner.random_forest(
#     X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
# )

# accuracy = accuracy_score(y_test, class_test_y)
# print("Test accuracy:", accuracy)

# classes = class_train_prob_y.columns
# cm = confusion_matrix(y_test, class_test_y, labels=classes)

# # create confusion matrix for cm
# plt.figure(figsize=(10, 10))
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.title("Confusion matrix")
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# thresh = cm.max() / 2.0
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(
#         j,
#         i,
#         format(cm[i, j]),
#         horizontalalignment="center",
#         color="white" if cm[i, j] > thresh else "black",
#     )
# plt.ylabel("True label")
# plt.xlabel("Predicted label")
# plt.grid(False)
# plt.show()


# # --------------------------------------------------------------
# # Try a more complex model with the selected features
# # --------------------------------------------------------------

# (
#     class_train_y,
#     class_test_y,
#     class_train_prob_y,
#     class_test_prob_y,
# ) = learner.feedforward_neural_network(
#     X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=False
# )

# accuracy = accuracy_score(y_test, class_test_y)
# print("Test accuracy:", accuracy)

# classes = class_train_prob_y.columns
# cm = confusion_matrix(y_test, class_test_y, labels=classes)

# # create confusion matrix for cm
# plt.figure(figsize=(10, 10))
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.title("Confusion matrix")
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=45)
# plt.yticks(tick_marks, classes)

# thresh = cm.max() / 2.0
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(
#         j,
#         i,
#         format(cm[i, j]),
#         horizontalalignment="center",
#         color="white" if cm[i, j] > thresh else "black",
#     )
# plt.ylabel("True label")
# plt.xlabel("Predicted label")
# plt.grid(False)
# plt.show()

