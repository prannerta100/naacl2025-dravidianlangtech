import pandas as pd
import json
from make_chatgpt_calls import make_open_ai_request
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import asyncio
import sys
# Load datasets
multi_class_data = pd.read_csv(
    "/Users/4760393/Downloads/DravidianLangTechData/Fake News/fake_news_classification_mal_train.csv"
)
multi_class_data["Label"] = multi_class_data["Label"].apply(lambda x: x.strip())
class_names = sorted(list(multi_class_data["Label"].unique()))
train_data, _ = train_test_split(
    multi_class_data, test_size=0.3, random_state=42
)
valid_data = pd.read_csv("/Users/4760393/Downloads/fake_test_multiclass_labeled.csv")

print(class_names)

text_col = "News"  # "Translation"
label_col = "Label"

# Combine all training data as context for the GPT model
training_context = "\n".join(
    [
        f"Text: {row[text_col]} | Label: {row[label_col]}"
        for _, row in train_data.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=5, random_state=42))
        .iterrows()
    ]
)


# Function to predict classes in batches of 5
async def predict_with_gpt4o(data, training_context, batch_size=20):
    predictions = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i : i + batch_size]
        prompt = f"You are a classifier. Use the training data below to classify each text as {class_names}, output only a json that is a list of records with fields '{text_col}' and 'prediction':\n\n"
        prompt += "Training Data:\n" + training_context + "\n\n"
        prompt += "Data to Classify:\n"
        prompt += "\n".join([f"Text: {row[text_col]}" for _, row in batch.iterrows()])

        # Prepare API request
        request_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an NLP expert helping classify fake news in Kerala. Before outputting, you will think what the text means within the cultural context of a Malayali. The categories like false, half true, etc. will tell how trustworthy the news text is. For example, 'half true' means the text is half true. Follow reasoning like this:
                        1. Think about what this sentence means, and put in the larger societal context of Kerala.
                        2. Revisit the training examples, and check whether your prediction agrees with the kind of labels that the training examples have.
                        3. Make sure you choose your final answer after carefully weighing the possibilities, for example, is it 'mostly false' or 'false'.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.000001,
        }

        # Call GPT-4o API
        try:
            model_response = await make_open_ai_request(
                "/chat/completions", request_data
            )
            print(
                "gpt answer: \n",
                model_response.json()["choices"][0]["message"]["content"],
            )
            gpt_response = "\n".join(
                model_response.json()["choices"][0]["message"]["content"].split("\n")[
                    1:-1
                ]
            )
            # Extract predictions
            batch_predictions = json.loads(gpt_response)
            predictions.extend(batch_predictions)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            predictions.extend(
                [{f"{text_col}": row[text_col], "prediction": "FALSE"} for _, row in batch.iterrows()]
                # [{f"{text_col}": "Error", "prediction": "Error"}] * len(batch)
            )

    return predictions


# Predict on train and validation datasets
# train_predictions = predict_with_gpt4o(train_data, training_context)
async def predict():
    res = await predict_with_gpt4o(valid_data, training_context, batch_size=20)
    return pd.DataFrame(res)


def evaluate_predictions(data, predictions, dataset_name):
    # Remove any "Error" predictions to avoid skewing results
    merged = pd.merge(data, predictions, on=text_col)
    merged = merged.loc[merged["prediction"] != "Error"]

    # Calculate metrics
    accuracy = accuracy_score(merged[label_col], merged["prediction"])
    report = classification_report(
        merged[label_col],
        merged["prediction"],
        # target_names=class_names,
        output_dict=True,
    )
    f1 = report["macro avg"]["f1-score"]
    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]

    print(f"--- {dataset_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}\n")

    return accuracy, f1, precision, recall


# Evaluate predictions and calculate metrics
valid_predictions = asyncio.run(
    predict()
)  # pd.read_csv("valid_predictions.csv") # asyncio.run(predict()) #

# Save predictions and metrics
# train_data["Predicted_Class"] = train_predictions
pd.merge(valid_data, valid_predictions, on=text_col).to_csv("valid_predictions.tsv", index=False, sep="\t")


# Train dataset metrics
print("Evaluating Train Dataset")
# train_metrics = evaluate_predictions(train_data, train_predictions, "Train")

# Validation dataset metrics
print("Evaluating Validation Dataset")
valid_metrics = evaluate_predictions(valid_data, valid_predictions, "Validation")

# print("Predicting on test dataset")


# with open("metrics.json", "w") as f:
#     json.dump({"Train": train_metrics, "Validation": valid_metrics}, f)

# print("Predictions and metrics saved!")
