import pandas as pd
import json
from make_chatgpt_calls import make_open_ai_request
from sklearn.metrics import classification_report, accuracy_score
import asyncio

# Load datasets
train_data = pd.read_csv(
    "/Users/4760393/Downloads/DravidianLangTechData/Fake News/Fake_train.csv"
)
valid_data = pd.read_csv(
    "/Users/4760393/Downloads/DravidianLangTechData/Fake News/Fake_dev.csv"
)

train_data["label"] = train_data["label"].str.lower()
valid_data["label"] = valid_data["label"].str.lower()

# Combine all training data as context for the GPT model
training_context = "\n".join(
    [
        f"Text: {row['text']} | Label: {row['label']}"
        for _, row in train_data.sample(n=100, random_state=42).iterrows()
    ]
)


# Function to predict classes in batches of 5
async def predict_with_gpt4o(data, training_context, batch_size=20):
    predictions = []

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i : i + batch_size]
        prompt = "You are a classifier. Use the training data below to classify each text as 'original' or 'fake', output only a json that is a list of records with fields 'text' and 'prediction':\n\n"
        prompt += "Training Data:\n" + training_context + "\n\n"
        prompt += "Data to Classify:\n"
        prompt += "\n".join([f"Text: {row['text']}" for _, row in batch.iterrows()])

        # Prepare API request
        request_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an NLP expert helping classify Malayalam fake news. Before outputting, you will think what the text means within the cultural context of a Malayalam speaker.",
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
            predictions.extend([{"text": "Error", "prediction": "Error"}] * len(batch))

    return predictions


# Predict on train and validation datasets
# train_predictions = predict_with_gpt4o(train_data, training_context)
async def predict():
    res = await predict_with_gpt4o(valid_data, training_context, batch_size=20)
    return pd.DataFrame(res)


def evaluate_predictions(data, predictions, dataset_name):
    # Remove any "Error" predictions to avoid skewing results
    merged = pd.merge(data, predictions, on="text")
    merged = merged.loc[merged["prediction"] != "Error"]

    # Calculate metrics
    accuracy = accuracy_score(merged["label"], merged["prediction"])
    report = classification_report(
        merged["label"],
        merged["prediction"],
        target_names=["original", "fake"],
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
valid_predictions.to_csv("valid_predictions.csv", index=False)


# Train dataset metrics
print("Evaluating Train Dataset")
# train_metrics = evaluate_predictions(train_data, train_predictions, "Train")

# Validation dataset metrics
print("Evaluating Validation Dataset")
valid_metrics = evaluate_predictions(valid_data, valid_predictions, "Validation")


# with open("metrics.json", "w") as f:
#     json.dump({"Train": train_metrics, "Validation": valid_metrics}, f)

print("Predictions and metrics saved!")
