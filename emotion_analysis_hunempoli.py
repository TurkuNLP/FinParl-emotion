import logging
import mlflow
import datasets
import transformers
import evaluate
import pandas as pd
from pprint import pprint
import os
import optuna

# Setup logging
logging.basicConfig(level=logging.INFO,  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s [%(levelname)s] - %(message)s'
                   )

# Give model id
MODEL_ID = "hunempoli_model_1"
print(f"Training model {MODEL_ID}")

# ML-flow setup
mlflow.set_tracking_uri("/scratch/project_2006385/otto/sentiment-analysis/mlruns")
mlflow.autolog()
os.environ["MLFLOW_EXPERIMENT_NAME"] = "mlflow_sentiment_analysis"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
mlflow.start_run(run_name=os.getenv(MODEL_ID))



#Load dataset
filepath = "../data/HunEmPoli_fi/"
dataset = datasets.load_dataset("csv", data_files={"train": os.path.join(filepath, "HunEmPoli_fi_train.tsv"),
                                                   "validation": os.path.join(filepath, "HunEmPoli_fi_validation.tsv"),
                                                   "test": os.path.join(filepath, "gpt4_annotations_hun_labels_test.tsv"),
                                                  },
                                sep = "\t",
                                cache_dir = "../../hf_cache"
                               )

label_names = ["neutral", "fear", "sadness", "anger-digsust", "success-joy", "trust"]
num_labels = len(label_names)
id2label = { k: v for k, v in enumerate(label_names) }
label2id = { v: k for k, v in enumerate(label_names) }


#Tokenize and vectorize dataset
MODEL = "TurkuNLP/bert-base-finnish-cased-v1"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, cache = "../../hf_cache/")

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length = 512,
        truncation = True,
    )

print("Tokenizing data...")
dataset = dataset.map(tokenize)

# Instantiate model
#def model_init():
#    return transformers.AutoModelForSequenceClassification.from_pretrained(
#        MODEL,
#        num_labels=num_labels,
#        id2label=id2label,
#        label2id=label2id,
#    )

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)


# Evaluation function
f1_metric = evaluate.load("f1")
def compute_f1(outputs_and_labels):
    outputs, labels = outputs_and_labels
    predictions = outputs.argmax(axis=-1) # pick index of "winning" label
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    micro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")
    return {"macro_f1": macro_f1["f1"], "micro_f1": micro_f1["f1"]}


# Training configuration
trainer_args = transformers.TrainingArguments(
    output_dir=f"../models/{MODEL_ID}",
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    save_total_limit=1, # Only keep the last and best checkpoints
    load_best_model_at_end=True,
    disable_tqdm=True, # Disable progress bar to make logs more readable
    warmup_steps=500,
    label_smoothing_factor=0.2,
    learning_rate=3.89e-05,
    per_device_train_batch_size=16,
    max_steps= 20_000,
)

# Data collator to pad short input
data_collator = transformers.DataCollatorWithPadding(tokenizer)

# Setup trainer.
trainer = transformers.Trainer(
    model=model,
    #model_init=model_init,
    args=trainer_args,
    train_dataset=dataset['train'],#.shard(index=1, num_shards=10),
    eval_dataset=dataset['validation'],
    compute_metrics=compute_f1,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# Define hyperparameter space for ray hyperparameter search
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),
        "label_smoothing_factor":trial.suggest_categorical("label_smoothing_factor", [0.1, 0.2]),
        "warmup_steps":trial.suggest_categorical("warmup_steps", [0, 500, 1000]),
    }


# Train (fine-tune) model
#best_trial = trainer.hyperparameter_search(
#    direction="maximize",
#    backend="optuna",
#    hp_space=optuna_hp_space,
#    n_trials=50,
# )

print("Training...")
trainer.train()

# Evaluate trained model
print("Evaluating trained model on test set...")
model.eval() # Set model in evaluation mode
test_results = trainer.predict(dataset['test'])
pprint("Model evaluation on test set:")
pprint(test_results[2])

with open(f'../models/evaluation_{MODEL_ID}.txt', 'w') as f:
    f.write('Macro-f1: ')
    f.write(f'{test_results[2]["test_macro_f1"]}\n')
    f.write('Micro-f1: ')
    f.write(f'{test_results[2]["test_micro_f1"]}\n')
    f.write('Loss: ')
    f.write(f'{test_results[2]["test_loss"]}\n')
