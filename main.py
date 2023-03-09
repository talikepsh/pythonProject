from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from project_evaluate import read_file, compute_metrics
from datasets import load_dataset
import pandas as pd
from transformers import DataCollatorForSeq2Seq

model_name = 't5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(dataset):
    prefix = "translate German to English: "
    inputs = [prefix + sample for sample in dataset['text']]
    targets = [sample for sample in dataset['labels']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding="max_length")
    return model_inputs


if __name__ == "__main__":
    # Reformatting the data to csv so can be used as in the tutorial.

    train_en, train_de = read_file('data/train.labeled')
    train_df = {'text': train_de, 'labels': train_en}
    train_df = pd.DataFrame(train_df)
    train_df.to_csv('train.csv', index=False)

    val_en, val_de = read_file('data/val.labeled')
    val_df = {'text': val_de, 'labels': val_en}
    val_df = pd.DataFrame(val_df)
    val_df.to_csv('val.csv', index=False)

    data_files = {
        'train': 'train.csv',
        'val': 'val.csv'
    }

    raw_datasets = load_dataset("csv", data_files=data_files)

    print(raw_datasets['train'][0])

    # tokenized_datasets = raw_datasets.map(tokenizer, input_columns='text', fn_kwargs={"max_length": 128,
    #                                                                                   "truncation": True,
    #                                                                                   "padding": "max_length"})
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets.set_format('torch')
    print(tokenized_datasets)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_total_limit=3,
        num_train_epochs=2,
        do_train=True,
        greater_is_better=True,
        save_strategy='no',
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    inputs = tokenized_datasets['val']['input_ids']
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    print()
