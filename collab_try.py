from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import pandas as pd
from transformers import DataCollatorForSeq2Seq
import torch
import evaluate

model_name = 't5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def read_file(file_path):
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str)
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def compute_metrics(tup):
    tagged_en, true_en = tup
    metric = evaluate.load("sacrebleu")
    # metric = evaluate.load("accuracy")
    tagged_en = [x.strip().lower() for x in tagged_en]
    true_en = [x.strip().lower() for x in true_en]

    result = metric.compute(predictions=tagged_en, references=true_en)
    result = result['score']
    result = round(result, 2)
    return result


def preprocess_function(dataset):
    prefix = "translate German to English: "
    inputs = [prefix + sample for sample in dataset['text']]
    targets = [sample for sample in dataset['labels']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True, padding="max_length")
    return model_inputs


# Reformatting the data
train_en, train_de = read_file('train.labeled')

train_en = train_en[:1000]
train_de = train_de[:1000]

train_df = {'text': train_de, 'labels': train_en}
train_df = pd.DataFrame(train_df)
train_df.to_csv('train.csv', index=False)

val_en, val_de = read_file('val.labeled')

val_en = val_en[:100]
val_de = val_de[:100]

val_df = {'text': val_de, 'labels': val_en}
val_df = pd.DataFrame(val_df)
val_df.to_csv('val.csv', index=False)

data_files = {
    'train': 'train.csv',
    'val': 'val.csv'
}

raw_datasets = load_dataset("csv", data_files=data_files)

print(raw_datasets['train'][0])

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
tokenized_datasets.set_format('torch')
print(tokenized_datasets)


# Train section
training_args = Seq2SeqTrainingArguments(
    output_dir="try",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
    data_collator=data_collator
)

trainer.train()

# Try to translate one sentence
input = tokenizer('translate German to English: ' + val_de[0], return_tensors="pt").input_ids
cuda = torch.device('cuda')
model = model.to(device=cuda)
input = input.to(device=cuda)
outputs = model.generate(input, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
tokenizer.decode(outputs[0], skip_special_tokens=True)