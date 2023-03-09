from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from project_evaluate import read_file, compute_metrics
import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def get_inputs(file_en, file_de):
    inputs = [prefix + line for line in file_en]
    targets = [line for line in file_de]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


if __name__ == "__main__":
    model_name = 'google/mt5-base'
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    model.train()

    train_dataset = list(zip(*read_file('data/train.labeled')))

    train_input_ids = []
    train_attention_masks = []
    train_labels = []

    for de_sentence, en_sentence in train_dataset:
        input_ids = tokenizer.encode(de_sentence, padding='max_length', truncation=True, return_tensors='pt')
        label_ids = tokenizer.encode(en_sentence, padding='max_length', truncation=True, return_tensors='pt')
        attention_mask = input_ids != tokenizer.pad_token_id

        train_input_ids.append(input_ids)
        train_labels.append(label_ids)
        train_attention_masks.append(attention_mask)

    train_input_ids = torch.cat(train_input_ids, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    train_attention_masks = torch.cat(train_attention_masks, dim=0)

    optimizer = torch.optim.Adam(model.parameters, lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        pbar = tqdm(range(len(train_dataset)))
        for i in pbar:
            input_ids = train_input_ids[i].unsqueeze(0)
            label_ids = train_labels[i].unsqueeze(0)
            attention_mask = train_attention_masks.unsqueeze(0)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, labels=label_ids, attention_mask=attention_mask)

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            pbar.set_description(f'Loss: {loss.item}')
