import json
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import boto3
import os

def tokenize_and_upload(config):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        legacy=True,
        truncation_side="left",
        padding_side="left",
        token="",
    )
    
    # Load dataset
    dataset = load_dataset('json', data_files={
        'train': config['train_path'],
        'validation': config['valid_path']
    })

    def tokenize_function(examples):
        system_message = "[Instruction]\nBased on the question provided below, predict the score an expert evaluator would give to an AI assistant's response, considering its helpfulness, relevance, adherence to facts, depth, creativity, and detail. Your prediction should infer the level of proficiency needed to address the question effectively. Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. Provide your prediction as: \"[[predicted rating]]\".\n\nScore criteria:\n- **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.\n- **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.\n- **1-2**: The AI assistant will struggle to produce a strong answer due to the question's difficulty, vagueness, or the assistant's limitations."
        classifier_message = "\n[Question]\n{question}\n\nPrediction:\n"

        prompts = []
        for question in examples['prompt']:
            parsed_prompt = json.loads(question)[0]
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": classifier_message.format(question=parsed_prompt)}
            ]
            prompt = "".join(turn["content"] for turn in messages)
            prompts.append(prompt)
        tokenizer.pad_token = tokenizer.eos_token
        model_inputs = tokenizer(
            prompts,
            max_length=config['context_length'],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        model_inputs["labels"] = torch.tensor(examples['label'], dtype=torch.long)
        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=100
    )

    # Save tokenized dataset locally
    tokenized_dataset.save_to_disk("tokenized_dataset")

    # Upload to S3
    s3 = boto3.client('s3')
    for split in ['train', 'validation']:
        for filename in os.listdir(f"tokenized_dataset/{split}"):
            file_path = f"tokenized_dataset/{split}/{filename}"
            s3.upload_file(file_path, 'anlp', f'tokenized_data/{split}/{filename}')

if __name__ == "__main__":
    config = {
        'model_id': 'meta-llama/Meta-Llama-3-8B',
        'context_length': 2048,
        'train_path': 'train_split.jsonl',
        'valid_path': 'validation_split.jsonl',
    }
    tokenize_and_upload(config)
