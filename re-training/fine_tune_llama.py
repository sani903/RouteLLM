import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from router_config import RouterModelConfig, ModelTypeEnum
from llm_utils import load_prompt_format, get_model, get_tokenizer

def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'],
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2" if config['flash_attention_2'] else None,
        token="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"
    )
    tokenizer = get_tokenizer(
        config['model_id'],
        special_tokens=config['classifier_config']['label_tokens'],
        truncation_side="left",
        padding_side="left"
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def prepare_dataset(config, tokenizer, prompt_format):
    def preprocess_function(examples):
        system_message = "[Instruction]\nBased on the question provided below, predict the score an expert evaluator would give to an AI assistant's response, considering its helpfulness, relevance, adherence to facts, depth, creativity, and detail. Your prediction should infer the level of proficiency needed to address the question effectively. Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. Provide your prediction as: \"[[predicted rating]]\".\n\nScore criteria:\n- **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.\n- **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.\n- **1-2**: The AI assistant will struggle to produce a strong answer due to the question's difficulty, vagueness, or the assistant's limitations."
        classifier_message = "\n[Question]\n{question}\n\nPrediction:\n"
        
        messages = [{"role": "system", "content": system_message}]
        for question in examples['prompt']:
            messages.append({"role": "user", "content": classifier_message.format(question=question)})
        
        prompts = [prompt_format.generate_prompt(msg) for msg in messages]
        model_inputs = tokenizer(prompts, max_length=config['context_length'], truncation=True, padding="max_length")
        
        labels = tokenizer(examples['label'], max_length=5, truncation=True, padding="max_length")["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    dataset = load_dataset('json', data_files={'train': config['train_path'], 'validation': config['valid_path']})
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_dataset

def main(config_path):
    config = load_config(config_path)
    model, tokenizer = prepare_model_and_tokenizer(config)
    prompt_format = load_prompt_format(config['model_id'])
    
    dataset = prepare_dataset(config, tokenizer, prompt_format)
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['train_batch_size_per_device'],
        per_device_eval_batch_size=config['eval_batch_size_per_device'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_steps=100,
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=not config['no_gradient_checkpoint'],
        fp16=True,
        deepspeed=config['deepspeed']['config_path'],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(config['output_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama model for routing")
    parser.add_argument("config_path", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()
    main(args.config_path)