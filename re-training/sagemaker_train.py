import argparse
import json
import os
import io
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from router_config import RouterModelConfig, ModelTypeEnum
from llm_utils import load_prompt_format, get_model, get_tokenizer
import torch
import torch.nn.functional as F
from sagemaker.huggingface import HuggingFaceModel
from huggingface_hub import login
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import boto3
from botocore.exceptions import ClientError
from transformers import TrainingArguments, Trainer
import deepspeed
import s3fs
from transformers import BitsAndBytesConfig
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
login(token="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")
os.environ['HF_HOME'] = '/opt/ml/model/huggingface'
    
def setup_distributed():
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = '/opt/ml/checkpoints'

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir, _internal_call)

    def _save(self, output_dir= None, state_dict=None):
        output_dir = output_dir or self.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)
        return super()._save(output_dir, state_dict)
    def _move_model_to_device(self, model, device):
        # Do nothing, as the model is already distributed
        pass

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        loss = custom_loss_function(logits, labels, self.tokenizer)
        return (loss, outputs) if return_outputs else loss
def custom_loss_function(logits, labels, tokenizer):
    # Get the indices of the special tokens
    special_token_ids = tokenizer.convert_tokens_to_ids(["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"])

    # Extract logits for special tokens
    special_token_logits = logits[:, special_token_ids]

    # Apply softmax to get probabilities
    probabilities = F.softmax(special_token_logits, dim=-1)

    # Sum probabilities for 0 (tokens 1, 2, 3) and 1 (tokens 4, 5)
    prob_1 = probabilities[:, :3].sum(dim=1)
    prob_0 = probabilities[:, 3:].sum(dim=1)

    # Combine probabilities
    combined_probs = torch.stack([prob_0, prob_1], dim=1)

    # Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy(combined_probs, labels.float())

    return loss

def load_config():
    config = {
        'model_id': 'meta-llama/Meta-Llama-3-8B',
        'context_length': 2048,
        'num_epochs': 1,
        # 'train_batch_size_per_device': 1,
        # 'eval_batch_size_per_device': 1,
        'train_batch_size_per_device': 1,
        'eval_batch_size_per_device': 1,
        'learning_rate': 1e-6,
        'lr_scheduler_type': 'constant',
        'flash_attention_2': True,
        'no_gradient_checkpoint': False,
        'classifier_config': {
            'label_tokens': ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
        },
        'huggingface_token': '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',
        # 'huggingface_token': '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'
        'train_path': 's3://anlp/train_split.jsonl',
        'valid_path': 's3://anlp/validation_split.jsonl',
        'output_dir': '/opt/ml/model',  # Default SageMaker model output directory
    }
    # 'model_id': 'meta-llama/Meta-Llama-3-8B',
    #     'context_length': 2048,
    #     'num_epochs': 1,
    #     'train_batch_size_per_device': 2,  # Increased from 1
    #     'eval_batch_size_per_device': 2,   # Increased from 1
    #     'learning_rate': 1e-5,  # Slightly increased from 1e-6
    #     'lr_scheduler_type': 'cosine',  # Changed from 'constant' for better convergence
    #     'flash_attention_2': True,
    #     'gradient_checkpointing': True,  # Renamed from 'no_gradient_checkpoint'
    #     'classifier_config': {
    #         'label_tokens': ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    #     },
    #     'huggingface_token': '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',
    #     'train_path': 's3://anlp/train_split.jsonl',
    #     'valid_path': 's3://anlp/validation_split.jsonl',
    #     'output_dir': '/opt/ml/model',
    #     'fp16': True,  # Enable mixed precision training
    #     'gradient_accumulation_steps': 4,  # Add gradient accumulation
    #     'warmup_steps': 100,  # Add warmup steps
    #     'max_grad_norm': 1.0,  # Add gradient clipping
    #     'save_strategy': 'epoch',
    #     'evaluation_strategy': 'epoch',
    #     'load_best_model_at_end': True,
    #     'metric_for_best_model': 'eval_loss',
    #     'greater_is_better': False,
    #     'logging_steps': 10,
    #     'logging_dir': '/opt/ml/output/logs'
    return config

def prepare_model_and_tokenizer(config):
    s3_bucket = 'anlp'
    s3_prefix = 'checkpoints'
    local_cache_dir = '/opt/ml/checkpoints'
    os.makedirs(local_cache_dir, exist_ok=True)

    s3 = boto3.client('s3')
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
        if 'Contents' in response:
            print("Loading model from S3 cache...")
            for obj in response['Contents']:
                file_key = obj['Key']
                local_file_path = os.path.join(local_cache_dir, os.path.basename(file_key))
                s3.download_file(s3_bucket, file_key, local_file_path)
            
            model = AutoModelForCausalLM.from_pretrained(
                local_cache_dir,
                torch_dtype=torch.bfloat16,
                use_cache=False,
                quantization_config=quantization_config,
                device_map="auto",
                use_safetensors=True,
                trust_remote_code=True
            )
        else:
            print("Downloading model from Hugging Face...")
            model = AutoModelForCausalLM.from_pretrained(
                config['model_id'],
                torch_dtype=torch.bfloat16,
                use_cache=False,
                token=config['huggingface_token'],
                # device_map=None,
                use_safetensors=True,
                trust_remote_code=True
            )
            # Save model to local cache
            model.save_pretrained(local_cache_dir, safe_serialization=True, max_shard_size="4GB")
            # model.save_pretrained(local_cache_dir, safe_serialization=True)
            
            # Upload model files to S3
            for root, _, files in os.walk(local_cache_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    s3_key = f"{s3_prefix}/{file}"
                    s3.upload_file(local_file_path, s3_bucket, s3_key)
            if not os.path.exists(os.path.join(local_cache_dir, 'config.json')):
                print("Model files were not saved correctly.")
            
            # print("Moving model to GPU...")
            # model = model.to(device_map="auto")
            
            print("Model preparation complete.")
    except Exception as e:
        print(f"Error accessing S3: {e}")
        print("Downloading model from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained(
            config['model_id'],
            torch_dtype=torch.bfloat16,
            use_cache=False,
            token=config['huggingface_token'],
            # device_map="auto",
            use_safetensors=True,
            trust_remote_code=True
        )

    tokenizer = get_tokenizer(
        config['model_id'],
        special_tokens=config['classifier_config']['label_tokens'],
        truncation_side="left",
        padding_side="left",
        cache_dir=local_cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def prepare_dataset(config):
    s3 = boto3.client('s3')
    local_train_path = '/tmp/train_split.jsonl'
    local_valid_path = '/tmp/validation_split.jsonl'

    def download_from_s3(s3_path, local_path):
        try:
            bucket, key = s3_path.replace("s3://", "").split("/", 1)
            s3.download_file(bucket, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading {s3_path}: {e}")
            return False

    # Download files from S3
    download_from_s3(config['train_path'], local_train_path)
    download_from_s3(config['valid_path'], local_valid_path)

    try:
        dataset = load_dataset('json', 
                               data_files={
                                   'train': local_train_path,
                                   'validation': local_valid_path
                               },streaming=True)
        
        if len(dataset['train']) == 0 or len(dataset['validation']) == 0:
            raise ValueError("Dataset is empty. Check your data files.")
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# def prepare_dataset(config, tokenizer, prompt_format):
#     s3 = boto3.client('s3')
#     local_train_path = '/tmp/train_split.jsonl'
#     local_valid_path = '/tmp/validation_split.jsonl'

#     def download_from_s3(s3_path, local_path):
#         try:
#             bucket, key = s3_path.replace("s3://", "").split("/", 1)
#             s3.download_file(bucket, key, local_path)
#             return True
#         except ClientError as e:
#             print(f"Error downloading {s3_path}: {e}")
#             return False

#     # Try to download from S3 first
#     s3_success = download_from_s3(config['train_path'], local_train_path) and \
#                  download_from_s3(config['valid_path'], local_valid_path)

#     # If S3 download fails, use local paths
#     if not s3_success:
#         local_train_path = os.path.join('/opt/ml/input/data', config['train_path'])
#         local_valid_path = os.path.join('/opt/ml/input/data', config['valid_path'])

#     try:
#         dataset = load_dataset('json', 
#                                data_files={
#                                    'train': local_train_path,
#                                    'validation': local_valid_path
#                                })
        
#         if len(dataset['train']) == 0 or len(dataset['validation']) == 0:
#             raise ValueError("Dataset is empty. Check your data files.")
        
#         tokenized_dataset = dataset.map(
#             preprocess_function,
#             fn_kwargs={'tokenizer': tokenizer, 'config': config},
#             batched=True,
#             remove_columns=dataset["train"].column_names,
#             batch_size=100
#         )
#         return tokenized_dataset
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         # Here you could implement a fallback method or raise the exception
#         raise


def main(config):
    setup_distributed()
    # dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    model, tokenizer = prepare_model_and_tokenizer(config)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    dataset = prepare_dataset(config)
    # ds_config = {
    #     "fp16": {
    #         "enabled": True
    #     },
    #     "bf16": {
    #         "enabled": False
    #     },
    #     "zero_optimization": {
    #         "stage": 3,
    #         "offload_optimizer": {
    #             "device": "cpu",
    #             "pin_memory": True
    #         },
    #         "offload_param": {
    #             "device": "cpu",
    #             "pin_memory": True
    #         },
    #         "overlap_comm": True,
    #         "contiguous_gradients": True,
    #         "sub_group_size": 1e9,
    #         "reduce_bucket_size": 5e8,
    #         "stage3_prefetch_bucket_size": 5e8,
    #         "stage3_param_persistence_threshold": 1e6,
    #         "stage3_max_live_parameters": 1e9,
    #         "stage3_max_reuse_distance": 1e9,
    #         "stage3_gather_16bit_weights_on_model_save": True
    #     },
    #     "gradient_accumulation_steps": 4,
    #     "gradient_clipping": 1.0,
    #     "train_batch_size": 32,
    #     "train_micro_batch_size_per_gpu": 2,
    #     "wall_clock_breakdown": False,
    #     "tensor_parallel": {
    #         "size": 4
    #     },
    #     "pipeline_parallel": {
    #         "size": 4
    #     }
    # }
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # ds_config = json.loads(os.environ['SM_HP_DEEPSPEED_CONFIG'])
    # model_engine, _, _, _ = deepspeed.initialize(
    #     model=model,
    #     config_params=ds_config,
    #     model_parameters=model.parameters()
    # )
    


    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['train_batch_size_per_device'],
        per_device_eval_batch_size=config['eval_batch_size_per_device'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_steps=100,
        # deepspeed=ds_config,
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        gradient_checkpointing=not config['no_gradient_checkpoint'],
        fp16=True,
        # Add these lines
        use_cpu=False,
        use_ipex=False,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    # model_engine, optimizer, _, _ = deepspeed.initialize(
    #     model=trainer.model,
    #     config_params=ds_config,
    #     model_parameters=trainer.model.parameters()
    # )
    # trainer.model = model_engine
    # trainer.optimizer = optimizer
    
    
    checkpoint_dir = '/opt/ml/checkpoints'
    if os.path.exists(checkpoint_dir) and any(f.startswith('checkpoint-') for f in os.listdir(checkpoint_dir)):
        print("Resuming from checkpoint...")
        latest_checkpoint = max(
            [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-')],
            key=lambda x: int(x.split('-')[1])
        )
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("Starting new training run...")
        trainer.train()


    trainer.save_model(config['output_dir'])

if __name__ == "__main__":
    config = load_config()
    main(config)
