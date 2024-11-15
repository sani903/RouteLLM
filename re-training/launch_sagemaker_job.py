import sagemaker
from sagemaker.huggingface import HuggingFace
from huggingface_hub import login
login(token="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")

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::423623863398:role/firstrun'

hyperparameters = {
    'model_id': 'meta-llama/Meta-Llama-3-8B',
    'context_length': 2048,
    'num_epochs': 1,  # Adjust based on your dataset size to achieve ~2000 steps
    'train_batch_size_per_device': 1,  # Total batch size of 8 across 4 GPUs
    'eval_batch_size_per_device': 1,
    'learning_rate': 1e-6,
    'lr_scheduler_type': 'constant',
    'flash_attention_2': 'True',
    'no_gradient_checkpoint': 'False',
}
hyperparameters.update({
    'classifier_config': {
        'label_tokens': ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    },
    'output_dir': '/opt/ml/model'
})
huggingface_estimator = HuggingFace(
    entry_point='sagemaker_train.py',
    source_dir='.',
    # instance_type='ml.g5.48xlarge',
    instance_type='ml.p4d.24xlarge',  # 4 NVIDIA A10G GPUs
    instance_count=1,  # Single instance for cost-effectiveness
    role=role,
    transformers_version='4.36.0',
    pytorch_version='2.1.0',
    py_version='py310',
    hyperparameters=hyperparameters,
    distribution={
        "torch_distributed": {
            "enabled": True
        }
    },
    enable_sagemaker_metrics=True,
    model_cache_root='/opt/ml/model',
    use_spot_instances=True,
    max_run=7200,
    max_wait=10000,
    checkpoint_s3_uri='s3://anlp/checkpoints',
    checkpoint_local_path='/opt/ml/checkpoints',
    environment={
        "HUGGING_FACE_TOKEN": "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",
        "HF_HUB_CACHE": "s3://anlp/model-cache"
    }
)

data_channels = {
    'train': 's3://anlp/train_split.jsonl',
    'validation': 's3://anlp/validation_split.jsonl'
}

huggingface_estimator.fit(data_channels)