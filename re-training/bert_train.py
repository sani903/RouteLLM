import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import json
from tqdm import tqdm

class PreferenceDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        if isinstance(prompt, list):
            prompt = ' '.join(prompt)
        
        inputs = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

def train(model, train_dataloader, optimizer, scheduler, device, total_steps):
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    for step, batch in enumerate(train_dataloader):
        if step >= total_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / total_steps

def main():
    # Hyperparameters
    model_name = "bert-base-uncased"
    num_labels = 3
    max_length = 512
    batch_size = 16
    learning_rate = 1e-5
    weight_decay = 0.01
    total_steps = 2000
    data_file = "preference_data.jsonl"  # Your data file path
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    
    # Prepare dataset and dataloader
    dataset = PreferenceDataset(data_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Training loop
    print("Starting training...")
    train_loss = train(model, dataloader, optimizer, scheduler, device, total_steps)
    print(f"Training completed. Final loss: {train_loss:.4f}")
    
    # Save the model
    model.save_pretrained("./bert_preference_model")
    tokenizer.save_pretrained("./bert_preference_model")

if __name__ == "__main__":
    main()