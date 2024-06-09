import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Specify the directories where the chunks are saved
hidden_states_dir = "hidden_states"
raw_tokens_dir = "raw_tokens"

# Define a custom dataset class
class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, raw_tokens):
        self.hidden_states = hidden_states
        self.raw_tokens = raw_tokens
        self.unique_ids = list(self.hidden_states.keys())
    
    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, idx):
        unique_id = self.unique_ids[idx]
        hidden_layer = self.hidden_states[unique_id]
        raw_tokens = self.raw_tokens[unique_id]
        return hidden_layer, raw_tokens

if __name__ == "__main__":
    
    # Load the hidden states and raw tokens from chunks
    hidden_states = {}
    for hidden_state_file in sorted([f for f in os.listdir(hidden_states_dir) if f.startswith("chunk_")]):
        hidden_state_chunk = torch.load(os.path.join(hidden_states_dir, hidden_state_file))
        hidden_state_chunk = {k: v[6] for k, v in hidden_state_chunk.items()}
        hidden_states.update(hidden_state_chunk)

    raw_tokens = {}
    for raw_token_file in sorted([f for f in os.listdir(raw_tokens_dir) if f.startswith("chunk_")]):
        raw_token_chunk = torch.load(os.path.join(raw_tokens_dir, raw_token_file))
        raw_tokens.update(raw_token_chunk)

    # decode some raw tokens
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    for i, (k, v) in enumerate(raw_tokens.items()):
        print(k, tokenizer.decode(v))
        if i == 5:
            break

    # Create an instance of the SixthHiddenStateDataset
    dataset = HiddenStateDataset(hidden_states, raw_tokens)

    # Save the entire dataset
    dataset_path = "sixth_hidden_state_dataset.pt"
    torch.save(dataset, dataset_path)

    # Load the saved dataset
    loaded_dataset = torch.load(dataset_path)

    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=False)

    # Iterate over the DataLoader
    for batch in dataloader:
        hidden_layers, raw_tokens = batch
        
        print(f"Batch Size: {len(hidden_layers)}")
        print(f"Hidden State Shape: {hidden_layers.shape}")
        print(f"Raw Tokens Shape: {raw_tokens.shape}")
        print()