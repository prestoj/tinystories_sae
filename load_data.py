from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import os
import gc

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the wikitext dataset
    dataset = load_dataset("roneneldan/TinyStories")

    model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Define the batch size and desired sequence length
    batch_size = 512
    sequence_length = 64

    # Concatenate all the text samples into a single string
    def concatenate_text(examples):
        return {"text": ["".join(examples["text"])]}

    concatenated_dataset = dataset.map(concatenate_text, batched=True, num_proc=4)

    # Tokenize the concatenated text
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    # print(len(concatenated_dataset["validation"]))
    # for i in range(len(concatenated_dataset["validation"])):
    #     print(i, concatenated_dataset["validation"][i]["text"])

    tokenized_dataset = concatenated_dataset["validation"].map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Split the tokenized text into sequences of the desired length
    def split_sequences(examples):
        input_ids = []
        attention_masks = []
        for ids in examples["input_ids"]:
            for i in range(0, len(ids), sequence_length):
                if i + sequence_length > len(ids):
                    break
                input_ids.append(ids[i : i + sequence_length])
                attention_masks.append([1] * len(input_ids[-1]))
        return {"input_ids": input_ids, "attention_mask": attention_masks}

    split_dataset = tokenized_dataset.map(split_sequences, batched=True, num_proc=4, remove_columns=["input_ids"])

    # print(len(split_dataset))
    # for i in range(len(split_dataset)):
    #     print(i, len(split_dataset[i]["input_ids"]))
    #     # decode the tokenized text
    #     decoded = tokenizer.decode(split_dataset[i]["input_ids"])
    #     print(decoded)
    #     if i == 10:
    #         break

    def collate_fn(examples):
        input_ids = torch.LongTensor([example["input_ids"] for example in examples])
        attention_mask = torch.LongTensor([example["attention_mask"] for example in examples])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    dataloader = torch.utils.data.DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(len(dataloader))

    # Initialize dictionaries to store the mappings
    id_to_hidden_states = {}
    id_to_raw_tokens = {}

    # Initialize chunk counter
    chunk_counter = 0

    # Process the dataset in batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}, len(id_to_hidden_states) = {len(id_to_hidden_states)}")
        # print(batch["input_ids"].shape)
        # for i in range(10):
        #     print(i, len(batch["input_ids"][i]), tokenizer.decode(batch["input_ids"][i]))
    
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = [hidden_state.cpu().to(torch.float16) for hidden_state in outputs.hidden_states]

        # Move tensors to CPU and free up GPU memory
        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()
        del outputs
        torch.cuda.empty_cache()

        # Generate unique identifiers for each data point in the batch
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            unique_id = f"batch_{batch_idx}_sample_{i}"
            id_to_hidden_states[unique_id] = [hidden_state[i].numpy() for hidden_state in hidden_states]
            id_to_raw_tokens[unique_id] = input_ids[i].numpy().tolist()

        # for i in range(10):
        #     print(i, len(id_to_raw_tokens[f"batch_{batch_idx}_sample_{i}"]), tokenizer.decode(id_to_raw_tokens[f"batch_{batch_idx}_sample_{i}"]))

        # print('-'*50)
        # Free up memory
        del input_ids
        del hidden_states
        gc.collect()

        if len(id_to_hidden_states) >= 10000:
            chunk_counter += 1
            torch.save(id_to_hidden_states, f"hidden_states/chunk_{chunk_counter}.pt")
            torch.save(id_to_raw_tokens, f"raw_tokens/chunk_{chunk_counter}.pt")
            # Clear the dictionaries
            id_to_hidden_states.clear()
            id_to_raw_tokens.clear()
            gc.collect()

    # Save the final chunk if any
    if len(id_to_hidden_states) > 0:
        chunk_counter += 1
        torch.save(id_to_hidden_states, f"hidden_states/chunk_{chunk_counter}.pt")
        torch.save(id_to_raw_tokens, f"raw_tokens/chunk_{chunk_counter}.pt")