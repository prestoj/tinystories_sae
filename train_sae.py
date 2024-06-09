import torch
from torch.utils.data import DataLoader
from load_data_pt2 import HiddenStateDataset
from transformers import AutoTokenizer
import torch.optim as optim
import torch.nn as nn
from sae import SAE


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # Load the saved dataset
    dataset_path = "sixth_hidden_state_dataset.pt"
    dataset = torch.load(dataset_path)

    # Create a DataLoader
    # batch_size = 74112 // 16
    batch_size = 1024
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    d_model = 64
    d_encoder = 1024
    num_active_features = 4
    num_active_features_aux = 32

    model = SAE(d_model, d_encoder, num_active_features).to(device)

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    i_step = 0
    for epoch in range(100):
        active_indices_set = set()  # Create a set to store active indices for the epoch
        loss_sum = 0

        for i, batch in enumerate(dataloader):
            hidden_layers, raw_tokens = batch
            samples = hidden_layers.view(-1, 64)
            samples = samples.to(torch.float32).to(device)

            decoded, encoded_indices = model(samples, num_active_features)

            # Update the active indices set for the epoch
            active_indices_set.update(encoded_indices.view(-1).cpu().numpy())

            aux_decoded, aux_encoded_indices = model(samples, num_active_features_aux)

            loss = criterion(decoded, samples) + criterion(aux_decoded, samples) / 32
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i_step += 1

        # Print the total number of active indices for the epoch
        num_active_indices = len(active_indices_set)
        # print(f"Epoch {epoch}: Total number of active indices = {num_active_indices}")
        print(f"Epoch {epoch}: Total number of active indices = {num_active_indices}, Loss = {loss_sum / i}")

    checkpoint_path = f"sae_{d_encoder}_{num_active_features}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint at epoch {epoch}: {checkpoint_path}")
