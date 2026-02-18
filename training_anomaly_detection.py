import torch
import torch.nn as nn
import torch.optim as optim
from cnn_anomaly_detection import Autoencoder
from dataset_anomaly_detection import train_loader, test_loader
from tqdm import tqdm

def main():
    # Device (GPU se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Istanza del modello
    model = Autoencoder().to(device)

    # Loss function
    loss = nn.MSELoss()  # Mean Squared Error

    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20  # puoi aumentare a 30
    model.train()    # mette il modello in modalità training
    
    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)  # sposta i batch su GPU se disponibile

            # Forward pass
            outputs = model(batch)

            # Calcolo della loss
            loss_fn = loss(outputs, batch)

            # Backpropagation
            optimizer.zero_grad()
            loss_fn.backward()
            optimizer.step()

            epoch_loss += loss_fn.item() * batch.size(0)  # somma loss del batch

        epoch_loss /= len(train_loader.dataset)  # media su tutto il dataset
        
        model.eval()  # modalità evaluation
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test Epoch {epoch+1}/{num_epochs}"):
                batch = batch.to(device)
                outputs = model(batch)
                loss_fn = loss(outputs, batch)
                test_loss += loss_fn.item() * batch.size(0)
        test_loss /= len(test_loader.dataset)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_autoencoder.pth")
            print(f"Nuovo miglior modello salvato! Test Loss: {best_test_loss:.6f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    


if __name__ == '__main__':
    main()

