import model_setup as ms
import losses
import config_and_data as c
import torch
from tqdm import tqdm

def train_epoch():
    # Create the model by calling create_cinn_model from model_setup
    model = ms.create_cinn_model()

    # Move the model to the appropriate device
    model = model.to(c.device)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_init, betas=c.adam_betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Store the optimizer and scheduler in the model for later reference
    model.optimizer = optimizer
    model.scheduler = scheduler

    # Progress bar for epochs
    for epoch in range(c.n_epochs):
        model.train()
        train_loss = 0.0

        # Progress bar for training batches
        for x_batch, y_batch in tqdm(c.train_loader, desc=f'Epoch {epoch+1}/{c.n_epochs}'):
            x_batch = x_batch.to(c.device)
            y_batch = y_batch.to(c.device)

            model.optimizer.zero_grad()

            # Forward pass: Transform x given y
            z, jacobian = model(x_batch, y_batch)

            # Compute loss using the MLE loss function
            loss = losses.loss_max_likelihood(z, jacobian)
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clamp)

            # Optimization step
            model.optimizer.step()

            train_loss += loss.item()

        # Average training loss
        avg_train_loss = train_loss / len(c.train_loader)

        # Evaluation every 'eval_test' epochs
        if (epoch + 1) % c.eval_test == 0:
            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for x_batch, y_batch in c.test_loader:
                    x_batch = x_batch.to(c.device)
                    y_batch = y_batch.to(c.device)

                    z, jacobian = model(x_batch, y_batch)
                    loss = losses.loss_max_likelihood(z, jacobian)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(c.test_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Test Loss = {avg_test_loss:.5f}")

        # Step the learning rate scheduler
        model.scheduler.step()

    # Save the trained model after all epochs
    torch.save(model.state_dict(), 'cinn_model.pth')
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_epoch()