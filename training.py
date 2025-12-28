import json
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader , WeightedRandomSampler
import torchvision.transforms.v2 as transforms_v2
from network import Network
import data_handling

dc = data_handling.DataConverter()
split_train = 0.7
split_val = 0.001
X_train, Y_train, X_test, Y_test = dc.train_test_splitting(split_train)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device being used: {device}')
model = Network()
model.to(device)
model.train()

lr = 0.001
optimiser = optim.Adam(model.parameters(),lr,weight_decay=0.0001)

criterion = nn.BCEWithLogitsLoss()

num_epochs = 50
BATCH_SIZE = 32


train_ds = TensorDataset(X_train, Y_train)

class_counts = torch.tensor([(Y_train == t).sum().item() for t in [0,1]])
class_weights = 1. / class_counts
sample_weights = class_weights[Y_train]
sampler = WeightedRandomSampler(sample_weights,num_samples=len(sample_weights),replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

transform_with_crop = transforms_v2.Compose([transforms_v2.CenterCrop(350),
                                            transforms_v2.Resize(512)])

transform_without_crop = transforms_v2.Compose([transforms_v2.Resize(512)])

general_transforms = transform_with_crop

train_transform = transforms_v2.Compose([transforms_v2.RandomHorizontalFlip(),
                                         transforms_v2.RandomVerticalFlip(),
                                         transforms_v2.RandomRotation((-15,15))])

print("Starting training...")
print(f'Train for {num_epochs} epochs')
epoch_losses = []
forward_pass_losses = []

for epoch in range(num_epochs):
    epoch_training_loss = 0.0

    for x_batch, y_batch in train_dl:

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        augmented_batch = train_transform(general_transforms(x_batch))

        logits = model(augmented_batch)
        loss = criterion(logits, y_batch.float())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_training_loss += loss.item() / x_batch.size(0)
        forward_pass_losses.append(loss.item() / x_batch.size(0))

    epoch_losses.append(epoch_training_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Total Training Loss per Sample: {epoch_training_loss:.4f}")
print()

with open('model-results//training_results.json','w') as f:
    data = {'epoch loss': epoch_losses, 
            'forward pass loss': forward_pass_losses}
    json.dump(data, f)
print('Training results saved')

torch.save({'model_state_dict':model.state_dict()},'model-results//model_checkpoint.pth')
print('Model Saved!')

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(epoch_losses)
axs[0].set_xlabel('# Epoch')
axs[0].set_ylabel('Training loss')
axs[0].set_title('Training loss of model over epochs')

axs[1].plot(forward_pass_losses)
axs[1].set_xlabel('# Forward pass')
axs[1].set_ylabel('Forward pass training loss')
axs[1].set_title('Training loss of model over forward pass')

plt.show()

print('\nEvaluating Model')
model.to('cpu')
model.eval()

print('Testing model with full testing data')
with torch.no_grad():
    x_test , y_test = X_test.clone() , Y_test.clone()

    print(f'X test shape: {x_test.size()}')
    print(f'Y test shape: {y_test.size()}')

    negatives = (y_test==0).sum()
    positives = (y_test==1).sum()

    print(f'Non COVIDS: {negatives}')
    print(f'COVIDS: {positives}\n')

    augmented_test_batch = general_transforms(x_test)
    p_logits = model(augmented_test_batch)

    probs = torch.sigmoid(p_logits)
    preds = (probs > 0.5).float()

    correct = (preds == y_test).sum().item()
    total = x_test.size(0)

tp = preds[y_test==1].sum().item()
fp = preds[y_test==0].sum().item()

tn = (1 - preds[y_test==0]).sum().item()
fn = (1 - preds[y_test==1]).sum().item()

print('True positives', tp)
print('False positives', fp)
print('True negatives', tn)
print('False negatives', fn)

acc = correct / total * 100
print(f"Test Accuracy: {acc:.2f}%\n")

print('Testing model with reduced testing data')
with torch.no_grad():

    x_test = torch.cat([x_test[y_test==0], x_test[y_test==1][:(y_test==0).sum()]])
    y_test = torch.cat([y_test[y_test==0], y_test[y_test==1][:(y_test==0).sum()]])

    print(f'X test shape: {x_test.size()}')
    print(f'Y test shape: {y_test.size()}')

    print(f'Non COVIDS: {y_test[y_test==0].size(0)}')
    print(f'COVIDS: {y_test[y_test==1].size(0)}')

    augmented_test_batch = general_transforms(x_test)
    p_logits = model(augmented_test_batch)

    probs = torch.sigmoid(p_logits)
    preds = (probs > 0.5).float()

    correct = (preds == y_test).sum().item()
    total = x_test.size(0)

tp = preds[y_test==1].sum().item()
fp = preds[y_test==0].sum().item()

tn = (1 - preds[y_test==0]).sum().item()
fn = (1 - preds[y_test==1]).sum().item()

print('True positives', tp)
print('False positives', fp)
print('True negatives', tn)
print('False negatives', fn)

acc = correct / total * 100
print(f"Test Accuracy: {acc:.2f}%")