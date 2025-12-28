import torch
from network import Network
import data_handling

model = Network()
checkpoint = torch.load('model-results//model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dc = data_handling.DataConverter(False)
