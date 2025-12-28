import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms_v2
import random
import matplotlib.pyplot as plt
import data_handling
from network import Network

dc = data_handling.DataConverter(verbose=False)
transform_with_crop = transforms_v2.Compose([transforms_v2.CenterCrop(350),
                                            transforms_v2.Resize(512)])
transform_without_crop = transforms_v2.Compose([transforms_v2.Resize(512)])
general_transforms = transform_with_crop

if general_transforms is transform_with_crop:
    print('General transformation includes cropping')
else:
    print('General transformation does not include cropping')

model = Network()
checkpoint = torch.load('model-results//model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations['value'] = output.detach()

def backward_hook(module, grad_in, grad_out):
    gradients['value'] = grad_out[0].detach()

target_layer = model.conv3
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

total_num_images = 5
total_num_covid = total_num_images // 2
total_num_non_covid = total_num_images - total_num_covid

num_covid = 0
num_non_covid = 0

fig , axs = plt.subplots(total_num_images,2,figsize=(20,20))
fig.suptitle('Images and CAMs')

i = 0
while i < total_num_images:

    random_index = random.randint(0,dc.N)
    selected_image = dc.norm_inputs[random_index].unsqueeze(0)
    true_class = dc.labels[random_index]

    if num_covid < total_num_covid and true_class == 1:
        num_covid += 1
    elif num_non_covid < total_num_non_covid and true_class == 0:
        num_non_covid += 1
    else:
        continue

    transformed_image = general_transforms(selected_image)

    logit = model(transformed_image)
    prob = torch.sigmoid(logit)

    model.zero_grad()
    logit.backward()

    weights = gradients['value'].mean(dim=[2, 3], keepdim=True)
    cam = (weights * activations['value']).sum(dim=1, keepdim=True)

    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / cam.max()

    cam = F.interpolate(cam, size=(selected_image.size(2), selected_image.size(3)), mode='bilinear', align_corners=False)
    cam = cam.squeeze()

    img = transformed_image.squeeze()

    pred_class_string = 'COVID' if prob > .5 else 'Non-COVID'
    true_class_string = 'COVID' if bool(true_class) else 'Non-COVID'

    axs[i,0].imshow(img, cmap='gray')
    axs[i,0].imshow(cam, cmap='jet', alpha=0.5)

    axs[i,1].imshow(dc.unnorm_inputs[random_index].squeeze(), cmap='gray')
    axs[i,0].set_title('Predicted class: ' + pred_class_string + '.True class: ' + true_class_string , fontsize=5)

    prob = 1 - prob if prob < .5 else prob
    axs[i,1].set_title(f'Probability: {prob:.4f}',fontsize=5)

    axs[i,0].axis('off')
    axs[i,1].axis('off')

    i += 1

plt.tight_layout()
plt.show()