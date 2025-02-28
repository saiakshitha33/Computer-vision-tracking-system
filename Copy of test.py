import torch
import torchvision.models as models

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Save the model
torch.save(model.state_dict(), 'resnet18_model.pt')