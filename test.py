import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model, amount=0.3):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# Load the trained model
model = YOLO('best-6.pt')  # Replace with your trained model path

results = model.val(data="/content/drive/MyDrive/yolov8_tracking/yolov8_detection/data.yaml")
print(f"mAP50-95: {results.box.map}")  # Print mean Average Precision

# Access the PyTorch model
torch_model = model.model

print(torch_model)

# Apply pruning
print("Pruning model...")
pruned_torch_model = prune_model(torch_model, amount=0.1)  # Adjust amount as needed
print("Model pruned.")

# Update the Ultralytics YOLO model with the pruned PyTorch model
model.model = pruned_torch_model

print("Saving pruned model...")
# Save the pruned model
model.save('best-6_pruned.pt')

print("Pruned model saved.")

# Optional: Fine-tune the pruned model
# Uncomment the following lines if you want to fine-tune
# results = model.train(data='your_dataset.yaml', epochs=50)
# model.save('yolov8s_trained_pruned_finetuned.pt')

# Evaluate the pruned model (and optionally the fine-tuned model)

model = YOLO('yolov8s_trained_pruned.pt') 
results = model.val(data="/content/drive/MyDrive/yolov8_tracking/yolov8_detection/data.yaml")
print(f"mAP50-95: {results.box.map}")  # Print mean Average Precision

# If you fine-tuned, you can evaluate the fine-tuned model similarly
# fine_tuned_model = YOLO('yolov8s_trained_pruned_finetuned.pt')
# results = fine_tuned_model.val()
# print(f"Fine-tuned mAP50-95: {results.box.map}")