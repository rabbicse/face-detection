import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models import resnet50


class GenderNet(nn.Module):
    def __init__(self):
        super(GenderNet, self).__init__()
        # Load pre-trained ResNet-50
        self.base_model = resnet50(pretrained=True)

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        # Forward pass through ResNet-50
        x = self.base_model(x)
        return x

class Tester:
    def __init__(self, model_path, img_size, means, stds, class_labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()  # Set the model to evaluation mode
        self.img_size = img_size
        self.means = means
        self.stds = stds
        self.class_labels = class_labels

        # Define preprocessing pipeline
        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means, std=self.stds),
        ])

    def preprocess_image(self, image_path):
        # Load image
        img = Image.open(image_path).convert("RGB")
        # Apply transformations
        img_tensor = self.transforms(img).unsqueeze(0)  # Add batch dimension
        return img_tensor

    def predict(self, image_path):
        # Preprocess the image
        img_tensor = self.preprocess_image(image_path).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(img_tensor)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy().flatten()

        # Get predicted class
        predicted_class = self.class_labels[probabilities.argmax()]
        return predicted_class, probabilities


# Usage Example
if __name__ == "__main__":
    # Parameters
    model_path = "../../models/model_gender.pth"
    image_path = "man_10005.jpg"
    img_size = 112
    means = [0, 0, 0]  # Example for ImageNet
    stds = [1, 1, 1]  # [0.229, 0.224, 0.225]  # Example for ImageNet
    class_labels = ["Men", "Women"]

    # Initialize tester
    tester = Tester(model_path, img_size, means, stds, class_labels)

    # Make prediction
    predicted_class, probabilities = tester.predict(image_path)

    print(f"Predicted Class: {predicted_class}")
    print(f"Probabilities: {probabilities}")
