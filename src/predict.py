import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# -----------------------------------
# 1. Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_mining.pth"


# -----------------------------------
# 2. Device
# -----------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------
# 3. Image preprocessing
# -----------------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# 4. Load ResNet18
# -----------------------------------

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# -----------------------------------
# 5. Class names
# -----------------------------------

class_names = [
    "Non-Mining",
    "Mining"
]


# -----------------------------------
# 6. Prediction function
# -----------------------------------

def predict_image(image):

    image = image.convert("RGB")

    image = image_transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_label = class_names[
        predicted_class.item()
    ]

    confidence_percentage = confidence.item() * 100

    return predicted_label, confidence_percentage


    
if __name__ == "__main__":

    print("ResNet18 model loaded successfully!")
    print("Device:", device)

    # Find a mining image from the dataset
    dataset_folder = PROJECT_ROOT / "Imagens_Dataset_Garimpo" / "Imagens - Rios_Floresta"

    image_files = list(dataset_folder.glob("*"))

    # Select the first image file
    test_image_path = None

    for file in image_files:
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            test_image_path = file
            break

    if test_image_path is None:
        print("No image found!")
    else:

        print("\nTest image:")
        print(test_image_path)

        # Open image
        test_image = Image.open(test_image_path)

        # Predict
        prediction, confidence = predict_image(test_image)

        print("\n========== PREDICTION ==========")
        print("Prediction :", prediction)
        print(f"Confidence : {confidence:.2f}%")
        