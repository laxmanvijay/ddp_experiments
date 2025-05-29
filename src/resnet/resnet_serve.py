import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import argparse
import os

class ResNetInference:
    def __init__(self, model_path, num_classes=1000, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        self.model = self.load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_labels = self.load_imagenet_labels()
    
    def load_model(self, model_path):
        """Load the trained ResNet model"""
        model = models.resnet50(num_classes=self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Training accuracy: {checkpoint.get('accuracy', 'N/A')}")
        print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
        
        return model
    
    def load_imagenet_labels(self):
        labels_path = "imagenet_classes.txt"
        
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
        else:
            labels = [f"class_{i}" for i in range(self.num_classes)]
            print(f"Warning: {labels_path} not found. Using generic labels.")
        
        return labels
    
    def predict(self, image, top_k=5):
        if image is None:
            return "Please upload an image"
        
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = {}
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            
            if class_idx < len(self.class_labels):
                class_name = self.class_labels[class_idx]
            else:
                class_name = f"class_{class_idx}"
            
            results[class_name] = prob
        
        return results

def create_imagenet_labels_file():
    """Create ImageNet labels file if it doesn't exist"""
    labels_path = "imagenet_classes.txt"
    
    if not os.path.exists(labels_path):
        sample_labels = [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
            "electric ray", "stingray", "cock", "hen", "ostrich",
        ]
        
        try:
            import requests
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            response = requests.get(url)
            if response.status_code == 200:
                with open(labels_path, 'w') as f:
                    f.write(response.text)
                print(f"Downloaded ImageNet labels to {labels_path}")
            else:
                with open(labels_path, 'w') as f:
                    for i in range(1000):
                        f.write(f"class_{i}\n")
                print(f"Created generic labels file: {labels_path}")
        except:
            with open(labels_path, 'w') as f:
                for i in range(1000):
                    f.write(f"class_{i}\n")
            print(f"Created generic labels file: {labels_path}")

def main():
    parser = argparse.ArgumentParser(description="Serve ResNet model with Gradio")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classes the model was trained on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to serve the interface on')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to serve the interface on')
    parser.add_argument('--share', action='store_true',
                        help='Create a public link')
    args = parser.parse_args()
    
    create_imagenet_labels_file()
    
    inference = ResNetInference(args.model_path, args.num_classes)
    
    def predict_and_format(image):
        if image is None:
            return "Please upload an image"
        
        results = inference.predict(image, top_k=5)
        
        output_text = "Top 5 Predictions:\n\n"
        for i, (class_name, prob) in enumerate(results.items(), 1):
            output_text += f"{i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)\n"
        
        return output_text, results
    
    interface = gr.Interface(
        fn=predict_and_format,
        inputs=[
            gr.Image(type="pil", label="Upload an image")
        ],
        outputs=[
            gr.Textbox(label="Predictions", lines=10),
            gr.JSON(label="Raw Predictions")
        ],
        title="ResNet Image Classification",
        description=f"""
        Upload an image to classify it using a trained ResNet-50 model.
        
        **Model Information:**
        - Model: ResNet-50
        - Classes: {args.num_classes}
        - Checkpoint: {args.model_path}
        
        The model will return the top 5 most likely classes with their confidence scores.
        """,
        examples=[
            # ["example1.jpg"],
            # ["example2.jpg"],
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    # Launch interface
    print(f"Launching Gradio interface...")
    print(f"Model path: {args.model_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {inference.device}")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )

if __name__ == "__main__":
    main()