import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import List

class GradCam:
    def __init__(self, model, target_layer: List[str]):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = self.model
        for part in self.target_layer:
          target_layer = target_layer._modules[part]
        #target_layer = self.model._modules[self.target_layer]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        self.handles = [forward_handle, backward_handle]

    def forward(self, input_img):
        return self.model(input_img)

    def backward(self, output):
        one_hot = torch.zeros_like(output, dtype=torch.float)
        one_hot[0][output[0].argmax()] = 1.0

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

    def generate_gradcam(self, input_img):
        assert input_img.shape[0] == 1, "You should pass one image at once!"
        output = self.forward(input_img)
        self.backward(output)

        weights = F.adaptive_avg_pool2d(self.gradients, 1).squeeze()
        activations = self.activations[0]

        gradcam = torch.mul(activations, weights).sum(dim=0).detach().cpu()
        gradcam = torch.relu(gradcam)

        gradcam = transforms.Resize((input_img.shape[2], input_img.shape[3]))(gradcam.unsqueeze(0))

        return gradcam