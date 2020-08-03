import cv2
import numpy as np

from torch import nn
import torch

try:
    from utils import SaveOutput
except:
    from src.utils import SaveOutput

class GradCam():

    def __init__(self, model, module, device):
        self.device = device
        self.module = module.to(device)
        self.model = model.to(device)

    def __call__(self, *args, **kwargs):
        return self.compute_gradcam(*args, **kwargs)

    def get_hook(self):
        forward = SaveOutput()
        back = SaveOutput()
        handle1 = self.module.register_forward_hook(forward)
        handle2 = self.module.register_backward_hook(back)
        return forward, back

    def get_values(self, inputs, target):
        inputs = inputs.to(self.device)
        forward, back = self.get_hook()
        x = inputs.clone()
        if len(x.shape) == 3:
            x = x.reshape(1, *list(inputs.shape))

        #print(x.shape)
        out = self.model(x)
        out.view(-1)[target].backward()
        activation, gradient = forward.outputs[0], back.outputs[0]
        return activation, gradient[0]

    def compute_gradcam(self, inputs, target, out_type = 'id', interpolation_mode=cv2.INTER_LANCZOS4):
        activation, gradient = self.get_values(inputs, target)
        alpha = torch.sum(gradient, dim=(0, 2, 3))
        C = alpha.view(-1, 1, 1) * activation[0]
        out = torch.sum(C, dim=(0))

        # print('Input:\n', inputs.shape)
        # print('Gradient:\n', gradient.shape)
        # print('Alpha:\n', alpha.shape)
        # print('Activation:\n', activation.shape)
        # print('alpha*A:\n', C.shape)
        # print('Out:\n', out.shape)

        out = self.output_processing(out, out_type)
        out = self.resize(inputs, out, interpolation_mode=interpolation_mode)
        return out.view(*list(inputs.shape[-2:]))

    def resize(self, input, output, interpolation_mode):
        dim = tuple(input.shape[-2:])
        resized_img = cv2.resize(output.numpy(), dim, interpolation = interpolation_mode)
        return torch.tensor(resized_img)

    def output_processing(self, out, out_type):
        if out_type == 'id':
            return out.detach().cpu()
        elif out_type == 'relu':
            return nn.ReLU()(out).detach().cpu()
        elif out_type == 'abs':
            return torch.abs(out).detach().cpu()
        else:
            raise NotImplementedError


def smoothing(method, X, iter, var, **kwargs):
    results = torch.Tensor([])
    for i in range(iter):
        x = X + torch.randn(*list(X.shape))*var
        output = method(**kwargs)
        results = torch.cat((results, torch.tensor(np.mean(cycle, axis=0)).view(1, 28, 28)))
