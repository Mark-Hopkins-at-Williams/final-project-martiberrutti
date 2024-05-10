import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


device = "cpu"
torch.set_default_device(device)


def load_image(image_name):
    image = Image.open(image_name)
    image_size = 256 
    loader = transforms.Compose([transforms.Resize(image_size),  # scales imported image
                                 transforms.ToTensor()])  # transforms it into a torch tensor
    image = loader(image)
    return image.to(device, torch.float)


def show_image(tensor, title=None):
    unloader = transforms.ToPILImage()  # converts into PIL image
    image = tensor.cpu().clone().clamp_(0, 1)  # clones the tensor so as not to perform changes on it
    image = unloader(image)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pauses a bit so that plots are updated
