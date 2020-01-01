import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models 
import numpy as np
import torch.optim as optim 
from tqdm import tqdm
device = torch.device("cpu")
def load_image_to_tensor(image,imgsize=128):
	loader = transforms.Compose([transforms.Resize(imgsize),
				    transforms.ToTensor()])
	image = Image.open(image)
	tensor = loader(image)
	return tensor.unsqueeze(0).to(device,torch.float)

content_image_file = "../data/dancing.jpg"
style_image_file = "../data/picasso.jpg"
c_image_tensor = load_image_to_tensor(content_image_file)
s_image_tensor = load_image_to_tensor(style_image_file)
def imshow(img_tensor):
    im = np.array(img_tensor.squeeze(0).permute(1,2,0).cpu().numpy() *255,dtype=np.uint8)
    im = Image.fromarray(im)
    im.show()
def load_feature_extraction_model(model_name= 'vgg19'):
    model = models.vgg19(pretrained=True).features.to(device).eval()
    return model
cnn_norm_mean = torch.tensor([0.458,0.456,0.406],dtype=torch.float).to(device)
cnn_norm_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float).to(device)
class Norm(nn.Module):
    def __init__(self,mean,std):
        super(Norm,self).__init()
        self.mean = mean
        self.std = std
    def forward(self,input):
        return (input - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self,content_weight):
        super(ContentLoss,self).__init__()
        self.cw = content_weight
    def forward(self,input,target):
        target = target.detach()
        return self.cw * F.mse_loss(input,target)

class StyleLoss(nn.Module):
    def __init__(self,style_weight=None):
        super(StyleLoss,self).__init__()
        self.sw = style_weight
    def gram_matrix(self,input):
        in_s = input.shape
        row,col = in_s[0] * in_s[1],in_s[2] * in_s[3]
        input = input.view(row,col)
        gm = torch.mm(input,input.t())
        return gm
    def forward(self,input,target):
        input_gm = self.gram_matrix(input)
        target_gm = self.gram_matrix(target.cpu().detach())
        return F.mse_loss(input_gm,target_gm)

def image_generation(model,
                     content_loss,
                     style_loss,
                     content_image,
                     style_image,
                     steps = 300):
    """use the gradient descent to optim
    the image that make loss minimize"""
    generation_image = content_image.clone().detach().requires_grad_(True)
    optimizer = optim.LBFGS([generation_image])
    style_layer = ['conv1','conv2','conv3','conv4','conv5']
    content_layer = ['conv4']
    i = 0
    name = ""
    style_losses = []
    content_losses = []
    print(content_image.requires_grad)
    print(style_image.requires_grad)
    for step in tqdm(range(steps)):
        def closure(generation_image,content_image,style_image):
            for i,layer in enumerate(model.children()):
                if isinstance(layer,nn.Conv2d):
                    name = "conv%d"%(i+1)
                elif isinstance(layer,nn.ReLU):
                    name = "relu"
                elif isinstance(layer,nn.MaxPool2d):
                    name = "maxpool"
                if i == 0:
                    c = layer(content_image).detach()
                    s = layer(style_image).detach()
                    g = layer(generation_image).detach()
                else:
                    c = layer(c).detach()
                    s = layer(s).detach()
                    g = layer(g).detach()
                if name in style_layer:
                    style_losses.append(style_loss(generation_image,style_image))
                elif name in content_layer:
                    content_losses.append(content_loss(generation_image,content_image))
            loss = sum(style_losses) + sum(content_losses)
            print(loss) 
            loss.backward()
            return loss
        optimizer.step(closure(generation_image,content_image,style_image))
    return generation_image
model = models.vgg19(pretrained=True).features.eval()
style_loss = StyleLoss()
content_loss = ContentLoss(10000)
generation_image = image_generation(model,
                                    content_loss,
                                    style_loss,
                                    c_image_tensor,
                                    s_image_tensor)
imshow(generation_image)
