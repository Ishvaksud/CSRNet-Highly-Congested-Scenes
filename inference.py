from torch_snippets import *
import cv2 as cv
from PIL import Image

class CSRnet(nn.Module):
    def __init__(self):
        super(CSRnet, self).__init__()
        vgg_16=torchvision.models.vgg16(pretrained=True)
        
        self.frontend=nn.Sequential(
            *list(vgg_16.features.children())[:10],
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(), )


        
        self.backend=nn.Sequential(
            
                     
             nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding='same',dilation=2),
             nn.ReLU(),
             nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1),
        )
        
    def forward(self,x):

        x=self.frontend(x)
        x=self.backend(x)

        return x

# tfm=T.Compose([ 
#     T.ToTensor()
# ])

model=CSRnet()
checkpoint = torch.load('model.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
img=cv.imread('/Users/ishvaksud/Desktop/crowds_spare.jpeg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
#img=cv.resize(img,(640,640))
img=torch.tensor(img).permute(2,0,1).float()
output = model(img[None])
x=output.squeeze(0).permute(1,2,0).detach().numpy()
print("Predicted Count : ",int(output.detach().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
plt.imshow(temp)
plt.show()
# img = tfm(Image.open('/Users/ishvaksud/Desktop/crowd.jpg').convert('RGB')).to('cpu')
# output = model(img[None])
# print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
# cv2.imshow('detect',img)
# cv.waitKey(0)

    

    
    
    





    
