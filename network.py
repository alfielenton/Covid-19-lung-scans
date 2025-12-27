from torch import nn

class Network(nn.Module):

    def __init__(self,in_channels = 1):
        super(Network,self).__init__()

        self.conv_head = nn.Sequential(nn.Conv2d(in_channels , 32 , kernel_size=8 , stride=4),
                                       nn.Dropout(),
                                       nn.ReLU(),
                                       nn.Conv2d(32 , 64 , kernel_size = 7 , stride = 4),
                                       nn.Dropout(),
                                       nn.ReLU())
        
        self.conv3 = nn.Conv2d(64 , 64 , kernel_size = 4 , stride = 3)

        self.fc_layers = nn.Sequential(nn.Dropout(),
                                       nn.Linear(64 * 10 * 10,512),
                                       nn.ReLU(), 
                                       nn.Dropout(),
                                       nn.Linear(512, 1))

    def forward(self,x):

        x = self.conv_head(x)
        x = nn.ReLU()(self.conv3(x))

        x = x.view(-1 , 64 * 10 * 10)

        x = self.fc_layers(x)
        return x.squeeze()