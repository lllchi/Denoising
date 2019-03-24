#Residual Dense Network
import torch 
import torch.nn as nn
import torch.nn.init as init

class RDB_Conv(nn.Module):
    def __init__(self, in_C, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_C, growRate, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(in_C),
        #     Modulecell(in_channels=in_C,out_channels=growRate,kernel_size=kSize))
        # self.conv = nn.Sequential(*[
        #     nn.Conv2d(in_C, growRate, kSize, padding=(kSize-1)//2, stride=1),
        #     nn.BatchNorm2d(growRate),
        #     nn.PReLU()
        # ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)
        
class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        
        G0 = growRate0
        G = growRate
        C = nConvLayers
        
        convs = []
        for i in range(C):
            convs.append(RDB_Conv(G0 + i*G, G, kSize))
        self.convs = nn.Sequential(*convs)
        
        #Local Feature Fusion
        self.lff = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        
    def forward(self, x):
        y = self.lff(self.convs(x)) + x
        return y
        
class RDN(nn.Module):
    def __init__(self, growRate0, RDBkSize):
        super(RDN, self).__init__()
        
        G0 = growRate0
        kSize = RDBkSize
        
        #D, C, G = (20, 6, 32)
        D, C, G = (16, 8, 64)
        
        self.RDB_num = D
        
        #Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(3, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.sfe2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        
        #Residual Dense Blocks
        self.RDBs = nn.ModuleList()
        for i in range(D):
            self.RDBs.append(RDB(G0, G, C, kSize))
            
        #Global Feature Fusion
        self.gff = nn.Sequential(*[
            nn.Conv2d(D*G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        #self.non_local = Self_Att(G0, 'relu')

        self.out_conv = nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    def forward(self, x):
        f1 = self.sfe1(x)
        y = self.sfe2(f1)
        
        RDB_out = []
        for i in range(self.RDB_num):
            y = self.RDBs[i](y)
            RDB_out.append(y)
            
        y = self.gff(torch.cat(RDB_out, 1))

        #y = self.non_local(y)
        
        y = self.out_conv(f1 + y)
        y += x
        
        return y
            
        
                    
        

