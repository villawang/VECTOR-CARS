import torch
import torch.nn.functional as F
from torch import nn
import pdb


def cal_conv_feature(L_in, kernel_size, stride, padding):
    return (L_in - kernel_size + 2*padding) // stride + 1

def cal_dconv_feature(L_in, kernel_size, stride, padding, dilation, output_padding):
    return (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

# construct autoencoder
class encoder(nn.Module):
    def __init__(self, data_len, inplanes, planes, kernel_size, stride=1, padding=0):
        super(encoder, self).__init__()
        self.feature_len = cal_conv_feature(data_len, kernel_size, stride, padding)
        self.conv1 = nn.Conv1d(inplanes, planes, 
                                kernel_size=kernel_size, 
                                stride=stride, bias=False, padding=padding)
        self.bn1 = nn.BatchNorm1d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# construct autoencoder
class decoder(nn.Module):
    def __init__(self, data_len, data_len_match, inplanes, planes, kernel_size, stride=1, padding=0,
                dilation=1, output_padding=0):
        super(decoder, self).__init__()
        self.feature_len = cal_dconv_feature(data_len, kernel_size, stride, padding, dilation, output_padding)
        if data_len_match > self.feature_len:
            output_padding = data_len_match - self.feature_len
        else:
            output_padding = 0
        self.feature_len = cal_dconv_feature(data_len, kernel_size, stride, padding, dilation, output_padding)

        self.conv1 = nn.ConvTranspose1d(inplanes, planes, 
                                        kernel_size=kernel_size, 
                                        stride=stride, bias=False, 
                                        padding=padding,
                                        dilation=dilation,
                                        output_padding=output_padding)
        self.bn1 = nn.BatchNorm1d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class CAE_4(nn.Module):
    def __init__(self, data_len, kernel_size, is_skip=True):
        super(CAE_4, self).__init__()
        kernel_size = kernel_size
        print('Kernel size: {}'.format(kernel_size))
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, 
                                stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len 
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, 
                                stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, 
                                stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len

        print('Latent space dimension {}'.format(self.en4_num_feature))

        self.decoder1 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.de1_num_feature = self.decoder1.feature_len 
        self.decoder2 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature, 
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.de2_num_feature = self.decoder2.feature_len     
        self.decoder3 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)
        self.de3_num_feature = self.decoder3.feature_len  

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.de3_num_feature, kernel_size, stride, padding, dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.de3_num_feature, kernel_size, stride, padding, dilation, output_padding)
        self.decoder4 = nn.ConvTranspose1d(64, 1, 
                                            kernel_size=kernel_size, stride=stride, 
                                            bias=False,
                                            padding=padding,
                                            dilation = dilation,
                                            output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip

    def forward(self, x):
        x = x.unsqueeze(1)
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        encoding = self.encoder4(en3)
        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en3)
            de3 = self.decoder3(de2+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
        de4 = self.decoder4(de3)
        out = self.sigmoid(de4)
        return out.squeeze(1)


class CAE_5(nn.Module):
    def __init__(self, data_len, kernel_size, is_skip=True):
        super(CAE_5, self).__init__()
        kernel_size = kernel_size
        print('Kernel size: {}'.format(kernel_size))
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len
        self.encoder5 = encoder(data_len=self.en4_num_feature, inplanes=512, planes=1024, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en5_num_feature = self.encoder5.feature_len

        print('Latent space dimension {}'.format(self.en5_num_feature))

        self.decoder1 = decoder(data_len=self.en5_num_feature, data_len_match=self.en4_num_feature, 
                                inplanes=1024, planes=512, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder2 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder3 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature, 
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder4 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        self.decoder5 = nn.ConvTranspose1d(64, 1, 
                                            kernel_size=kernel_size, stride=stride, 
                                            bias=False,
                                            padding=padding,
                                            dilation = dilation,
                                            output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip


    def forward(self, x):
        x = x.unsqueeze(1)
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        encoding = self.encoder5(en4)
        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en4)
            de3 = self.decoder3(de2+en3)
            de4 = self.decoder4(de3+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
            de4 = self.decoder4(de3)
        de5 = self.decoder5(de4)        
        out = self.sigmoid(de5)
        return out.squeeze(1)



class CAE_6(nn.Module):
    def __init__(self, data_len, kernel_size, is_skip=True):
        super(CAE_6, self).__init__()
        kernel_size = kernel_size
        print('Kernel size: {}'.format(kernel_size))
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len
        self.encoder5 = encoder(data_len=self.en4_num_feature, inplanes=512, planes=1024, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en5_num_feature = self.encoder5.feature_len
        self.encoder6 = encoder(data_len=self.en5_num_feature, inplanes=1024, planes=2048, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en6_num_feature = self.encoder6.feature_len

        print('Latent space dimension {}'.format(self.en6_num_feature))

        self.decoder1 = decoder(data_len=self.en6_num_feature, data_len_match=self.en5_num_feature, 
                                inplanes=2048, planes=1024, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder2 = decoder(data_len=self.en5_num_feature, data_len_match=self.en4_num_feature, 
                                inplanes=1024, planes=512, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder3 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder4 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature, 
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder5 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
            pdb.set_trace()
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        self.decoder6 = nn.ConvTranspose1d(64, 1, 
                                            kernel_size=kernel_size, stride=stride, 
                                            bias=False,
                                            padding=padding,
                                            dilation = dilation,
                                            output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip

    def forward(self, x):
        x = x.unsqueeze(1)
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)
        encoding = self.encoder6(en5)

        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en5)
            de3 = self.decoder3(de2+en4)
            de4 = self.decoder4(de3+en3)
            de5 = self.decoder5(de4+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
            de4 = self.decoder4(de3)
            de5 = self.decoder5(de4)
        de6 = self.decoder6(de5)
        out = self.sigmoid(de6)
        return out.squeeze(1)





class CAE_7(nn.Module):
    def __init__(self, data_len, kernel_size, is_skip=True):
        super(CAE_7, self).__init__()
        kernel_size = kernel_size
        print('Kernel size: {}'.format(kernel_size))
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len
        self.encoder5 = encoder(data_len=self.en4_num_feature, inplanes=512, planes=1024, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en5_num_feature = self.encoder5.feature_len
        self.encoder6 = encoder(data_len=self.en5_num_feature, inplanes=1024, planes=2048, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en6_num_feature = self.encoder6.feature_len
        self.encoder7 = encoder(data_len=self.en6_num_feature, inplanes=2048, planes=2048, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en7_num_feature = self.encoder7.feature_len

        print('Latent space dimension {}'.format(self.en7_num_feature))

        self.decoder1 = decoder(data_len=self.en7_num_feature, data_len_match=self.en6_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder2 = decoder(data_len=self.en6_num_feature, data_len_match=self.en5_num_feature, 
                                inplanes=2048, planes=1024, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder3 = decoder(data_len=self.en5_num_feature, data_len_match=self.en4_num_feature, 
                                inplanes=1024, planes=512, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder4 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder5 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature,
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder6 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
            pdb.set_trace()
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        self.decoder7 = nn.ConvTranspose1d(64, 1, 
                                            kernel_size=kernel_size, stride=stride, 
                                            bias=False,
                                            padding=padding,
                                            dilation = dilation,
                                            output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip

    def forward(self, x):
        x = x.unsqueeze(1)
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)
        en6 = self.encoder6(en5)
        encoding = self.encoder7(en6)

        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en6)
            de3 = self.decoder3(de2+en5)
            de4 = self.decoder4(de3+en4)
            de5 = self.decoder5(de4+en3)
            de6 = self.decoder6(de5+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
            de4 = self.decoder4(de3)
            de5 = self.decoder5(de4)
            de6 = self.decoder6(de5)
        de7 = self.decoder7(de6)
        out = self.sigmoid(de7)
        return out.squeeze(1)




class CAE_8(nn.Module):
    def __init__(self, data_len, kernel_size, is_skip=True):
        super(CAE_8, self).__init__()
        kernel_size = kernel_size
        print('Kernel size: {}'.format(kernel_size))
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len
        self.encoder5 = encoder(data_len=self.en4_num_feature, inplanes=512, planes=1024, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en5_num_feature = self.encoder5.feature_len
        self.encoder6 = encoder(data_len=self.en5_num_feature, inplanes=1024, planes=2048, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en6_num_feature = self.encoder6.feature_len
        self.encoder7 = encoder(data_len=self.en6_num_feature, inplanes=2048, planes=2048, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en7_num_feature = self.encoder7.feature_len
        self.encoder8 = encoder(data_len=self.en7_num_feature, inplanes=2048, planes=2048,
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en8_num_feature = self.encoder8.feature_len

        print('Latent space dimension {}'.format(self.en8_num_feature))

        self.decoder1 = decoder(data_len=self.en8_num_feature, data_len_match=self.en7_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder2 = decoder(data_len=self.en7_num_feature, data_len_match=self.en6_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder3 = decoder(data_len=self.en6_num_feature, data_len_match=self.en5_num_feature, 
                                inplanes=2048, planes=1024, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder4 = decoder(data_len=self.en5_num_feature, data_len_match=self.en4_num_feature, 
                                inplanes=1024, planes=512, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder5 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder6 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature,
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder7 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
            pdb.set_trace()
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        self.decoder8 = nn.ConvTranspose1d(64, 1, 
                                            kernel_size=kernel_size, stride=stride, 
                                            bias=False,
                                            padding=padding,
                                            dilation = dilation,
                                            output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip

    def forward(self, x):
        x = x.unsqueeze(1)
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)
        en6 = self.encoder6(en5)
        en7 = self.encoder7(en6)
        encoding = self.encoder8(en7)

        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en7)
            de3 = self.decoder3(de2+en6)
            de4 = self.decoder4(de3+en5)
            de5 = self.decoder5(de4+en4)
            de6 = self.decoder6(de5+en3)
            de7 = self.decoder7(de6+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
            de4 = self.decoder4(de3)
            de5 = self.decoder5(de4)
            de6 = self.decoder6(de5)
            de7 = self.decoder7(de6)
        de8 = self.decoder8(de7)
        out = self.sigmoid(de8)
        return out.squeeze(1)



class CAE_9(nn.Module):
    def __init__(self, data_len, kernel_size, is_skip=True):
        super(CAE_9, self).__init__()
        kernel_size = kernel_size
        print('Kernel size: {}'.format(kernel_size))
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len
        self.encoder5 = encoder(data_len=self.en4_num_feature, inplanes=512, planes=1024, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en5_num_feature = self.encoder5.feature_len
        self.encoder6 = encoder(data_len=self.en5_num_feature, inplanes=1024, planes=2048, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en6_num_feature = self.encoder6.feature_len
        self.encoder7 = encoder(data_len=self.en6_num_feature, inplanes=2048, planes=2048, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en7_num_feature = self.encoder7.feature_len
        self.encoder8 = encoder(data_len=self.en7_num_feature, inplanes=2048, planes=2048,
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en8_num_feature = self.encoder8.feature_len
        self.encoder9 = encoder(data_len=self.en8_num_feature, inplanes=2048, planes=2048,
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en9_num_feature = self.encoder9.feature_len

        print('Latent space dimension {}'.format(self.en9_num_feature))

        self.decoder1 = decoder(data_len=self.en9_num_feature, data_len_match=self.en8_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder2 = decoder(data_len=self.en8_num_feature, data_len_match=self.en7_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder3 = decoder(data_len=self.en7_num_feature, data_len_match=self.en6_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder4 = decoder(data_len=self.en6_num_feature, data_len_match=self.en5_num_feature, 
                                inplanes=2048, planes=1024, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder5 = decoder(data_len=self.en5_num_feature, data_len_match=self.en4_num_feature, 
                                inplanes=1024, planes=512, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder6 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder7 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature,
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder8 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
            pdb.set_trace()
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding, dilation, output_padding)
        self.decoder9 = nn.ConvTranspose1d(64, 1, 
                                            kernel_size=kernel_size, stride=stride, 
                                            bias=False,
                                            padding=padding,
                                            dilation = dilation,
                                            output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip

    def forward(self, x):
        x = x.unsqueeze(1)
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)
        en6 = self.encoder6(en5)
        en7 = self.encoder7(en6)
        en8 = self.encoder8(en7)
        encoding = self.encoder9(en8)

        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en8)
            de3 = self.decoder3(de2+en7)
            de4 = self.decoder4(de3+en6)
            de5 = self.decoder5(de4+en5)
            de6 = self.decoder6(de5+en4)
            de7 = self.decoder7(de6+en3)
            de8 = self.decoder8(de7+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
            de4 = self.decoder4(de3)
            de5 = self.decoder5(de4)
            de6 = self.decoder6(de5)
            de7 = self.decoder7(de6)
            de8 = self.decoder8(de7)
        de9 = self.decoder9(de8)
        out = self.sigmoid(de9)
        return out.squeeze(1)






if __name__ == '__main__':
    model = CAE_7(data_len=1000, kernel_size=8)

    pdb.set_trace()
