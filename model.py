import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.downsample_layer = None
        self.do_downsample = False
        if in_channels != out_channels or stride != 1:
            self.do_downsample = True
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels))
        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        identity = x
        out = self.net(x)
        if self.do_downsample:
            identity = self.downsample_layer(x)
        return F.relu(out + identity, inplace=True)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

def dense_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        block = []
        channel = in_channels
        for i in range(num_layers):
            block.append(dense_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)


class ConvSelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ConvSelfAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (batch_size, h*w, c/2)
        key = self.key_conv(x).view(batch_size, -1, height * width)  # (batch_size, c/2, h*w)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (batch_size, c, h*w)

        attention = torch.matmul(query, key)  # (batch_size, h*w, h*w)
        attention = F.softmax(attention, dim=2)  #

        attention_feature = torch.matmul(value, attention)  # (batch_size, h*w, c)
        attention_feature = attention_feature.view(batch_size, -1, height, width)  # (batch_size, c, h, w)
        out = self.gamma * attention_feature + x
        return out


class Extractor(nn.Module):
    def __init__(self, num_feature, hash_bits, type_bits):
        super(Extractor, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch = nn.BatchNorm2d(64)
        self.rule = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #branch
        self.resnet1 = ResBlock(in_channels=64, out_channels=128, stride=1)
        self.densenet1 = DenseBlock(in_channels=64, growth_rate=16, num_layers=4)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.resnet2 = ResBlock(in_channels=128, out_channels=256, stride=1)
        self.densenet2 = DenseBlock(in_channels=128, growth_rate=16, num_layers=8)

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.AvgPool2d(2, 2))
        self.attention = ConvSelfAttentionModule(384)

        self.encode1 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.encode2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        # Deconv2D
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decode4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decode5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.classifier = nn.Conv2d(16, 3, kernel_size=1)
        self.fc = nn.Linear(512 * 7 * 7, num_feature)
        self.hash_layer = nn.Linear(num_feature, hash_bits)
        self.class_layer = nn.Linear(num_feature, type_bits)
        self.Softmax = nn.Softmax()
        self.tanh = torch.nn.Tanh()
        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        x1 = self.rule(self.batch(self.conv(x)))
        x2 = self.maxpool(x1)

        res1 = self.resnet1(x2)
        den1 = self.densenet1(x2)

        x3 = self.avgpool(res1)
        x4 = self.avgpool(den1)

        res2 = self.resnet2(x3)
        den2 = self.densenet2(x4)

        low = self.conv1(torch.cat((res1, den1), dim=1))
        low = self.downsample(low)
        high = self.conv2(torch.cat((res2, den2), dim=1))
        fusion = torch.cat((low, high), dim=1)
        att = self.attention(fusion)

        en1 = self.encode1(att)
        en2 = self.encode2(en1)

        out1 = en2.view(en2.size(0), -1)
        feature = self.fc(out1)
        hash_bits = self.hash_layer(feature)
        hash_code = self.tanh(hash_bits)
        type_bits = self.class_layer(feature)
        type_bits = self.Softmax(type_bits)

        # ## Deconv2D
        up1 = self.decode1(en2)
        up2 = self.decode2(up1)
        up3 = self.decode3(up2)
        up4 = self.decode4(up3)
        up5 = self.decode5(up4)
        segs = self.classifier(up5)
        segs = self.Softmax(segs)

        return hash_code, hash_bits, type_bits, segs

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)





