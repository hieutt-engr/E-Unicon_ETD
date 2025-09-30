import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from networks.resnet import BasicBlock, Bottleneck
from attentions.SEAttention import SEAttention
from attentions.ECAAttention import ECAAttention
from attentions.CBAM import CBAMBlock
from representation.MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer

class AttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, att_dim=64):
        super(AttentionBlock, self).__init__()

        self.ch_dim = att_dim

        self.conv_for_DR = nn.Conv2d(inplanes, self.ch_dim, kernel_size=1, stride=1, bias=False)
        self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(int(self.ch_dim * (self.ch_dim + 1) / 2), planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        # NxCxHxW
        out = self.conv_for_DR(x)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = CovpoolLayer(out)
        out = SqrtmLayer(out, 5)
        out = TriuvecLayer(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)  # NxC
        out = out.view(out.size(0), out.size(1), 1, 1).contiguous()  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        out = residual * out

        return out


class TinyMPNCOVResNetBackbone(nn.Module):
    def __init__(self, block, layers, attention='Cov', input_size=None, out_dim=1280, in_ch=37):
        super().__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.layer_reduce = nn.Conv2d(128 * block.expansion, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(128)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        # Attention
        if attention == 'Cov':
            self.attention = AttentionBlock(128, 128, att_dim=64)
        elif attention == 'ECA':
            self.attention = ECAAttention(kernel_size=3)
        elif attention == 'SE':
            self.attention = SEAttention(channel=128, reduction=8)
        elif attention == 'CBAM':
            k = 4 if input_size == 32 else 8
            self.attention = CBAMBlock(channel=128, reduction=16, kernel_size=k - 1)
        else:
            self.attention = nn.Identity()

        self.out_dim = int(128 * (128 + 1) / 2)  # MPNCOV output dim

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)
        x = self.attention(x)

        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, dim=1)
        return x  # [B, out_dim]
    

class ProjectionHead(nn.Module):
  def __init__(self, in_dim, feat_dim=128, head_type='mlp'):
      super().__init__()
      if head_type == 'mlp':
          self.net = nn.Sequential(
              nn.Linear(in_dim, in_dim),
              nn.ReLU(inplace=True),
              nn.Linear(in_dim, feat_dim)
          )
      elif head_type == 'linear':
          self.net = nn.Linear(in_dim, feat_dim)
      else:
          raise ValueError(f"Unsupported head type: {head_type}")

  def forward(self, x):
      return F.normalize(self.net(x), dim=1)
  

class ConTinyMPNCOVResNet(nn.Module):
    def __init__(self, 
                 attention='Cov', 
                 input_size=64, 
                 feat_dim=128, 
                 head='mlp',
                 in_ch=37):
        super().__init__()
        
        block = Bottleneck
        layers = [3, 4, 6, 3]  # ResNet-50 style
        
        # Backbone encoder with MPNCOV representation
        self.encoder = TinyMPNCOVResNetBackbone(
            block=block,
            layers=layers,
            attention=attention,
            input_size=input_size,
            in_ch=in_ch
        )
        
        # Projection head
        self.head = ProjectionHead(
            in_dim=self.encoder.out_dim,
            feat_dim=feat_dim,
            head_type=head
        )

    def forward(self, x, return_embedding=False):
        emb = self.encoder(x)  # [B, 1280]
        if return_embedding:
            return emb
        return self.head(emb)  # [B, feat_dim]


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)