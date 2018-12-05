from __future__ import absolute_import
import torch as t
from torch import nn
# from torchvision.models import vgg16
from pretrainedmodels import pnasnet5large
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt


class Extractor_Module(nn.Module):
    def __init__(self, model, num_classes=10):
        super(Extractor_Module, self).__init__()
        self.num_classes = num_classes
        self.conv_0 = model.conv_0
        self.cell_stem_0 = model.cell_stem_0
        self.cell_stem_1 = model.cell_stem_1
        self.cell_0 = model.cell_0
        self.cell_1 = model.cell_1
        self.cell_2 = model.cell_2
        self.cell_3 = model.cell_3
        self.cell_4 = model.cell_4
        self.cell_5 = model.cell_5
        self.cell_6 = model.cell_6
        self.cell_7 = model.cell_7
        self.cell_8 = model.cell_8
        self.cell_9 = model.cell_9
        self.cell_10 = model.cell_10
        self.cell_11 = model.cell_11
        self.relu = model.relu
        
    def features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        # print("{}\n{}\n\n{}\n{}".format('='*16, x_cell_2.size(), x_cell_3.size(), '='*16))
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.relu(x_cell_11)
        # print(x.size())
        return x

    def forward(self, input):
        x = self.features(input)
        return x


class Classifier_Module(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier_Module, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(512*7*7, 4096)

    def forward(self, x):
        x = self.dropout(x)
        x = self.last_linear(x)
        return x


def decom_pnasnet5large(n_classes, init_weights=True):
    # the 30th layer of features is relu of conv5_3
    if opt.pretrain:
        model = pnasnet5large(pretrained='imagenet', num_classes=1000)
        if opt.load_path:
            model.load_state_dict(t.load(opt.pretrain_path))
    else:
        model = pnasnet5large(pretrained=False, num_classes=1000)

    extractor = Extractor_Module(model, num_classes=n_classes)
    # classifier = Classifier_Module(num_classes=n_classes)
    classifier = nn.Sequential(
        nn.Linear(4320 * 6 * 6, 100),
    )

    if init_weights:
        for m in classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
    # features = 

    # classifier = Classifier_Module
    # classifier = list(classifier)

    # del classifier[6]
    # if not opt.use_drop:
    #     del classifier[5]
    #     del classifier[2]
    # classifier = nn.Sequential(*classifier)

    # # freeze top4 conv
    # for layer in features[:10]:
    #     for p in layer.parameters():
    #         p.requires_grad = False

    return extractor, classifier


class FasterRCNNpnasnet5large(FasterRCNN):
    """Faster R-CNN based on PNASnet5Large.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        extractor, classifier = decom_pnasnet5large(n_fg_class)

        rpn = RegionProposalNetwork(
            4320, 120,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = Pnasnet5largeRoIHead(
            n_class=n_fg_class + 1,
            roi_size=6,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNpnasnet5large, self).__init__(
            extractor,
            rpn,
            head,
        )


class Pnasnet5largeRoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(Pnasnet5largeRoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        print(indices_and_rois.size())
        print(x.size())
        pool = self.roi(x, indices_and_rois)
        print(pool.size())
        pool = pool.view(pool.size(0), -1)
        print(pool.size())
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


