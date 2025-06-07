import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device  # Get device information
    B = points.shape[0]  # Get batch size

    # Create lists for view shape and repeat shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    # Create tensor containing integers from 0 to B-1 for batch dimension indexing
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    # Rearrange point cloud data using indices
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device  # Get device type of point cloud data
    B, N, C = xyz.shape  # Get batch size, number of points and dimensions per point
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # Initialize sampled point cloud indices
    distance = torch.ones(B, N).to(device) * 1e10  # Initialize distance with large value
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # Randomly select initial point
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # Generate sequence of integers from 0 to B-1
    for i in range(npoint):
        centroids[:, i] = farthest  # Add current farthest point to sampled point cloud indices
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Get coordinates of farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)  # Calculate distance between each point and farthest point
        distance = torch.min(distance, dist)  # Update distance, keep minimum distance to all sampled points
        farthest = torch.max(distance, -1)[1]  # Select point with maximum distance as next sampling point
    return centroids  # Return sampled point cloud indices

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device  # Get device type of point cloud data
    B, N, C = xyz.shape  # Get batch size, number of points and dimensions per point
    _, S, _ = new_xyz.shape  # Get batch size, number of points and dimensions per point for query points
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # Initialize grouped point indices
    sqrdists = square_distance(new_xyz, xyz)  # Calculate squared distance between query points and all points
    group_idx[sqrdists > radius ** 2] = N  # Set indices of points beyond radius to N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # Sort points around each query point and select first nsample points
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # Get index of first point in each local region
    mask = group_idx == N  # Create mask for points not in local region
    group_idx[mask] = group_first[mask]  # Replace indices of points not in local region with first point index
    return group_idx  # Return grouped point indices

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)

        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )

        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):

        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):

        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


##############-----MSFA------###########
# Non-Parametric Networks for 3D Point Cloud Part Segmentation
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors_FPS_kNN):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors_FPS_kNN = k_neighbors_FPS_kNN

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors_FPS_kNN, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim - 4, N)
        padding = torch.zeros(B, 4, N).cuda()
        position_embed = torch.cat((position_embed, padding), dim=1)
        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, feat_dim * 6, G, K)

        padding = torch.zeros(B, self.out_dim - feat_dim * 6, G, K).cuda()
        position_embed = torch.cat((position_embed, padding), dim=1)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w

import torch
import torch.nn as nn
import torch.nn.functional as F

################--Point Transformer--####################
class PointTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dim_head, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(PointTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# CCTA 模块
class CCTA(nn.Module):
    def __init__(self, input_channel):
        super(CCTA, self).__init__()
        self.con2d = nn.Conv2d(in_channels=input_channel*2, out_channels=input_channel, kernel_size=1, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.Softmax2d = nn.Softmax2d()
        self.ACD = ACD(input_channel, 0.5)

    def forward(self, inputs):
        input_dim0 = int(inputs.shape[0])
        input_dim1 = int(inputs.shape[1])
        input_dim2 = int(inputs.shape[2])
        input_dim3 = int(inputs.shape[3])

        a = inputs.permute((0, 3, 1, 2))
        a = a.reshape((input_dim0, input_dim3, input_dim2, input_dim1))
        a = self.Softmax2d(a)
        a_probs = a.permute((0, 3, 2, 1))

        b = inputs.permute((0, 2, 3, 1))
        b = b.reshape((input_dim0, input_dim2, input_dim3, input_dim1))
        b = self.Softmax2d(b)
        b_probs = b.permute((0, 3, 1, 2))

        A = torch.mul(a_probs, b_probs)
        A = torch.exp(A)
        Maximum = torch.maximum(a_probs, b_probs)
        knowledgelayer = A + Maximum + inputs
        k = self.ACD([A, Maximum, knowledgelayer])

        return k

class ACD(nn.Module):
    def __init__(self, units, Thr, **kwargs):
        super(ACD, self).__init__(**kwargs)
        self.units = units
        self.Thr = Thr

    def forward(self, x):
        assert isinstance(x, list)
        H1, V1, X1 = x

        cos1 = torch.multiply(torch.sqrt(torch.multiply(H1, H1)), torch.sqrt(torch.multiply(V1, V1)))
        cosin1 = torch.multiply(H1, V1) / cos1

        Zeos = torch.zeros_like(X1)
        Ones = torch.ones_like(X1)

        Y = torch.where(cosin1 > self.Thr, Ones, Zeos)
        concatenate = torch.cat([Y * X1], axis=3)

        return concatenate

class CSCA(nn.Module):
    def __init__(self, input_channel, transformer_dim, num_heads, dim_head):
        super(CSCA, self).__init__()
        self.point_transformer = PointTransformerLayer(dim=transformer_dim, num_heads=num_heads, dim_head=dim_head)
        self.ccta = CCTA(input_channel=transformer_dim)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1)
        transformer_output = self.point_transformer(x)
        transformer_output = transformer_output.permute(0, 2, 1).unsqueeze(3)
        attention_output = self.ccta(transformer_output)
        attention_output = attention_output.squeeze(3).permute(0, 2, 1)
        attention_output = attention_output.permute(0, 2, 1)
        return attention_output

class RTreeNet(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2],gmp_dim=64,cls_dim=64,
                 input_points=2048, num_stages=4,de_neighbors=6, beta=1000, alpha=100,k_neighbors_FPS_kNN=128,embed_dim_new = 64,**kwargs):
        super(RTreeNet, self).__init__()
        self.stages = len(pre_blocks)  # Number of model stages
        self.class_num = num_classes  # Number of classes
        self.points = input_points  # Number of points
        self.embedding = ConvBNReLU1D(6, embed_dim_new, bias=bias, activation=activation)

        # Initialize attention
        self.CSCA_64 = CSCA(input_channel=64, transformer_dim=64, num_heads=4, dim_head=16)

        # Parameter initialization
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim_new = embed_dim_new
        self.alpha, self.beta = alpha, beta
        self.de_neighbors = de_neighbors
        out_dim = self.embed_dim_new
        group_num = self.input_points

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim_new, self.alpha, self.beta)
        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.LGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors_FPS_kNN))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()  # Store local neighborhood information
        self.pre_blocks_list = nn.ModuleList()  # Store preprocessing blocks
        self.pos_blocks_list = nn.ModuleList()  # Store position extraction blocks
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]

        ### Building Encoder ###
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)

        ### Building Decoder ###
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        self.act = get_activation(activation)

        # Class label mapping
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(1, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        )
        # Global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            index_points(xyz1, idx)
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points

    def forward(self, x, norm_plt, cls_label):
        xyz = x.permute(0, 2, 1)
        x = torch.cat([x,norm_plt],dim=1)
        x = self.embedding(x)
        x = self.CSCA_64(x)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]

        # Start MSFA module
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)
            xyz_list.append(xyz)
            x_list.append(x)
        # End MSFA module

        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]

        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)

        # Global context processing
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))

        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1))
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

def get_model(num_classes=2, **kwargs) -> RTreeNet:
    return RTreeNet(num_classes=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],gmp_dim=64,cls_dim=64,
                input_points=2048, num_stages=4,de_neighbors=6, beta=1000, alpha=100,k_neighbors_FPS_kNN=128,embed_dim_new=64,**kwargs)

# Alias for backward compatibility
RTreeNet_model = get_model

if __name__ == '__main__':
    data = torch.rand(2, 3, 2048)
    norm = torch.rand(2, 3, 2048)
    cls_label = torch.rand([2, 2])
    print("===> testing modelD ...")
    model = RTreeNet(2)
    out = model(data, cls_label)
    print(model.shape)
