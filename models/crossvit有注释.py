import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from models.Block import Block
from models.Identity import Identity
from models.drop_path import DropPath
from models.helpers import to_2tuple
from models.mlp import Mlp
import mindspore.common.initializer as init
from mindspore.common.initializer import TruncatedNormal


#ms.set_seed(7)
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

# 动态图的配置
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE)

_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=True):
        super().__init__()
        img_size = to_2tuple(img_size)  # 生成224*224
        patch_size = to_2tuple(patch_size)  # 生成16*16
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 像素面积的缩小倍数
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:  # conv2d修改 sequentialcell修改
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, pad_mode='pad', kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, pad_mode='pad', kernel_size=3, stride=3, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, pad_mode='pad', kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, pad_mode='pad', kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, pad_mode='pad', kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, pad_mode='pad', kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid')
            #如果不需要padding，可以改为valid，ms默认进行填充

    def construct(self, x):
        B, C, H, W = x.shape  # x的四维
        # FIXME look at relaxing size constraints

        # assert H == self.img_size[0] and W == self.img_size[1], \
        # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # print(x)
        B, C, H, W = x.shape
        # print(B,C,H,W)
        x = x.reshape(B, C, H * W)  #flatten无法在特定维度展开
        x = ops.transpose(x, (0, 2, 1))
        return x  # 修改，输出结果的shape大小相同，但是数据不相同


# 这个函数的问题，flatten的问题，无法在特定维度进行展开,已经解决,只是输出不同，应该是初始权重的问题，类型规格相同
# net1 = PatchEmbed()
# x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
# out = net1(x)
# print(out.shape)
# print(out)
# 此函数静态图没有问题


class CrossAttention(nn.Cell):  # 交叉注意力的算法
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 数组是否可以整除？
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Dense(dim, dim, has_bias=qkv_bias)  # 修改
        self.wk = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.wv = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)  # 修改，反人类化，pyops表示丢弃，ms表示保留的概率
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

    def construct(self, x):  # transformer的qkv进行计算，修改construct

        # print(x)
        B, N, C = x.shape  # 3,3,16
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads)
        q = ops.transpose(q, (0, 2, 1, 3))  # B1C -> B1H(C/H) -> BH1(C/H) 3 8 1 2

        # ops的transpose只能转换一个维度，而permute可以转换好几个维度，修改
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = ops.transpose(k, (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H) 3832

        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = ops.transpose(v, (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)3832

        # 修改@的结果运算以及，transpose的转换
        batchmatual = ops.BatchMatMul(transpose_b=True)
        attn = batchmatual(q, k) * self.scale
        # print(attn) 与pyops相同

        # attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        softmax = nn.Softmax()
        attn = softmax(attn)
        # print(attn)  相同
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(attn)

        batchmatual2 = ops.BatchMatMul()
        x = batchmatual2(attn, v)
        # print(attn.shape)
        # print(v.shape)
        # print(x.shape)

        x = ops.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, 1, C)
        # print(x.shape)

        # print(x)
        # x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        # print(111111)
        # print(x)
        x = self.proj_drop(x)
        # print(x.shape)
        # print(2222)

        return x


# net2 = CrossAttention(dim=16)
# x = ms.Tensor(np.ones((3, 3, 16)), ms.float32)
# x = net2(x)
# print(x)
# 此函数静态图输出没有问题

class CrossAttentionBlock(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()

        self.norm1 = norm_layer((dim,))
        # print(self.norm1(x))
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # print(self.attn(self.norm1(x)))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.has_mlp = has_mlp
        if has_mlp:  # 进入这个分支
            self.norm2 = norm_layer((dim,))  # normlayer无法输入单个数字
            mlp_hidden_dim = int(dim * mlp_ratio)
            # 修改mlp
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        # 这个0:1表示只选择这个维度的两行
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


#
# x = ms.Tensor(np.ones((3, 3, 16)), ms.float32)
# #print(x.shape)
# net3 = CrossAttentionBlock(dim=16, num_heads=4)
# x = net3(x)
# print(x)
# 此函数静态图没有问题
# 经过检测，目前前三个函数的输出shape没有问题


class MultiScaleBlock(nn.Cell):
    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches  # 等于dim的长度
        # different branch could have different embedding size, the first one is the base  不同的拥有不懂的尺寸，第一个是基础
        blocks = [] #如果直接定义多个celllsit，会和后面的celllist的形成网络参数重名的问题。
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                blocks.append(nn.SequentialCell(tmp))  # sequential只能使用列表或者字典进行添加，可以使用类的列表添加
        if len(blocks) == 0:  # blocks当中装着每一个都是sequential都可以直接输入x进行求解运算，可以使用下标找寻
            self.blocks = None
        else:
            self.blocks = nn.CellList(blocks)

        # self.ccc = nn.CellList()
        # for d in range(num_branches):
        #     tmp = []
        #     for i in range(depth[d]):
        #         tmp.append(
        #             Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
        #                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer))
        #     if len(tmp) != 0:
        #         self.ccc.append(nn.SequentialCell(tmp))  # sequential只能使用列表或者字典进行添加，可以使用类的列表添加
        # if len(self.ccc) == 0:
        #     self.ccc = None
        #
        # print(self.ccc)  #从输出来看，结构形式是一模一样的
        projs = []
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [Identity()]
            else:
                tmp = [norm_layer((dim[d],), epsilon=1e-6), act_layer(), nn.Dense(dim[d], dim[(d + 1) % num_branches])]
            projs.append(nn.SequentialCell(tmp))
        self.projs = nn.CellList(projs)

        fusion = []
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                tmp2 = [CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                            norm_layer=norm_layer,
                                            has_mlp=False)]
                fusion.append(nn.SequentialCell(tmp2))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                   norm_layer=norm_layer,
                                                   has_mlp=False))
                fusion.append(nn.SequentialCell(tmp))
        self.fusion = nn.CellList(fusion)
        revert_projs = []
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [Identity()]
            else:
                tmp = [norm_layer((dim[(d + 1) % num_branches],), epsilon=1e-6), act_layer(),
                       nn.Dense(dim[(d + 1) % num_branches], dim[d])]
            revert_projs.append(nn.SequentialCell(tmp))
        self.revert_projs = nn.CellList(revert_projs)

    def construct(self, x):
        # x = x.asnumpy()
        # x = x.tolist()
        # print(self.blocks)
        # x = [x, ]
        outs_b = []  #静态图不支持列表生成式
        i = 0
        for block in self.blocks:
            outs_b.append(block(x[i]))
            i = i + 1
        # outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = []
        j = 0
        for proj in self.projs:
            proj_cls_token.append(proj(outs_b[j][:, 0:1]))
            j = j + 1
        # outs_b = [block(x_) for x_, block in zip([x], self.blocks)]
        # only take the cls token out
        # proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            # print(proj_cls_token[i].shape)
            # print(outs_b[(i + 1) % self.num_branches][:, 1:, ...].shape)
            a = proj_cls_token[i]
            # print(type(a))
            b = outs_b[(i + 1) % self.num_branches][:, 1:, ...]
            # print(type(b))
            con = ops.Concat(1)
            # print("a", a.dtype)
            # print("b", b.dtype)
            tmp = con((a, b))
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = con((reverted_proj_cls_token, outs_b[i][:, 1:, ...]))
            outs.append(tmp)
        return outs


# net3 = MultiScaleBlock(dim=(192,), patches=[400], depth=[1, 0], num_heads=[6], mlp_ratio=(2, 2),
#                        qkv_bias=False, drop_path=[0.5, 0.5], qk_scale=None, drop=0, attn_drop=0,
#                        norm_layer=nn.LayerNorm)
# # for name, param in net3.parameters_and_names():
# #     print(name, param)
# x = ms.Tensor(np.ones((3, 192, 192)), ms.float32)
# x = net3(x)
# print(x)


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size, patches)]


def interploate(self, x, output_size, size):
    B, N, C = x.shape
    H, W = size


class VisionTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        # print(num_patches)
        self.num_branches = len(patch_size)

        patch_embed = []
        if hybrid_backbone is None:  # 循环修改，因为tuple只接受元祖，并且元祖无法改变元祖内的元素
            b = []
            for i in range(self.num_branches):
                c = ms.Parameter(Tensor(np.zeros([1, 1 + num_patches[i], embed_dim[i]], np.float32)), name=str(i + 100))
                b.append(c)
            b = tuple(b)
            self.pos_embed = ms.ParameterTuple(b)
            #self.pos_embed = b
            # print(self.pos_embed)
            # self.pos_embed = ms.ParameterTuple(
            #     tuple([ms.Parameter(ops.Zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)]))
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                patch_embed.append(
                    PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
            self.patch_embed = nn.CellList(patch_embed)  # 修改
        else:
            # self.pos_embed = ms.ParameterTuple()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            b = []  # 修改
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                c = ms.Parameter(get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]),
                                 name=str(idx + 500), requires_grad=False)
                b.append(c)
            self.patch_embed = nn.CellList(patch_embed)
            b = tuple(b)
            self.pos_embed = ms.ParameterTuple(b)
            #self.pos_embed = b

            del self.pos_embed
            m = []  # 修改
            for i in range(self.num_branches):
                c = ms.Parameter(Tensor(np.zeros([1, 1 + num_patches[i], embed_dim[i]], np.float32)), name=str(i + 200))
                m.append(c)
            m = tuple(b)
            self.pos_embed = ms.ParameterTuple(m)
            #self.pos_embed = b

        d = []
        for i in range(self.num_branches):  # 修改
            # print("i ",i)
            c = ms.Parameter(Tensor(np.zeros([1, 1, embed_dim[i]], np.float32)), name=str(i + 300))
            d.append(c)
        # print(d)
        d = tuple(d)
        self.cls_token = ms.ParameterTuple(d)
        # self.cls_token = d
        # print(self.cls_token)
        self.pos_drop = nn.Dropout(1.0 - drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        # print(drop_path_rate, total_depth)
        # 原代码是列表，这个也是
        dpr = np.linspace(0, drop_path_rate, total_depth)  # stochastic depth decay rule
        # print(dpr)
        dpr_ptr = 0
        self.blocks = nn.CellList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            # print(dpr_) 四个元素，均为0
            # print(block_cfg)
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.CellList([norm_layer((embed_dim[i],), epsilon=1e-6) for i in range(self.num_branches)])
        self.head = nn.CellList([nn.Dense(embed_dim[i], num_classes) if num_classes > 0 else Identity() for i in
                                 range(self.num_branches)])

        for i in range(self.num_branches):
            # print(type(self.pos_embed[i]))
            if self.pos_embed[i].requires_grad:
                # print(Tensor(init.TruncatedNormal(sigma=.02), self.pos_embed[i].data.shape),ms.float32)
                # print(self.pos_embed[i])  # 是一个parameter需要进行初始化
                tensor1 = init.initializer(TruncatedNormal(sigma=.02), self.pos_embed[i].data.shape, ms.float32)
                # print(tensor1) 经过验证，tensor是一个tensor类型
                # self.pos_embed[i].set_data(
                #     init.initializear(TruncatedNormal(sigma=.02), self.pos_embed[i].data.shape))
                self.pos_embed[i].set_data(tensor1)
            tensor2 = init.initializer(TruncatedNormal(sigma=.02), self.cls_token[i].data.shape, ms.float32)
            # print(self.cls_token[i])
            # tensor2 = init.initializear(TruncatedNormal(sigma=.02), self.cls_token[i].data.shape, ms.float32)
            self.cls_token[i].set_data(tensor2)
            # cc=self.cls_token[i].T
            # print("cc", cc)  均可以进行转置
            # self.pos_embed[i].set_data(init.initializear(TruncatedNormal(sigma=.02), self.cls_token[i].data.shape))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.Constant(1), cell.gamma.shape))
                cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape))

    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        # print(x)
        for i in range(self.num_branches):
            # print(i, self.num_branches)
            # print(self.cls_token[i])
            resize_bilinear = nn.ResizeBilinear()  # 将cubic插值转换为线性插值
            x_ = resize_bilinear(x, size=(self.img_size[i], self.img_size[i])) if H != self.img_size[i] else x
            # print(i, " yes1")
            tmp = self.patch_embed[i](x_)
            # print(i, " yes2")
            z = self.cls_token[i].shape
            # print(i, " yes3")
            y = Tensor(np.ones((B, z[1], z[2])), dtype=mstype.float32)
            # print(i, " yes4")
            # print(z)
            # print(self.cls_token[i])
            # print(y.shape)
            cls_tokens = self.cls_token[i]
            cls_tokens = cls_tokens.expand_as(y)  # stole cls_tokens impl from Phil Wang, thanks
            # print(cls_tokens.shape)
            # print(cls_tokens)
            # print(i, " yes5")
            con = ops.Concat(1)
            # print(type(tmp))
            # print(type(cls_tokens))
            # print(i, " yes6")
            # print("cls_tokens", cls_tokens.dtype)
            # print("tmp", tmp.dtype)
            tmp = con((cls_tokens, tmp))
            # print(i, " yes7")
            tmp = tmp + self.pos_embed[i]
            # print(i, " yes8")
            tmp = self.pos_drop(tmp)
            # print(i, " yes9")
            xs.append(tmp)
            # print(222222)

        for blk in self.blocks:
            # print(333333)
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        k = 0
        xs2 = []
        for x in xs:
            xs2.append(self.norm[k](x))
            k=k+1
        # xs = [self.norm[i](x) for i, x in enumerate(xs)]
        xs = xs2
        out = []
        for x in xs:
            out.append(x[:, 0])
        # out = [x[:, 0] for x in xs]

        return out

    def construct(self, x):
        xs = self.forward_features(x)
        # print(xs)
        # print(11111)
        ce_logits=[]
        zz = 0
        for x in xs:
            ce_logits.append(self.head[zz](x))
            zz = zz+1
        #ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        # print(len(ce_logits))
        z = ops.stack([ce_logits[0], ce_logits[1]])
        # print(z.shape)
        op = ops.ReduceMean(keep_dims=False)
        ce_logits = op(z, 0)
        # ce_logits = ops.ReduceMean(ops.stack(ce_logits, 0), 0)
        # ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0) #拼接之后求平均

        return ce_logits


# net4 = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
#                               num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
#                               norm_layer=nn.LayerNorm)

# net4 = VisionTransformer(img_size=[240],
#                          patch_size=[12], embed_dim=[192], depth=[[1, 4, 0]],
#                          num_heads=[3], mlp_ratio=[4], qkv_bias=True,
#                          norm_layer=nn.LayerNorm)

#crossvit_tiny_224模型主体函数
net4 = VisionTransformer(img_size=[240, 224],
                         patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                         num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                         norm_layer=nn.LayerNorm)

# for name, param in net4.parameters_and_names():
#     print(name, param)

x = ms.Tensor(np.ones((3, 3, 1, 16)), ms.float32)
out = net4(x)
# print(net4)
print(out)

# @register_model
# def crossvit_tiny_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
#                               num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
#                               norm_layer=nn.LayerNorm, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         # state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_tiny_224'], map_location='cpu')
#         # model.load_state_dict(state_dict)
#         load_pretrained(model, model.default_cfg)
#     return model

# @register_model
# def crossvit_small_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
#                               num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_small_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def crossvit_base_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
#                               num_heads=[12, 12], mlp_ratio=[4, 4, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_base_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def crossvit_9_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
#                               num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def crossvit_15_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
#                               num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def crossvit_18_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
#                               num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def crossvit_9_dagger_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
#                               num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_dagger_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
# @register_model
# def crossvit_15_dagger_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
#                               num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
# @register_model
# def crossvit_15_dagger_384(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[408, 384],
#                               patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
#                               num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_384'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
# @register_model
# def crossvit_18_dagger_224(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[240, 224],
#                               patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
#                               num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_224'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
#
# @register_model
# def crossvit_18_dagger_384(pretrained=False, **kwargs):
#     model = VisionTransformer(img_size=[408, 384],
#                               patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
#                               num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
#                               norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_384'], map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model