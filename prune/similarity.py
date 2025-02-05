import torch
import torch.nn.functional as F
def similarity_for_single_layer(a, b, similarity_type='cos'):
    """
    计算两个张量的余弦相似度
    :param similarity_type: 使用的相似度计算方法, cos-余弦, euc-欧式距离
    :param a: 形状为 (N, C, H, W) 的张量
    :param b: 形状为 (N, C, H, W) 的张量
    :return: 余弦相似度值
    """
    # 展开成 (N, C, -1) 的向量
    N, C = a.size(0), b.size(1)
    a = a.view(N, C, -1)
    b = b.view(N, C, -1)

    if similarity_type == 'cos':
    # 计算余弦相似度
        dist = F.cosine_similarity(a, b, dim=-1)
    # 计算欧式距离
    elif similarity_type == 'euc':
        dist = torch.norm(a - b, p=2, dim=-1)

    dist = torch.sum(dist, dim=0)
    return dist
