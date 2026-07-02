
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.parameter import Parameter
from layers import GraphConvolution
import numpy as np


class DrugGCN(nn.Module):
    def __init__(self, input_dim=491, output_dim=64, hidden_dim=128):
        super(DrugGCN, self).__init__()
        # 分子图编码器
        self.mol_gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.mol_gcn2 = GraphConvolution(hidden_dim, output_dim)

        # 药物特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, adj_matrix, drug_memory):
        """
        adj_matrix: (131, 491) 分子二部图特征
        drug_memory: (131, 64) 药物特征表示
        """
        # 方案1：直接使用分子图作为特征
        x = adj_matrix  # (131, 491)

        # 构建药物-药物关系图（这里使用单位矩阵+余弦相似度）
        drug_sim = F.cosine_similarity(drug_memory.unsqueeze(1),
                                       drug_memory.unsqueeze(0), dim=2)
        adj = torch.eye(131).to(adj_matrix.device) + 0.5 * drug_sim  # (131,131)

        # 归一化邻接矩阵
        rowsum = adj.sum(1)
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        adj = r_mat_inv @ adj @ r_mat_inv

        # GCN编码分子图
        x = self.mol_gcn1(x, adj)  # (131, hidden_dim)
        x = self.relu(x)
        x = self.dropout(x)
        mol_features = self.mol_gcn2(x, adj)  # (131, output_dim)

        # 与原始药物特征融合
        fused_features = self.fusion(
            torch.cat([mol_features, drug_memory], dim=-1)
        )
        return fused_features


class GraphConvolution(nn.Module):
    """GCN层实现"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(64, 64)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

class CrossBiAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 双向注意力
        self.query_to_key_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.key_to_query_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # 融合层（保留但调整使用顺序）
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

        # 前馈网络和归一化
        self.ffn_q = nn.Sequential(  # 为Q路径单独添加FFN
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_k = nn.Sequential(  # 为K路径单独添加FFN
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_k = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # 原始输入保存用于残差连接
        orig_query = query.unsqueeze(0)  # (1, B, D)
        orig_key = key.unsqueeze(0)      # (1, B, D)

        # 双向注意力（与图中两个Cross Attention对应）
        attn_q2k, _ = self.query_to_key_attn(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
        attn_k2q, _ = self.key_to_query_attn(key.unsqueeze(0), query.unsqueeze(0), query.unsqueeze(0))

        # 独立FFN处理（对应图中两个Feed Forward）
        ffn_q = self.ffn_q(self.norm1_q(orig_query + attn_q2k))  # (1, B, D)
        ffn_k = self.ffn_k(self.norm1_k(orig_key + attn_k2q))    # (1, B, D)

        # 最后融合双向结果
        combined = torch.cat([ffn_q, ffn_k], dim=-1)              # (1, B, 2*D)
        fused = self.dropout(self.fusion(combined))               # (1, B, D)

        # 最终归一化（对应图中最后的Add & Norm）
        output = self.norm2(fused)
        return output.squeeze(0)  # (B, D)

class ExplainDrug(nn.Module):
    """
    The ExplainDrug class
    """

    def __init__(self, num_diag, num_pro, num_asp=1958, e_dim=64):
        super(ExplainDrug, self).__init__()
        self.num_asp = num_asp  # number of aspects
        self.asp_emb = Aspect_emb(num_asp, e_dim)
        self.mlp = nn.Sequential(nn.Linear(e_dim, 3*e_dim), nn.ReLU(), nn.Linear(3*e_dim, num_asp))
        self.e_dim = e_dim
        self.mapping = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 131))

    def forward(self,patient_representations, asp):

        # diag_latent = x
        # pro_latent = y
        # patient_representations = torch.cat([diag_latent, pro_latent], dim=-1)
        if patient_representations.dim() == 1:
            patient_representations = patient_representations.unsqueeze(0)
        patient_representations = patient_representations[-1 : ] # 将patient_representations中的最后一个元素（或者称为最后一个值）提取出来，作为本次患者表示。
        # out = self.mapping(query)
        # TODO:detach
        detached_query_latent = patient_representations.detach()  # gradient shielding 可解释模块中患者嵌入不进行反向传播

        asp_latent = self.asp_emb(asp.unsqueeze(0))
        # （1,128） -》 （1,1958）
        factor = F.softmax(self.mlp(detached_query_latent), dim=-1).unsqueeze(-1)

        patient_asp = torch.bmm(asp_latent.permute(0, 2, 1), factor).squeeze(-1)

        # # cosine similarity between patient_asp and patient
        sim = - F.cosine_similarity(patient_asp, detached_query_latent, dim=-1)
        return sim

class Aspect_emb(nn.Module):
    """
    module to embed each aspect to the latent space.
    """

    def __init__(self, num_asp, e_dim):
        super(Aspect_emb, self).__init__()
        # 使用参数化的权重矩阵W来表示所有方面的潜在表示
        # 所拥有aspect方面的数量
        self.num_asp = num_asp
        self.W = nn.Parameter(torch.randn(num_asp, e_dim))  # 包含所有方面的潜在表示，其中的元素是从标准正态分布（均值为0，方差为1）中随机采样的

    # 此处的x输入的是第i次诊断，第i次治疗的二进制编码
    def forward(self, x):
        shape = x.shape
        x = x.reshape([x.shape[0], x.shape[1], 1])
        # 变换维度X（1,1958,1）->(1,1958,128)
        x = x.expand(-1, -1, self.W.shape[1]) # 将x与self.W的形状对齐，以便后续的矩阵乘法。
        # 逐元素相乘，得到每个方面的潜在表示，即将原来的编码 Ed = W * 可解释性的特征
        asp_latent = torch.mul(x, self.W)  # [1, num_asp, e_dim]
        # 可选择对每个方面的潜在表示进行标准化
        # asp_latent = F.normalize(asp_latent, p=2, dim=2)

        return asp_latent




class ExplainDrug_drug(nn.Module):
    def __init__(self, num_diag, num_pro, num_asp=1958, e_dim=64):
        super(ExplainDrug_drug, self).__init__()
        self.num_asp = num_asp  # number of aspects
        self.asp_emb = Aspect_emb_drug(num_asp, 64)  # Aspect embedding layer
        # MLP for generating aspect weights
        self.mlp = nn.Sequential(
            nn.Linear(e_dim, 3 * e_dim),  # Input dimension increased to 3 * e_dim
            nn.ReLU(),
            nn.Linear(3 * e_dim, num_asp)
        )
        self.e_dim = e_dim
        # Mapping layer for drug memory
        self.drug_memory_mapping = nn.Sequential(
            nn.Linear(64, e_dim),  # Map drug memory to e_dim
            nn.ReLU()
        )

    def forward(self, patient_representations, asp, drug_memory=None):
        # 如果patient_representations的维度为1，则扩展为2维
        if patient_representations.dim() == 1:
            patient_representations = patient_representations.unsqueeze(0)
        # 将patient_representations中的最后一个元素提取出来，作为本次患者表示
        patient_representations = patient_representations[-1:]


        detached_query_latent = patient_representations.detach()

        asp_latent = self.asp_emb(asp.unsqueeze(0),drug_memory)

        factor = F.softmax(self.mlp(detached_query_latent), dim=-1).unsqueeze(-1)

        patient_asp = torch.bmm(asp_latent.permute(0, 2, 1), factor).squeeze(-1)

        sim = -F.cosine_similarity(patient_asp, detached_query_latent, dim=-1)
        return sim

class Aspect_emb_drug(nn.Module):
    def __init__(self, num_asp, e_dim):
        super(Aspect_emb_drug, self).__init__()
        self.num_asp = num_asp
        self.W = nn.Parameter(torch.randn(num_asp, e_dim))  # [num_asp, e_dim]

    def forward(self, x, drug_memory):
        x = x.reshape([x.shape[0], x.shape[1], 1])  # [1, 131, 1]

        # 结合输入x和潜在表示
        asp_latent = torch.mul(x.expand(-1, -1, self.W.size(1)), self.W + drug_memory)  # [1, 131, e_dim]
        return asp_latent

class AggregationFFN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 定义一个简单的FFN：输入是 2*hidden_dim（拼接后的维度），输出是 hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, drug_memory, drug_representation):
        # 拼接两个输入（按特征维度）
        combined = torch.cat([drug_memory, drug_representation], dim=-1)  # shape: (131, 128)
        # 通过FFN聚合
        output = self.ffn(combined)  # shape: (131, 64)
        return output

class DMGExNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        emb_dim=128,
        device=torch.device("cpu:0"),
    ):
        super(DMGExNet, self).__init__()

        self.device = device
        self.emb_dim = emb_dim

        # pre-embedding

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], 64) for i in range(3)]
        )
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList(
            [nn.TransformerEncoderLayer(emb_dim, 4, 4, batch_first=True, dropout=0.2) for _ in
             range(3)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))


        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()
        # Cross-Attention
        self.cross_attention = CrossBiAttention(emb_dim, num_heads=4)
        # aspect
        self.exp_diag = ExplainDrug(num_diag=4233, num_pro=1958, num_asp=1958, e_dim=64)
        self.exp_pro = ExplainDrug(num_diag=4233, num_pro=1430, num_asp=1430, e_dim=64)
        self.exp_med = ExplainDrug_drug(num_diag=4233, num_pro=131, num_asp=131, e_dim=64)
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(1, emb_dim))
        self.mapping = nn.Sequential(nn.Linear(2*emb_dim, 64), nn.ReLU(), nn.Linear(64, 131))
        self.map_vocab_size = nn.Sequential(
            nn.Linear(2*emb_dim, vocab_size[2])
        )
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.bipartite_transform = nn.Sequential(
            nn.Linear(64, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)
        self.drug_gcn = DrugGCN(input_dim=491, output_dim=64)

    def forward(self, input, diag, pro, med, step):

        # patient health representation
        diag_seq = []
        proc_seq = []
        med_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:


            diag_1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            proc_1 = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            diag_seq.append(diag_1)
            proc_seq.append(proc_1)

        for idx, adm in enumerate(input):
            if len(input) <= 1 or idx==0:
                med_1 = torch.zeros((1, 1, 64)).to(self.device)
            else:
                 adm[2] = input[idx - 1][2][:]
                 med_1 = sum_embedding(
                     self.dropout(
                         self.embeddings[2](torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device))))
            med_seq.append(med_1)


        diag_seq = torch.cat(diag_seq, dim=1)  # (1,seq,dim)
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,seq,dim)
        med_seq = torch.cat(med_seq, dim=1)

        # Apply multi-scale convolution to diagnosis and procedure sequences

        o1 = self.encoders[0](diag_seq)
        o2 = self.encoders[1](proc_seq)
        o3= self.encoders[2](med_seq)

        o1 = o1.squeeze(dim=0).squeeze(dim=0)
        o2 = o2.squeeze(dim=0).squeeze(dim=0)
        o3 = o3.squeeze(dim=0).squeeze(dim=0)

        # Cross-Attention between o1 and o2
        patient_representation = self.cross_attention(o1, o2)  # (emb_dim,)
        # Combine with o3
        diag = diag.to(self.device)
        pro = pro.to(self.device)
        med = med.to(self.device)
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter
        drug_memory = self.drug_gcn(self.tensor_ddi_mask_H, drug_memory)
        diag_sim = self.exp_diag(patient_representation, diag[step])
        pro_sim = self.exp_pro(patient_representation, pro[step])
        med_sim = self.exp_med(patient_representation, med[step],drug_memory)

        sim = (diag_sim + pro_sim + med_sim)/3.0

        query = torch.cat([patient_representation, o3], dim=-1).squeeze(0)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query[-1:]
        out = self.mapping(query)

        neg_pred_prob = F.sigmoid(out)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = F.sigmoid(neg_pred_prob.mul(self.tensor_ddi_adj)).sum()
        return out, batch_neg, sim

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


