import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .preprocess import cog_uncertainty_sample, cog_uncertainty_normal, reparameterize
from .attention import ChannelGate

# ========== VAE Encoder ==========
class EncoderVAE(Module):
    def __init__(self, in_feat, latent_dim):
        super(EncoderVAE, self).__init__()
        self.fc_mu = nn.Linear(in_feat, latent_dim)
        self.fc_logvar = nn.Linear(in_feat, latent_dim)
    
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ========== Overall Encoder ==========
class Encoder_overall(Module):
    def __init__(self, dim_in_omics1, dim_out_omics1, dim_in_omics2, dim_out_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        
        # Omics-specific encoders (Standard GCN encoder)
        self.encoder_omics1 = Encoder(dim_in_omics1, dim_out_omics1, dropout=0.5)
        self.encoder_omics2 = Encoder(dim_in_omics2, dim_out_omics2, dropout=0.5)

        # VAE encoders for each modality
        self.vae_encoder_omics1 = EncoderVAE(dim_out_omics1, dim_out_omics1)
        self.vae_encoder_omics2 = EncoderVAE(dim_out_omics2, dim_out_omics2)

        # Decoders (Standard GCN decoder)
        self.decoder_omics1 = Decoder(dim_out_omics1, dim_in_omics1)
        self.decoder_omics2 = Decoder(dim_out_omics2, dim_in_omics2)
        
        # 融合模块：输入通道为2（两个模态拼接），输出 128 维
        #self.fusion = ChannelGate(2, 128, pool_types='avg')
        self.atten_omics1 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_omics2 = AttentionLayer(dim_out_omics2, dim_out_omics2)
        self.atten_cross = AttentionLayer(dim_out_omics1, dim_out_omics2)
        # 后续全连接层，均使用 128 维
        self.mu = nn.Linear(128, 128)
        self.logvar = nn.Linear(128, 128)
        #self.IB_classfier = nn.Linear(128, 3)
        self.IB_classfier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )
        self.fc_fusion1 = nn.Sequential(
            nn.Linear(128, 128)
        )

    def forward(self, features_omics1, features_omics2, adj_spatial, adj_feature_omics1, adj_feature_omics2):
        '''
        # GCN encoder processing
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial)  # Hs1
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial)  # Hs2
        '''
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial)  # Hs1
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial)  # Hs2

        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        emb_latent_omics1 ,alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1,emb_latent_feature_omics1) 
        emb_latent_omics2 ,alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2,emb_latent_feature_omics2) 
        # VAE encoding for each modality
        mu_omics1, logvar_omics1 = self.vae_encoder_omics1(emb_latent_omics1)
        mu_omics2, logvar_omics2 = self.vae_encoder_omics2(emb_latent_omics2)
        var_omics1 = torch.exp(logvar_omics1)
        var_omics2 = torch.exp(logvar_omics2)
        
        def get_supp_mod(key):
            if key == "l":
                return mu_omics1
            elif key == "v":
                return mu_omics2
            else:
                raise KeyError
        
        # UDR、GatingFunction、Sum 的实现
        l_sample, v_sample = cog_uncertainty_sample(mu_omics1, var_omics1, mu_omics2, var_omics2, sample_times=30)
        sample_dict = {"l": l_sample, "v": v_sample}
        cog_uncertainty_dict = {}

        with torch.no_grad():
            for key, sample_tensor in sample_dict.items():
                bsz, sample_times, dim = sample_tensor.shape
                sample_tensor = sample_tensor.reshape(bsz * sample_times, dim)
                sample_tensor = sample_tensor.unsqueeze(1)  # [bsz*sample_times, 1, dim]
                
                # 获取对应的模态
                supp_mod = get_supp_mod(key)  # [bsz, dim]
                supp_mod = supp_mod.unsqueeze(1)  # [bsz, 1, dim]
                supp_mod = supp_mod.unsqueeze(1).repeat(1, sample_times, 1, 1)  # [bsz*sample_times, sample_times, 1, dim]
                supp_mod = supp_mod.reshape(bsz * sample_times, 1, dim)  # Reshaped to [bsz*sample_times, 1, dim]
                
                # 确保 sample_tensor 的维度与 supp_mod 匹配
                sample_tensor = sample_tensor.squeeze(1)  # [bsz*sample_times, 1, dim] (matching dimensions with supp_mod)
               
                # 拼接模态特征
                feature = torch.cat([supp_mod, sample_tensor.unsqueeze(1)], dim=1)  # [bsz*sample_times, 2, dim]
            
                # 使用 attention 进行融合
                emb_fusion, _ = self.atten_cross(supp_mod.squeeze(1), sample_tensor)  # [bsz*sample_times, 128]
                
                # 计算 mu 和 logvar
                _mu = self.mu(emb_fusion)  # [bsz*sample_times, 128]
                _logvar = self.logvar(emb_fusion)  # [bsz*sample_times, 128]
                
                # 重新参数化（VAE）
                _z = reparameterize(_mu, torch.exp(_logvar))
                
                # 分类
                _z = self.IB_classfier(_z)  # [bsz*sample_times, 3]
                
                # 输出
                o1_o2_out = self.fc_fusion1(_mu)  # [bsz*sample_times, 128]
                cog_un = torch.var(o1_o2_out, dim=-1)  # 计算认知不确定性
                
                # 保存认知不确定性
                cog_uncertainty_dict[key] = cog_un

        # 归一化认知不确定性
        cog_uncertainty_dict = cog_uncertainty_normal(cog_uncertainty_dict)

        # 根据不确定性计算权重
        weight = torch.softmax(torch.stack([var_omics1, var_omics2]), dim=0)
        w_omics1 = weight[0]
        w_omics2 = weight[1]

        # 将注意力权重与不确定性权重结合
        combined_weight_omics1 = w_omics1 * alpha_omics1.sum(dim=-1, keepdim=True)
        combined_weight_omics2 = w_omics2 * alpha_omics2.sum(dim=-1, keepdim=True)
        #combined_weight_omics1 = w_omics1
        #combined_weight_omics2 = w_omics2

        # 用加权的权重对特征进行加权
        emb_omics1 = mu_omics1 * combined_weight_omics1
        emb_omics2 = mu_omics2 * combined_weight_omics2
        # 拼接两个模态的特征，形状 [batch, 2, latent_dim]；此处 latent_dim 期望为 128
        emb = torch.stack((emb_omics1, emb_omics2), dim=1)
        
        # 最终融合：使用拼接后的 emb 进行融合
        #emb_fusion = self.fusion(emb)  # 输出: [batch, 128]
        emb_fusion,alpha = self.atten_cross(emb_omics1,emb_omics2)
        mu_out = self.mu(emb_fusion)   # [batch, 128]
        logvar_out = self.logvar(emb_fusion)  # [batch, 128]
        z_out = reparameterize(mu_out, torch.exp(logvar_out))
        z_out = self.IB_classfier(z_out)  # [batch, 3]（用于后续分类）
        omics1_omics2_out = self.fc_fusion1(mu_out)  # 输出: [batch, 128]

        # Reconstruction via decoders
        emb_recon_omics1 = self.decoder_omics1(omics1_omics2_out, adj_spatial)
        emb_recon_omics2 = self.decoder_omics2(omics1_omics2_out, adj_spatial)

        z_omics1 = reparameterize(mu_omics1,torch.exp(logvar_omics1))
        z_omics2 = reparameterize(mu_omics2,torch.exp(logvar_omics2))

        results = {
            'mu_omics1': mu_omics1, 
            'logvar_omics1': logvar_omics1, 
            'mu_omics2': mu_omics2, 
            'logvar_omics2': logvar_omics2, 
            'emb_recon_omics1': emb_recon_omics1, 
            'emb_recon_omics2': emb_recon_omics2,
            'z_omics1': z_omics1,
            'z_omics2': z_omics2,
            'emb_latent_combined': mu_out,
            'alpha':alpha,
        }
        return results

# ========== Standard GCN Encoder ==========
class Encoder(Module): 
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x

# ========== Standard GCN Decoder ==========
class Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x

# ========== Attention Layer (保留原版) ==========
class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = torch.cat([torch.unsqueeze(emb1, dim=1), torch.unsqueeze(emb2, dim=1)], dim=1)
        v = F.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu.squeeze() + 1e-6,dim=-1)
        emb_combined = torch.matmul(torch.transpose(emb,1,2), torch.unsqueeze(alpha, -1))
        emb_combined = F.relu(emb_combined)  # Add ReLU for non-linearity
        return emb_combined.squeeze(), alpha