import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from preprocess import cog_uncertainty_sample , cog_uncertainty_normal , reparameterize
from attention import ChannelGate

# ========== VAE + GCN + EURO + UDR 整合模型 ==========
class EncoderVAE(Module):
    """
    VAE Encoder for a single modality.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    latent_dim: int
        Dimension of latent space.
    
    Returns
    -------
    mu : Mean of latent space.
    logvar : Log variance of latent space.
    """
    
    def __init__(self, in_feat, latent_dim):
        super(EncoderVAE, self).__init__()
        self.fc_mu = nn.Linear(in_feat, latent_dim)
        self.fc_logvar = nn.Linear(in_feat, latent_dim)
    
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Encoder_overall(Module):
    """\
    Overall encoder with VAE and GCN.

    Returns
    -------
    Dictionary with latent representations and uncertainty estimates.
    """
    
    def __init__(self, dim_in_omics1, dim_out_omics1, dim_in_omics2, dim_out_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        
        # Omics-specific encoders
        self.encoder_omics1 = Encoder(dim_in_omics1, dim_out_omics1)
        self.encoder_omics2 = Encoder(dim_in_omics2, dim_out_omics2)

        # VAE encoders
        self.vae_encoder_omics1 = EncoderVAE(dim_out_omics1, dim_out_omics1)
        self.vae_encoder_omics2 = EncoderVAE(dim_out_omics2, dim_out_omics2)

        # Decoders
        self.decoder_omics1 = Decoder(dim_out_omics1, dim_in_omics1)
        self.decoder_omics2 = Decoder(dim_out_omics2, dim_in_omics2)
        
        '''
        ***************************
        注意力层弃用
        # Attention layers
        self.atten_omics1 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_omics2 = AttentionLayer(dim_out_omics2, dim_out_omics2)
        self.atten_cross = AttentionLayer(dim_out_omics1, dim_out_omics2)
        ***************************
        '''

        self.fusion=ChannelGate(3,3,'avg')
        self.mu=nn.Linear(128,128)
        self.logvar=nn.Linear(128,128)
        self.IB_classfier=nn.Linear(128,3)
        self.fc_fusion1=nn.Sequential(nn.Linear(128,3))

    def forward(self, features_omics1, features_omics2, adj_spatial, adj_feature_omics1, adj_feature_omics2):
        
        # GCN encoder processing
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial)#Hs1
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial)#Hs2
        '''
        ***************************
        #以下两个Encoder弃用
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        #以下两个模态内注意力层弃用
        # Within-modality attention
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        ***************************
        '''
            
        # VAE encoding
        mu_omics1, logvar_omics1 = self.vae_encoder_omics1(emb_latent_spatial_omics1)
        mu_omics2, logvar_omics2 = self.vae_encoder_omics2(emb_latent_spatial_omics2)

        var_omics1 = torch.exp(logvar_omics1)
        var_omics2 = torch.exp(logvar_omics2)
        
        def get_supp_mod(key):
            if key == "l":
                return mu_omics1
            elif key == "v":
                return mu_omics2
            else:
                raise KeyError
            
        '''
        ============================
        UDR、GatingFunction、Sum的实现
        认知不确定性的计算
        ============================
        '''
        l_sample, v_sample = cog_uncertainty_sample(mu_omics1, var_omics1, mu_omics2, var_omics2, sample_times=10)  
        sample_dict = {
            "l": l_sample, 
            "v": v_sample
        }
        cog_uncertainty_dict = {}
        with torch.no_grad():
            for key, sample_tensor in sample_dict.items():
                bsz, sample_times, dim = sample_tensor.shape
                sample_tensor = sample_tensor.reshape(bsz * sample_times, dim)#多次采样
                sample_tensor = sample_tensor.unsqueeze(1)  
                supp_mod = get_supp_mod(key)
                supp_mod = supp_mod.unsqueeze(1)
                supp_mod = supp_mod.unsqueeze(1).repeat(1, sample_times, 1, 1)
                supp_mod = supp_mod.reshape(bsz * sample_times, 1, dim)  
                feature = torch.cat([supp_mod, sample_tensor], dim=1)
    
                feature_fusion=self.fusion(feature)#这就是GatingFunction
                mu=self.mu(feature_fusion)
                logvar=self.logvar(feature_fusion)
                z=reparameterize(mu,torch.exp(logvar))
                z=self.IB_classfier(z)
                o1_o2_out=self.fc_fusion1(mu)
                
                cog_un = torch.var(o1_o2_out, dim=-1)  #认知不确定性
                cog_uncertainty_dict[key] = cog_un
            
        cog_uncertainty_dict = cog_uncertainty_normal(cog_uncertainty_dict)


        '''
        ============================
        根据不确定性分配不同权重
        融合得到单个表征
        ============================
        '''
        weight = torch.softmax(torch.stack([var_omics1,var_omics2]),dim = 0)
        w_omics1 = weight[0]
        w_omics2 = weight[1]

        emb_omics1 = mu_omics1 * w_omics1
        emb_omics2 = mu_omics2 * w_omics2

        emb = torch.stack((emb_omics1,emb_omics2),dim = 1)
        emb_fusion = self.fusion(z)
        mu=self.mu(emb_fusion)
        logvar=self.logvar(emb_fusion)
        z=reparameterize(mu,torch.exp(logvar))
        z=self.IB_classfier(z)
        omics1_omics2_out=self.fc_fusion1(mu)

        '''
        ***************************
        z_omics1 = reparameterize(mu_omics1, logvar_omics1)
        z_omics2 = reparameterize(mu_omics2, logvar_omics2)

        # Between-modality attention
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(z_omics1, z_omics2)
        ***************************
        '''
        # Reconstruction via decoders
        emb_recon_omics1 = self.decoder_omics1(omics1_omics2_out, adj_spatial)
        emb_recon_omics2 = self.decoder_omics2(omics1_omics2_out, adj_spatial)

        results = {
            'mu_omics1': mu_omics1, 
            'logvar_omics1': logvar_omics1, 
            'mu_omics2': mu_omics2, 
            'logvar_omics2': logvar_omics2, 
            'emb_recon_omics1': emb_recon_omics1, 
            'emb_recon_omics2': emb_recon_omics2,
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

# ========== Attention Layer ==========
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
        alpha = F.softmax(vu.squeeze() + 1e-6)  
        emb_combined = torch.matmul(torch.transpose(emb,1,2), torch.unsqueeze(alpha, -1))
        return emb_combined.squeeze(), alpha      