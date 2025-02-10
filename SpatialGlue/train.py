import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import Encoder_overall 
from .preprocess import adjacent_matrix_preprocessing

class Train_SpatialGlue:
    def __init__(self, 
        data,
        datatype='SPOTS',
        device=torch.device('cpu'),
        random_seed=2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=128,
        weight_factors=[1, 5, 1, 1, 1]  # 添加 KL 散度的权重因子
    ):
        """
        VAE + GCN + EURO + UDR 训练过程
        """
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        # 预处理邻接矩阵
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial = self.adj['adj_spatial_omics1'].to(self.device)  # RNA 和 Protein 共享
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # 处理输入特征
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        # 设置输入/输出维度
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        # 适配不同数据类型
        if self.datatype == 'SPOTS':
            self.epochs = 600 
            self.weight_factors = [1, 5, 1, 1, 0.1]  # KL 权重因子
        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 1500 
            self.weight_factors = [1, 10, 1, 10, 0.05]
        elif self.datatype == '10x':
            self.epochs = 200
            self.weight_factors = [1, 5, 1, 10, 0.05]
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
            self.epochs = 1600
            self.weight_factors = [1, 5, 1, 1, 0.05]

    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(
                self.features_omics1, self.features_omics2, self.adj_spatial, 
                self.adj_feature_omics1, self.adj_feature_omics2
            )
            
            # 计算 Reconstruction Loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            '''
            *******************************
            一致性损失弃用
            # 计算 Correspondence Loss
            self.loss_corr_omics1 = F.mse_loss(results['z_omics1'], results['mu_omics1'])
            self.loss_corr_omics2 = F.mse_loss(results['z_omics2'], results['mu_omics2'])
            *******************************
            '''
            # 计算 KL 散度损失（VAE 正则化）
            kl_div_omics1 = -0.5 * torch.sum(1 + results['logvar_omics1'] - results['mu_omics1']**2 - torch.exp(results['logvar_omics1']))
            kl_div_omics2 = -0.5 * torch.sum(1 + results['logvar_omics2'] - results['mu_omics2']**2 - torch.exp(results['logvar_omics2']))
            self.loss_kl = (kl_div_omics1 + kl_div_omics2) / self.n_cell_omics1

            # 计算 UDR 认知不确定性
            self.cog_uncertainty_dict = {
                'rna': torch.var(results['mu_omics1'], dim=0).mean(),
                'protein': torch.var(results['mu_omics2'], dim=0).mean()
            }

            # 计算最终损失
            loss = (self.weight_factors[0] * self.loss_recon_omics1 
                    + self.weight_factors[1] * self.loss_recon_omics2 
                    #+ self.weight_factors[2] * self.loss_corr_omics1 
                    #+ self.weight_factors[3] * self.loss_corr_omics2
                    + self.weight_factors[4] * self.loss_kl)  # KL 散度

            # EURO 机制调整梯度
            loss = self.apply_euro_gradient_adjustment(loss, self.cog_uncertainty_dict)

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial, self.adj_feature_omics1, self.adj_feature_omics2)
 
        emb_omics1 = F.normalize(results['z_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['z_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {
            'z_omics1': emb_omics1.detach().cpu().numpy(),
            'z_omics2': emb_omics2.detach().cpu().numpy(),
            'SpatialGlue': emb_combined.detach().cpu().numpy(),
        }
        
        return output

    def apply_euro_gradient_adjustment(self, loss, cog_uncertainty_dict):
        """
        EURO 机制：根据 UDR 计算不确定性，自适应调整梯度。
        """
        coeff_rna = torch.clamp(1 + 0.1 * cog_uncertainty_dict['protein'], min=0.8, max=1.2)
        coeff_protein = torch.clamp(1 + 0.1 * cog_uncertainty_dict['rna'], min=0.8, max=1.2)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if 'encoder_omics1' in name:
                    param.grad.mul_(coeff_rna)
                if 'encoder_omics2' in name:
                    param.grad.mul_(coeff_protein)
        return loss