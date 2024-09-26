import torch
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage, GaussianBlur
import clip
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class DDR(nn.Module):
    def __init__(self, device = device,  CLIP_visual_model="ViT-B/32", degradations = ['blur','color','noise','exposure']):
        super(DDR, self).__init__()
        self.device = device
        self.CLIP_model, _ = clip.load(CLIP_visual_model, device= self.device)
        self.n_px = self.CLIP_model.visual.input_resolution
        self.transform = Compose([
            Resize(self.n_px, interpolation=BICUBIC, antialias=False),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        
        # set all the parameters to be not trainable except trainable
        for param in self.CLIP_model.parameters():
            param.requires_grad = False
            
        self.degradtion_type = degradations
        
        self.clean_prompt = {}
        self.degraded_prompt = {}
        self.clean_prompt['color'] = "A real color photo with high-quality."
        self.degraded_prompt['color'] = "A unnatural color photo with low-quality."
        self.clean_prompt['noise'] = "A clean photo with high-quality."
        self.degraded_prompt['noise'] = "A noise degraded photo with low-quality."
        self.clean_prompt['blur'] = "A sharp photo with high-quality."
        self.degraded_prompt['blur'] = "A blurry photo with low-quality."
        self.clean_prompt['content'] = "A clear content photo with high-quality."
        self.degraded_prompt['content'] = "A bad content photo with low-quality."
        self.clean_prompt['exposure'] = "A natural exposure photo with high-quality."
        self.degraded_prompt['exposure'] = "A unnatural exposure photo with low-quality."
        
        
        self.clean_prompt_feature = {}
        self.degraded_prompt_feature = {}
        for degrad_type in self.degradtion_type:
            self.clean_prompt_feature[degrad_type] = self.CLIP_model.encode_text(clip.tokenize(self.clean_prompt[degrad_type]).to(self.device))
            self.degraded_prompt_feature[degrad_type] = self.CLIP_model.encode_text(clip.tokenize(self.degraded_prompt[degrad_type]).to(self.device))
        
    def cosine_distance(self, x, y):
        '''
            x,y : torch.Tensor
            calculate the cosine distance between x and y
        '''
        # normalized features
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        logits =  x @ y.t()
        return 1. - logits

    def DDR_score(self, x_features):
        '''
        x_features : torch.Tensor
        cal the DDR score of the input image feature
        '''
        dst_score = []
        
        for dst_type in self.degradtion_type:
            degradation = self.degraded_prompt_feature[dst_type] - self.clean_prompt_feature[dst_type]
            degradation = self.adaptation(degradation, x_features)
            x_degraded_features = x_features + degradation

            self_sim = self.cosine_distance(x_features, x_degraded_features)
            self_sim = torch.cat([self_sim[i][i].unsqueeze(0) for i in range(self_sim.shape[0])])
            dst_score.append(self_sim)
            
        result = torch.stack(dst_score).mean(dim = 0)
        
        return result

    def DDR_quality_score(self, x):
        '''
        x : torch.Tensor, with shape (batch_size, 3, H, W)
        utilize DDR as BIQA metric
        '''

        # crop the image into patches
        x_patches = []
        H, W = x.shape[-2], x.shape[-1]
        patch_size = min(H, W)
        patch_number = max(H, W) // patch_size + 1
        stride_H = (H - patch_size) // (patch_number - 1)
        stride_W = (W - patch_size) // (patch_number - 1)
        if H > W:
            for i in range(patch_number):
                patch_position_H = i * stride_H
                patch_position_W = 0
                x_patch = x[:, :, patch_position_H: patch_position_H + patch_size, patch_position_W: patch_position_W + patch_size]
                x_patch = self.transform(x_patch)
                x_patches.append(x_patch)
        else:
            for i in range(patch_number):
                patch_position_H = 0
                patch_position_W = i * stride_W
                x_patch = x[:, :, patch_position_H: patch_position_H + patch_size, patch_position_W: patch_position_W + patch_size]
                x_patch = self.transform(x_patch)
                x_patches.append(x_patch)
        
        # mean the result of all the patches
        x = self.transform(x)
        x_patches = torch.cat(x_patches, dim=0)
        patch_features = self.CLIP_model.encode_image(x_patches)
        patch_results = []
        for patch_feature in patch_features:
            patch_feature = patch_feature.unsqueeze(0)
            patch_result = self.DDR_score(patch_feature)
            patch_results.append(patch_result)
            
        result = torch.stack(patch_results).mean(dim = 0).squeeze(0)
        return result
            
    def adaptation(self, content, style):
        '''
        content : torch.Tensor
        content and style : torch.Tensor
        '''
        mu_c = content.mean(dim = -1, keepdim = True)
        sigma_c = content.std(dim = -1, keepdim = True)
        mu_s = style.mean(dim = -1, keepdim = True)
        sigma_s = style.std(dim = -1, keepdim = True)
        
        eps = 0.001
        
        content = (content - mu_c) / (sigma_c + eps)
        content = content * sigma_s + mu_s
        return content

    
    def forward(self, x):
        '''
        x : torch.Tensor, with shape (batch_size, 3, H, W)
        utilize DDR as unsupervised learning objective (higher DDR score means higher quality)
        '''
        # assert x.shape == y.shape, "x and y must have the same shape"
        x = self.transform(x)
        x_features = self.CLIP_model.encode_image(x)
        score_x = self.DDR_score(x_features)
        score_x = score_x.mean()
        return -1.0 * score_x

    
# if __name__ == '__main__':
#     image = Image.open("test.png")
#     x = torch.stack([ToTensor()(image)]).to(device)
#     criterion = DDR()
#     res = criterion(x)
#     print("res:", res)
    
    