# models/supcon_clip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# --- Model Definition ---
class SupConClipModel(nn.Module):
    """Wraps CLIP encoder and added projector for contrastive learning."""
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        try: self.encoder_dtype = next(self.encoder.parameters()).dtype
        except StopIteration: self.encoder_dtype = torch.float32
        logger.info(f"SupConClipModel using encoder dtype: {self.encoder_dtype}")

    def forward(self, x):
        # Ensure input dtype matches encoder expectation
        features = self.encoder(x.to(self.encoder_dtype))
        # Ensure input to projector is float32, then normalize output
        projection = self.projector(features.float())
        normalized_projection = F.normalize(projection, dim=-1)
        return normalized_projection

# --- Unfreezing Utility ---
def unfreeze_visual_encoder_layers(visual_model:  nn.Module, unfreeze_last_n_blocks: int):
    resblocks = None
    if hasattr(visual_model, 'transformer'):
        num_blocks = len(visual_model.transformer.resblocks)
        resblocks = visual_model.transformer.resblocks
    if unfreeze_last_n_blocks == 0:
        unfreeze_start_index = num_blocks
    elif unfreeze_last_n_blocks >= num_blocks:
        unfreeze_start_index = 0
    else:
        unfreeze_start_index = num_blocks - unfreeze_last_n_blocks

    for param in visual_model.parameters():
        param.requires_grad = False
    if resblocks is not None:
        for i, block in enumerate(resblocks):
            if i >= unfreeze_start_index:
                for param in block.parameters():
                    param.requires_grad = True
    
    ln_post_unfrozen = False
    if hasattr(visual_model, 'ln_post') and isinstance(visual_model.ln_post, nn.LayerNorm):
        for param in visual_model.ln_post.parameters():
            param.requires_grad = True
            ln_post_unfrozen = True
        if ln_post_unfrozen: print("Unfroze ln_post.")
    else: print("Cannot find ln_post.")
    
    proj_unfrozen = False
    if hasattr(visual_model, 'proj'):
        if isinstance(visual_model.proj, torch.Tensor) and visual_model.proj.is_leaf:
            visual_model.proj.requires_grad = True
            proj_unfrozen = True
        elif isinstance(visual_model.proj, nn.Module):
            for param in visual_model.proj.parameters():
                param.requires_grad = True; proj_unfrozen = True
        else: print(f"proj type {type(visual_model.proj)} not handled.")
        if proj_unfrozen: print("Unfroze proj.")
    else: print("Cannot find proj.")

    trainable = sum(p.numel() for p in visual_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in visual_model.parameters())
    print(f"Total number of trainable parameters: {trainable}/{total}")

# --- Loss Function ---
def contrastive_loss(features, labels, temperature=0.07, eps=1e-12):

    device=features.device; 
    batch_size=features.shape[0]; 

    features=F.normalize(features,p=2,dim=1)
    similarity_matrix=torch.matmul(features, features.T)/temperature
    labels_view=labels.contiguous().view(-1, 1)
    mask=torch.eq(labels_view, labels_view.T).float().to(device)
    self_mask=torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size,device=device).view(-1, 1), 0).to(device)
    mask=mask*self_mask
    
    logits_max, _ =torch.max(similarity_matrix,dim=1,keepdim=True)
    logits=similarity_matrix-logits_max.detach(); exp_logits=torch.exp(logits)*self_mask
    log_prob=logits-torch.log(exp_logits.sum(1,keepdim=True)+eps)
    mean_log_prob_pos=(mask*log_prob).sum(1)/(mask.sum(1).clamp(min=eps))
    loss=-mean_log_prob_pos.mean()
    if torch.isnan(loss): logger.warning("NaN SupCon loss!"); loss=torch.tensor(0.0,device=device,requires_grad=True)
    return loss
