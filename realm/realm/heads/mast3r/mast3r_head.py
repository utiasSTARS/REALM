from types import SimpleNamespace
import torch
import torch.nn as nn
from copy import deepcopy
from functools import partial

import realm.path_to_dust3r # noqa
from croco.models.pos_embed import RoPE2D
from realm.heads.mast3r.catmlp_dpt_head import mast3r_head_factory
from croco.models.blocks import DecoderBlock, PositionGetter
from dust3r.utils.misc import transpose_to_landscape

inf = float('inf')

class Mast3rDecoder(nn.Module):

    def __init__(self, 
                 enc_embed_dim=768, 
                 dec_embed_dim=512, 
                 dec_depth=8, 
                 dec_num_heads=16, 
                 patch_embed_size=14,
                 mlp_ratio=4,
                 norm_im2_in_dec=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 landscape_only=False,
                 desc_conf_mode=('exp', 0, inf),
                 desc_mode='norm',
                 two_confs=True
                 ):
        super().__init__()
        self.siamese = True # for compatibility with REALM
        self.desc_conf_mode = desc_conf_mode
        self.position_getter = PositionGetter()
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.desc_conf_mode = desc_conf_mode
        self.patch_embed = SimpleNamespace(patch_size=patch_embed_size)
        self.patch_embed_size = patch_embed_size
        self.desc_mode = desc_mode
        self.two_confs = two_confs

        # self.norm = torch.nn.LayerNorm(enc_embed_dim) # TODO: cehck the numeber of features
        self.rope = RoPE2D(freq=100.) # RoPE100
        
        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        # head
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size=16, img_size=(448, 448))

        if patch_embed_size != 16:  # might depend on mast3r stuff
            # we need to hack the head a bit ...
            for head in [self.downstream_head1, self.downstream_head2]:
                head.dpt.patch_size = (patch_embed_size, patch_embed_size)
                head.dpt.P_H = max(1, patch_embed_size // head.dpt.stride_level)
                head.dpt.P_W = max(1, patch_embed_size // head.dpt.stride_level)
                head.dpt.head[1].scale_factor *= 14 / 16

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)
        
    def forward(self, f1, f2, H, W):
        B, _, _ = f1.size()

        W_p = W
        W_p //=self.patch_embed_size
        H_p = H
        H_p //= self.patch_embed_size
        pos1 = self.position_getter(B, H_p, W_p, f1.device)
        pos2 = self.position_getter(B, H_p, W_p, f2.device)

        # feat1 = self.norm(f1)
        # feat2 = self.norm(f2)

        dec1, dec2 = self._decoder(f1, pos1, f2, pos2)

        shape1 = torch.tensor((H, W))[None].repeat(B, 1)
        shape2 = torch.tensor((H, W))[None].repeat(B, 1)

        with torch.amp.autocast('cuda', enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
        
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        
        return res1, res2
    
    def get_decoder_output(self, f1, f2, H, W):
        """ This is used to fine tune the decoder """
        
        B, _, _ = f1.size()

        W //= self.patch_embed_size
        H //= self.patch_embed_size

        pos1 = self.position_getter(B, H, W, f1.device)
        pos2 = self.position_getter(B, H, W, f2.device)

        return self._decoder(f1, pos1, f2, pos2)