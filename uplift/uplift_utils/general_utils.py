"""
General utils for UPLiFT.

Code by: Saksham Suri and Matthew Walmer
"""
import os
import natsort


# count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# [B, T, C] --> [B, C, H, W]
def convert_shape(x, H, W):
    x = x.permute(0, 2, 1)
    x = x.reshape(x.shape[0], -1, H, W)
    return x


# reshape feature to the right shape for cosine similarity loss
def reshape_for_cos(feat):
    feat = feat.permute(0, 2, 3, 1)
    feat = feat.reshape(-1, feat.shape[-1])
    return feat


# delete older checkpoints to reduce storage usage
def limit_checkpoints(logdir, num_keep=3, keyword='lift', only_list=False):
    files = os.listdir(logdir)
    checkpoints = []
    for f in files:
        if '.pth' in f and keyword in f:
            checkpoints.append(f)
    checkpoints = natsort.natsorted(checkpoints)
    for i in range(num_keep):
        if len(checkpoints) == 0: break
        checkpoints.pop(-1)
    for f in checkpoints:
        fp = os.path.join(logdir, f)
        if only_list:
            print('WOULD DELETE: ' + fp)
        else:
            print('DELETING: ' + fp)
            os.remove(fp)