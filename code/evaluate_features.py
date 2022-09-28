import numpy as np
import torch
from piq import FID, KID, PR

models = ['baseline', 'large', 'CELoss', 'CELoss_ft']
model_features = {'baseline' : 0, 'large' : 0, 'CELoss' : 0, 'CELoss_ft' : 0} # fill in model features pt

real_feats = torch.load('real_feats_25k.pt')

fid_metric = FID()
kid_metric = KID()
pr_metric = PR()


print(f'|models |  FID  |  KID  |  PR   |')
print(f'_________________________________')
for model in models:
    features = torch.load(model_features[model])

    fid_score = round(fid_metric(real_feats, features).item(), 3)
    kid_score = round(kid_metric(real_feats, features).item(), 3)
    pr_score = round(pr_metric(real_feats, features).item(), 3)

    print(f'{model} | {fid_score} | {kid_score} | {pr_score} |')