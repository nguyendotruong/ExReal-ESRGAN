import torch

stage1_path = r'experiments/train_ExRealESRNetx4plus/models/net_g_latest.pth'    
stage2_path = r'experiments/train_ExRealESRGANx4plus/models/net_g_latest.pth'   
alpha = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_state(d):
    if 'params_ema' in d:
        return d['params_ema']
    elif 'params' in d:
        return d['params']
    elif 'state_dict' in d:
        return d['state_dict']
    elif 'network_g' in d:
        return d['network_g']
    return d  

w1 = extract_state(torch.load(stage1_path, map_location=device))
w2 = extract_state(torch.load(stage2_path, map_location=device))

interp = {}
for k, v in w1.items():
    if k in w2 and hasattr(v, 'shape') and hasattr(w2[k], 'shape') and v.shape == w2[k].shape:
        interp[k] = ((1 - alpha) * v + alpha * w2[k]).to('cpu')
    else:
        interp[k] = v.to('cpu')

out_path = f'weights/ExReal-ESRGAN.pth'
torch.save({'params': interp}, out_path)
print('Saved:', out_path)

