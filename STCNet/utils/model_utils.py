import torch
import os
from collections import OrderedDict
from loguru import logger 
from copy import deepcopy
def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

    
def load_network_with_ev_converter(net, load_path, current_ev_chn=24, pretrained_ev_chn=6, strict=True, param_key='state_dict'):
    """Load network with EV channel conversion.
    
    Args:
        net (nn.Module): Network to be loaded.
        load_path (str): Path to the pretrained weights.
        current_ev_chn (int): Number of EV channels in current model.
        pretrained_ev_chn (int): Number of EV channels in pretrained model.
        strict (bool): Whether to strictly enforce that the keys match.
        param_key (str): Key for parameters in checkpoint dict.
    """
    logger.info(f'Loading {net.__class__.__name__} model from {load_path}')
    
    # Load pretrained weights
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        load_net = load_net[param_key]
    
    # Remove unnecessary 'module.' prefix
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    
    # Handle EV channel mismatch
    model_dict = net.state_dict()
    for k, v in deepcopy(load_net).items():
        if current_ev_chn==pretrained_ev_chn:
            break
        logger.info(f"Shape of {k}: {v.shape}")
        if "encoder_event.head.shallow_feat.0.weight" in k:
            # Get original weight shape
            out_channels, _, kh, kw = v.shape
            # Create new weights by repeating original channels
            repeats = current_ev_chn // pretrained_ev_chn
            remainder = current_ev_chn % pretrained_ev_chn
            new_weight = v.repeat(1, repeats, 1, 1)
            if remainder > 0:
                new_weight = torch.cat([new_weight, v[:, :remainder, :, :]], dim=1)
            
            # Check new weight shape
            logger.info(f'Converted EV conv layer from {v.shape} to {new_weight.shape}')
            
            # Update weights
            load_net[k] = new_weight
        
        # Ensure all layers are handled, otherwise log it for debugging
        if k not in model_dict:
            logger.warning(f'Layer {k} not found in model state dict.')
    
    # Load model state dict with strict checking (or set strict=False)
    logger.info("Loading updated state dict into model...")
    try:
        net.load_state_dict(load_net, strict=False if not strict else True)
        logger.info("Model loaded successfully.")
    except RuntimeError as e:
        logger.error(f"Error loading model: {e}")
        raise e

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict,strict=True)