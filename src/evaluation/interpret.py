import torch

def compute_integrated_gradients(model, x, target_class_idx, steps=20, device=None):
    """Computes 1D Integrated Gradients attribution for a target diagnostic class.
    
    x shape: [12, Length]
    Returns: attribution of shape [12, Length] representing local clinical saliency.
    """
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 1. Expand input to batch of size 1 and move to device
    x = x.unsqueeze(0).to(device) # [1, 12, Length]
    x.requires_grad = True
    
    # 2. Define baseline (all zeros, representing standard flatline baseline)
    baseline = torch.zeros_like(x)
    
    # 3. Generate interpolated inputs along the linear path from baseline to input
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    interpolated_inputs = []
    for alpha in alphas:
        interpolated = baseline + alpha * (x - baseline)
        interpolated_inputs.append(interpolated)
        
    # Stack along batch dimension
    interpolated_inputs = torch.cat(interpolated_inputs, dim=0) # [steps, 12, Length]
    interpolated_inputs.requires_grad = True
    
    # 4. Compute model forward pass and gradients for all steps
    logits = model(interpolated_inputs) # [steps, num_classes]
    
    # Target class logits
    target_logits = logits[:, target_class_idx]
    
    # Backward pass to get gradients
    target_logits.sum().backward()
    
    # Retrieve gradients
    grads = interpolated_inputs.grad # [steps, 12, Length]
    
    # 5. Integrate (average) the gradients and scale by (input - baseline)
    avg_grads = torch.mean(grads, dim=0) # [12, Length]
    delta = (x - baseline).squeeze(0) # [12, Length]
    
    # Attribution is average gradient multiplied by delta
    attribution = avg_grads * delta
    
    # Return as numpy array
    return attribution.detach().cpu().numpy()
