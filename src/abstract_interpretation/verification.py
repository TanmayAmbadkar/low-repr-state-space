import torch

def verify_network(model, input_domain):
    abstract_element = input_domain
    with torch.no_grad():
        for layer in model.layers:
            abstract_element = layer(abstract_element)
    
    return abstract_element
