import numpy as np
import torch

class Zonotope:
    def __init__(self, center, generators):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.generators = [torch.tensor(g, dtype=torch.float32) for g in generators]
    
    def affine_transform(self, W, b):
        W = torch.tensor(W, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        new_center = W @ self.center + b
        new_generators = [W @ g for g in self.generators]
        return Zonotope(new_center, new_generators)
    
    def relu(self):
        new_center = torch.relu(self.center)
        new_generators = []
        for g in self.generators:
            new_g = torch.where(self.center > 0, g, torch.zeros_like(g))
            new_generators.append(new_g)
        return Zonotope(new_center, new_generators)
    
    def to_hyperplanes(self):
        """
        Convert the zonotope to a set of hyperplane inequalities.
        Each generator contributes two hyperplanes.
        """
        c = self.center.numpy()
        G = np.array([g.numpy() for g in self.generators])

        num_generators = G.shape[0]
        input_dim = G.shape[1]

        A = np.vstack([G, -G])
        b = np.ones(2 * num_generators)

        inequalities = []
        for i in range(2 * num_generators):
            inequalities.append((A[i], np.dot(A[i], c) + b[i]))
        return inequalities
    
class Box:
    def __init__(self, lower, upper):
        self.lower = torch.tensor(lower, dtype=torch.float32)
        self.upper = torch.tensor(upper, dtype=torch.float32)
    
    def affine_transform(self, W, b):
        W = torch.tensor(W, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        new_lower = W @ self.lower + b
        new_upper = W @ self.upper + b
        return Box(new_lower, new_upper)
    
    def relu(self):
        new_lower = torch.relu(self.lower)
        new_upper = torch.relu(self.upper)
        return Box(new_lower, new_upper)


# # Define the observation space bounds
# obs_space_lower = np.array([-1.5, -1.5, -5.0, -5.0, -3.1415927, -5.0, 0.0, 0.0])
# obs_space_upper = np.array([ 1.5,  1.5,  5.0,  5.0,  3.1415927,  5.0, 1.0, 1.0])

# # Calculate the center of the zonotope
# center = (obs_space_lower + obs_space_upper) / 2

# # Create generators to reflect the constraints on vx and vy
# generators = []

# # vx and vy constraints (-2 <= vx, vy <= 2)
# vx_gen = np.zeros(8)
# vx_gen[2] = 2

# vy_gen = np.zeros(8)
# vy_gen[3] = 2

# # Add these generators to the list
# generators.append(vx_gen)
# generators.append(vy_gen)

# # Additional generators for other dimensions can be added if necessary
# # Example: small perturbations in other dimensions
# for i in range(8):
#     if i not in [2, 3]:
#         gen = np.zeros(8)
#         gen[i] = (obs_space_upper[i] - obs_space_lower[i]) / 2
#         generators.append(gen)

# # Create the zonotope
# input_zonotope = Zonotope(center, generators)

# # Print the zonotope details
# print("Zonotope center:", input_zonotope.center)
# print("Zonotope generators:")
# for gen in input_zonotope.generators:
#     print(gen)