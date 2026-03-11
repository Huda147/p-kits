import numpy as np
import itertools

# =============================================================================
# 1. Define the Problem (8 Cities)
# =============================================================================
N = 5
num_vars = N * N

# Generate a reproducible, random 8x8 distance matrix
np.random.seed(42) 
distances = np.random.randint(10, 100, size=(N, N))
distance_matrix = (distances + distances.T) // 2 
np.fill_diagonal(distance_matrix, 0)

print(f"--- 1. GENERATED {N}x{N} DISTANCE MATRIX ---")
print(distance_matrix)
print("\n")

# =============================================================================
# 2. Software Solver (The Ground Truth Validator)
# =============================================================================
print("--- 2. SOFTWARE EXACT SOLVER ---")
# To avoid checking the same circular route multiple times, we fix City 0 as the start
cities_to_permute = list(range(1, N))
best_distance = float('inf')
best_route = []

# Check all 5,040 permutations
for perm in itertools.permutations(cities_to_permute):
    current_route = [0] + list(perm)
    
    # Calculate total distance (including the trip back to the start)
    dist = 0
    for i in range(N):
        city_a = current_route[i]
        city_b = current_route[(i + 1) % N]
        dist += distance_matrix[city_a, city_b]
        
    if dist < best_distance:
        best_distance = dist
        best_route = current_route

print(f"Optimal Route Found: {best_route}")
print(f"Total Minimum Distance: {best_distance}\n")

# =============================================================================
# 3. Build QUBO and Convert to Ising (J and h)
# =============================================================================
print("--- 3. HARDWARE COMPILER (Generating J and h) ---")
max_dist = np.max(distance_matrix)
A_penalty = max_dist * 1.5  # Must be much larger than max distance to force valid routes
B_reward  = 1.0    

Q = np.zeros((num_vars, num_vars))

# Constraint 1: Every city visited once
for v in range(N):
    for t in range(N):
        idx = v * N + t
        Q[idx, idx] -= A_penalty 
        for t2 in range(N):
            if t != t2:
                Q[idx, v * N + t2] += A_penalty 

# Constraint 2: Every time step has one city
for t in range(N):
    for v in range(N):
        idx = v * N + t
        Q[idx, idx] -= A_penalty 
        for v2 in range(N):
            if v != v2:
                Q[idx, v2 * N + t] += A_penalty 

# Objective: Minimize Distance
for u in range(N):
    for v in range(N):
        if u != v:
            for t in range(N):
                idx_u_t = u * N + t
                idx_v_next = v * N + ((t + 1) % N) 
                dist_penalty = B_reward * distance_matrix[u, v] / 2.0
                Q[idx_u_t, idx_v_next] += dist_penalty
                Q[idx_v_next, idx_u_t] += dist_penalty

# Convert QUBO to Ising
J = np.zeros((num_vars, num_vars))
h = np.zeros(num_vars)

for i in range(num_vars):
    for j in range(num_vars):
        if i != j:
            J[i, j] = -Q[i, j] / 2.0
    h[i] = -0.5 * np.sum(Q[i, :])

# =============================================================================
# 4. Quantize and Export to 16-Bit .mem Files
# =============================================================================
max_val = max(np.max(np.abs(J)), np.max(np.abs(h)))
scale_factor = 32767.0 / max_val # CHANGED: Scale up to 16-bit limit

J_quant = np.round(J * scale_factor).astype(int)
h_quant = np.round(h * scale_factor).astype(int)
np.fill_diagonal(J_quant, 0)

def to_hex_16bit(val):
    # CHANGED: 16-bit Two's Complement (4 Hex Characters)
    return f"{(val & 0xFFFF):04X}" 

with open("tsp_J_matrix.mem", "w") as f:
    for row in J_quant:
        f.write(" ".join([to_hex_16bit(val) for val in row]) + "\n")

with open("tsp_h_vector.mem", "w") as f:
    for val in h_quant:
        f.write(to_hex_16bit(val) + "\n")

print(f"Success! Exported 16-bit matrices to .mem files.")
