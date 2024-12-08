import numpy as np
import matplotlib.pyplot as plt

# Parameters
total_steps = 150
warmup_steps = 10
lr_max = 5e-5

# Step values for plotting
steps = np.arange(total_steps)

# Schedulers
# Constant with Warmup
def constant_with_warmup(step, warmup_steps, lr_max):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    return lr_max

# Polynomial Decay
def polynomial_decay(step, warmup_steps, total_steps, lr_max, power=2):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    return lr_max * ((total_steps - step) / (total_steps - warmup_steps)) ** power

# Cosine Decay
def cosine_decay(step, warmup_steps, total_steps, lr_max):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    return lr_max * 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

# Inverse Square Root Decay
def inverse_sqrt_decay(step, warmup_steps, total_steps, lr_max):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    return lr_max * (warmup_steps / max(1, step)) ** 0.5

# Reduce LR on Plateau (simulated as a sharp drop after halfway)
def reduce_lr_on_plateau(step, warmup_steps, lr_max):
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    if step < total_steps / 2:
        return lr_max
    return lr_max * 0.1

# Generate learning rates
lr_constant = [constant_with_warmup(s, warmup_steps, lr_max) for s in steps]
lr_polynomial = [polynomial_decay(s, warmup_steps, total_steps, lr_max, power=1) for s in steps]
lr_cosine = [cosine_decay(s, warmup_steps, total_steps, lr_max) for s in steps]
lr_inverse_sqrt = [inverse_sqrt_decay(s, warmup_steps, total_steps, lr_max) for s in steps]
lr_plateau = [reduce_lr_on_plateau(s, warmup_steps, lr_max) for s in steps]

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(steps, lr_constant, label="Constant with Warmup", linestyle="--")
plt.plot(steps, lr_polynomial, label="Polynomial Decay (power=1)")
plt.plot(steps, lr_cosine, label="Cosine Decay")
plt.plot(steps, lr_inverse_sqrt, label="Inverse Sqrt Decay")
plt.plot(steps, lr_plateau, label="Reduce LR on Plateau")
plt.title("Learning Rate Schedulers")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()
plt.show()
