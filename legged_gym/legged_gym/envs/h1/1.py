num_envs = 6144
n_scan = 132
n_priv = 3+3 +3
n_priv_latent = 4 + 1 + 12 +12
n_proprio = 3 + 2 + 3 + 2 + 36 + 5
history_len = 10
num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv

print(num_observations) 