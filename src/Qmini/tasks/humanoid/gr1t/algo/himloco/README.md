# GR1T HIMLoco

HIMLoco uses five proprioceptive frames to estimate body velocity and a normalized 16-dimensional latent response. A target encoder maps the next policy frame into the same prototype space. The HIO phase minimizes current body-frame velocity error and a swapped prototype-assignment loss with Sinkhorn balancing. The following PPO phase freezes the estimator and updates only the policy head and privileged critic.

The simulator auto-resets completed environments before returning observations. Successor targets are captured in the same step as the action and excluded from both HIO losses on terminal and time-limit transitions. For `obs[t] -> step -> reset_obs[t+1]`, `dones[t]` excludes the pre-reset sample `obs[t]`; the mask is not shifted to `t-1`. The first nonterminal transition after reset remains valid. PPO keeps terminal transitions to learn termination rewards. Left-right augmentation mirrors policy history, critic input, action, successor frame, and velocity target together.

The implementation uses RSL-RL 5.5.1's symmetry extension and PPO update, including observation-normalization updates. Policy history must be flattened in term-major order. If no worker has valid HIO samples, the estimator optimizer step is skipped so Adam momentum cannot change its parameters. Workers with no local samples still participate in gradient reduction when other workers have valid samples. The estimator optimizer state is included in checkpoints.

```bash
uv run isaaclab train --rl_library rsl_rl --task Qmini-Flat-HIMLoco
```

For playback use `Qmini-Flat-HIMLoco-Play`. JIT and ONNX exports only need policy observations at inference time.
