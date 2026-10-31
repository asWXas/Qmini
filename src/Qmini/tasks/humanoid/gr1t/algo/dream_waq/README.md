# GR1T DreamWaQ

This implements the single-phase DreamWaQ variant without AdaBoot on RSL-RL 5.5.1. The actor uses five proprioceptive frames, predicts body-frame linear velocity and a 16-dimensional context, then combines those estimates with the latest frame to produce actions. The critic keeps the task's privileged observation group. Policy history must be flattened in term-major order; mixed precision is not supported.

Each PPO minibatch uses one optimizer step for the clipped policy loss, value loss, entropy, velocity regression, next-frame reconstruction, and VAE KL loss. Gradients from PPO and the auxiliary losses reach the same encoder. The action path uses the context mean so rollout and PPO replay have consistent action probabilities; the decoder samples context only for reconstruction.

Isaac Lab returns the **reset observation** for an environment that ended during `step()`. The algorithm stores the next frame immediately for each transition and excludes every `done`, including timeouts, from all three auxiliary losses (velocity, reconstruction, and VAE KL). For `obs[t] -> step -> reset_obs[t+1]`, `dones[t]` excludes the pre-reset sample `obs[t]`; the mask is not shifted to `t-1`. The first nonterminal transition of the new episode remains valid. PPO policy and value losses still use terminal transitions to learn termination rewards. Velocity targets come from the pre-step critic observation. No next-frame target is read from the following rollout, so the last transition is handled correctly. An all-terminal minibatch produces zero auxiliary losses while PPO continues updating.

Train:

```bash
uv run isaaclab train --rl_library rsl_rl --task Qmini-Flat-DreamWaQ
```

Play with `Qmini-Flat-DreamWaQ-Play`. The policy's JIT/ONNX export uses only the policy observation group; it does not require critic observations or the decoder at inference time.

The DreamWaQ task mirrors policy and critic observations, actions, next-frame targets, and velocity labels together for PPO data augmentation. AdaBoot is not included in this variant.
