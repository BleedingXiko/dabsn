# Native sparse MoE

`SparseMoEComponent` is an ordinary component, not a branch in `DABSNBlock`.
Its production dispatch is dropless: N routed items with top-k K always produce
exactly N×K assignments. Assignments are stably grouped by expert, executed,
weighted, and scattered back to their original items. There is no implicit
capacity factor or token-dropping fallback.

The built-in fast expert group stores weights as `[expert,input,output]` and
uses grouped matrix multiplication for both ReLU² projections. The reference
backend is deterministic and is the correctness oracle. Forcing the grouped
backend on an unsupported dtype or stride raises an error instead of silently
selecting the reference path.

Every architecture choice is explicit in the provider config:

- router policy (`switch` or `aux_loss_free`);
- top-k and expert count;
- balance coefficient for Switch routing;
- normalization (`none` or `rmsnorm`);
- residual behavior;
- routing granularity;
- built-in grouped ReLU² experts or registered `expert_specs`.

Registered expert specs make attention, a whole H-native transformer, CNN,
SSM/RWKV, another MoE, or future modules portable without DABSN source edits.
Nested configs are bounded before provider code executes.

`ExpertParallelExpertGroup` shards any ordinary expert group evenly across a
process group. Its differentiable all-to-all exchange preserves every
assignment and restores original order. Variable split sizes mean this path
does not claim CUDA-graph compatibility; a static distributed capture path must
be proven separately before such a capability is advertised.
