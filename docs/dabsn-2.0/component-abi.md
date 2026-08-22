# Component ABI

`DABSNGraph` executes a resolved ordered list of components. Provider lookup,
schema migration, and contract validation happen during construction, never in
the forward hot path. Runtime values remain ordinary tensors or PyTrees.

A portable provider defines a stable namespaced key, component ABI version,
configuration schema version, validation, an immutable input/output contract,
a builder, and named schema migrations. Installed providers use the
`dabsn.components.v2` entry-point group. Installed code is indexed without
being imported; execution requires explicit trust or direct registration.

Language-model graphs must begin with a component declaring `world_builder`.
The token embedding feeds only that first DABSN component. Every later component
receives only the preceding world representation, with no token-ID, embedding,
or DABSN-memory side channel.

The framework recognizes distinct native regimes through contracts rather than
architecture names:

- individual-H routing exposes a complete H-world as each routing item;
- H-native components may make H their internal sequence axis;
- structure-native components retain declared temporal, spatial, node, or item
  axes such as `[B,T,H]`, `[B,Y,X,H]`, or `[B,N,H]`.

Modules own their internal reshaping, masks, state, causality, and kernels.
DABSN neither creates a universal attention mask nor converts layouts at a
component boundary.

Training uses `forward_with_terms()` and a fixed `ComponentOutput`. Providers
statically declare auxiliary loss and report names/reductions. Lifecycle hooks
run only after a real optimizer step, excluding accumulation microbatches and
AMP-skipped updates.

Use:

```shell
dabsn component check acme.component --config component.json --trust-provider
```

The trust flag authorizes that installed provider for the process. It does not
grant arbitrary import paths, execute checkpoint metadata, or weaken canonical
configuration limits.
