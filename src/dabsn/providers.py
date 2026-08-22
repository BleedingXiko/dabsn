"""Auto-registered built-in component providers."""

from __future__ import annotations

from typing import Any, Mapping, cast

import torch

from .components import (
    AxisContract,
    BuildContext,
    ComponentCapabilities,
    ComponentContract,
    ComponentSpec,
    ParallelDirective,
    ParallelExecutionContext,
    ParallelExecutionReceipt,
    ParallelPlan,
    ResultDeclaration,
    ValueContract,
    component_registry,
)
from .core import TensorParallelDABSNCore
from .model import DABSNBlock, MLPRMSNorm, ResidualMLPComponent
from .moe import (
    ROUTER_REPORT_NAMES,
    AuxLossFreeTopKRouter,
    ExpertParallelExpertGroup,
    GenericExpertGroup,
    ReLU2MLPExpertGroup,
    SparseMoEComponent,
    SwitchTopKRouter,
)


def _int(value: object) -> int:
    return int(cast(Any, value))


def _float(value: object) -> float:
    return float(cast(Any, value))


def _world(width: int) -> ValueContract:
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", int(width)),
    )


def _expert_item(width: int) -> ValueContract:
    return ValueContract.tensor(
        AxisContract("dabsn:routed_item", "N", dynamic=True),
        AxisContract("world", int(width)),
    )


def _gather_first_dimension(
    value: torch.Tensor,
    *,
    sizes: tuple[int, ...],
    group,
    device: object,
) -> torch.Tensor:
    source_device = value.device
    working = value.to(device=cast(Any, device))
    maximum = max(sizes)
    if working.shape[0] < maximum:
        padding = list(working.shape)
        padding[0] = maximum - working.shape[0]
        working = torch.cat((working, working.new_zeros(padding)), dim=0)
    gathered = [torch.empty_like(working) for _ in sizes]
    torch.distributed.all_gather(gathered, working.contiguous(), group=group)
    return torch.cat(
        [piece[:width] for piece, width in zip(gathered, sizes)], dim=0
    ).to(source_device)


class DABSNBlockProvider:
    provider_key = "dabsn:block"
    component_abi_version = 2
    config_schema_version = 1
    capabilities = ComponentCapabilities(
        eager=True,
        compile_fullgraph=True,
        activation_checkpoint=True,
        amp_fp32=True,
        amp_bf16=True,
        amp_fp16=True,
        distributed=True,
        world_builder=True,
        dabsn_memory_owner=True,
        deterministic=False,
    )

    def capabilities_for_config(
        self,
        config: Mapping[str, object],
        context: BuildContext,
    ) -> ComponentCapabilities:
        tensor_axis = config.get("tensor_parallel_axis")
        # Whole-graph compilation is not available on CUDA, and saying so is the
        # point of declaring capabilities at all.
        #
        # The CUDA read goes through `admitted_three_way_read_compact_dense_bmm`,
        # a CompositeImplicitAutograd op with no fake kernel, so tracing executes
        # its Python body rather than a meta implementation -- and that body
        # tiles the query-time axis with `range(0, total_steps, chunk)` over
        # tensor shapes. Under tracing those are symbolic, and iterating one
        # raises out of the symbolic-shape machinery ("'int' object has no
        # attribute 'is_Add'"). The CPU read takes a different operator and does
        # compile, which is why this is device-conditional rather than a flat
        # False.
        #
        # Declaring it True anyway would make the conformance matrix assert a
        # capability the component does not have -- the same failure mode the
        # sparse MoE avoids by declaring its dropless dispatch untraceable.
        # Making it true is real work: a fake kernel for that op, which costs the
        # free autograd that CompositeImplicitAutograd currently provides.
        # An unset device means the caller takes the default placement, which is
        # CPU -- the path that does compile.
        device_type = getattr(context.device, "type", None)
        compilable = tensor_axis is None and device_type != "cuda"
        plan = None
        if tensor_axis is not None:
            plan = ParallelPlan(
                fsdp_boundary=False,
                directives=(
                    ParallelDirective(
                        "core.hidden.distribute",
                        "tensor",
                        (str(tensor_axis),),
                        {"dimension": "dabsn_state", "placement": "contiguous"},
                    ),
                ),
            )
        return ComponentCapabilities(
            eager=True,
            compile_fullgraph=compilable,
            activation_checkpoint=True,
            amp_fp32=True,
            amp_bf16=True,
            amp_fp16=True,
            distributed=True,
            world_builder=True,
            dabsn_memory_owner=True,
            deterministic=False,
            parallel_plan=plan,
        )

    def validate_config(self, config: Mapping[str, object]) -> None:
        allowed = {
            "input_dim",
            "hidden_dim",
            "state_dim",
            "read_geometry",
            "residual",
            "tensor_parallel_axis",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown dabsn:block configuration fields: {sorted(unknown)}")
        for name in ("input_dim", "hidden_dim", "state_dim"):
            if _int(config[name]) <= 0:
                raise ValueError(f"{name} must be positive")
        if config["read_geometry"] not in {"seq", "field", "hybrid"}:
            raise ValueError("read_geometry must be seq, field, or hybrid")
        tensor_axis = config.get("tensor_parallel_axis")
        if tensor_axis is not None and (not isinstance(tensor_axis, str) or not tensor_axis):
            raise ValueError("tensor_parallel_axis must be a non-empty axis name")

    def contract(self, config: Mapping[str, object]) -> ComponentContract:
        return ComponentContract(
            _world(_int(config["input_dim"])), _world(_int(config["hidden_dim"]))
        )

    def build(self, config: Mapping[str, object], context: BuildContext):
        return DABSNBlock(
            input_dim=_int(config["input_dim"]),
            hidden_dim=_int(config["hidden_dim"]),
            state_dim=_int(config["state_dim"]),
            read_geometry=str(config["read_geometry"]),
            residual=bool(config.get("residual", False)),
            mlp_ratio=None,
        )

    def parallelize(
        self,
        module: torch.nn.Module,
        config: Mapping[str, object],
        plan: ParallelPlan,
        context: ParallelExecutionContext,
    ) -> ParallelExecutionReceipt:
        del config
        if not isinstance(module, DABSNBlock):
            raise TypeError("dabsn:block parallel executor received the wrong module type")
        if len(plan.directives) != 1 or plan.directives[0].kind != "tensor":
            raise ValueError("dabsn:block supports exactly one tensor directive")
        directive = plan.directives[0]
        if len(directive.mesh_axes) != 1:
            raise ValueError("dabsn:block tensor distribution requires exactly one axis")
        if isinstance(module.core, TensorParallelDABSNCore):
            raise TypeError("DABSN core is already tensor parallelized")
        axis = directive.mesh_axes[0]
        workers = context.topology.axis_size(axis)
        worker = int(context.coordinate[axis])
        module.core = TensorParallelDABSNCore(
            module.core,
            group=context.group(axis),
            rank=worker,
            world_size=workers,
        )
        sharded = {
            "core.beta",
            "core.log_kappa",
            "core.logit_recover",
            "core.k_s",
            "core.k_y",
            "core.k_b",
            "core.k_n",
            "core.k_bias",
            "core.k_saturation",
            "core.r_s",
            "core.r_y",
            "core.r_b",
            "core.r_n",
            "core.r_bias",
            "core.r_saturation",
            "core.W.weight",
            "core.Wg.weight",
            "core.Wg.bias",
            "core.Ug.weight",
            "core.A.weight",
        }
        placements = {
            name: ((axis,) if name in sharded else ())
            for name, _parameter in module.named_parameters()
        }
        return ParallelExecutionReceipt(
            (directive.directive_id,),
            parameter_sharding=placements,
        )

    def consolidate_parallel_state(
        self,
        module: torch.nn.Module,
        config: Mapping[str, object],
        local_state: Mapping[str, torch.Tensor],
        context: ParallelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del config
        if not isinstance(module, DABSNBlock):
            raise TypeError("dabsn:block state consolidator received the wrong module type")
        if not isinstance(module.core, TensorParallelDABSNCore):
            return dict(local_state)
        core = module.core
        sharded_names = {
            "core.beta",
            "core.log_kappa",
            "core.logit_recover",
            "core.k_s",
            "core.k_y",
            "core.k_b",
            "core.k_n",
            "core.k_bias",
            "core.k_saturation",
            "core.r_s",
            "core.r_y",
            "core.r_b",
            "core.r_n",
            "core.r_bias",
            "core.r_saturation",
            "core.W.weight",
            "core.Wg.weight",
            "core.Wg.bias",
            "core.Ug.weight",
            "core.A.weight",
        }
        missing = sharded_names - local_state.keys()
        if missing:
            raise KeyError(f"DABSN tensor shard state is missing {sorted(missing)}")
        result = dict(local_state)
        for name in sorted(sharded_names):
            result[name] = _gather_first_dimension(
                local_state[name],
                sizes=core.tensor_sizes,
                group=core.tensor_group,
                device=context.device,
            )
        return result

    def migrate_config(self, old_version: int, config: Mapping[str, object]):
        if old_version != self.config_schema_version:
            raise ValueError(f"no dabsn:block migration from schema {old_version}")
        return dict(config)


class ResidualMLPProvider:
    provider_key = "dabsn:residual_mlp"
    component_abi_version = 2
    config_schema_version = 1
    capabilities = ComponentCapabilities(
        eager=True,
        compile_fullgraph=True,
        dynamic_shapes=True,
        export=True,
        cuda_graph=True,
        activation_checkpoint=True,
        amp_fp32=True,
        amp_bf16=True,
        amp_fp16=True,
        distributed=True,
        deterministic=True,
    )

    def validate_config(self, config: Mapping[str, object]) -> None:
        unknown = set(config) - {"dim", "ratio"}
        if unknown:
            raise ValueError(f"unknown dabsn:residual_mlp configuration fields: {sorted(unknown)}")
        if _int(config["dim"]) <= 0:
            raise ValueError("dim must be positive")
        if _float(config["ratio"]) <= 0:
            raise ValueError("ratio must be positive")

    def contract(self, config: Mapping[str, object]) -> ComponentContract:
        contract = _world(_int(config["dim"]))
        return ComponentContract(contract, contract)

    def build(self, config: Mapping[str, object], context: BuildContext):
        return ResidualMLPComponent(_int(config["dim"]), _float(config["ratio"]))

    def migrate_config(self, old_version: int, config: Mapping[str, object]):
        if old_version != self.config_schema_version:
            raise ValueError(f"no dabsn:residual_mlp migration from schema {old_version}")
        return dict(config)


class SparseMoEProvider:
    provider_key = "dabsn:sparse_moe"
    component_abi_version = 2
    config_schema_version = 1
    capabilities = ComponentCapabilities(
        eager=True,
        amp_fp32=True,
        amp_bf16=True,
        amp_fp16=True,
        distributed=True,
        deterministic=False,
    )
    loss_terms = (ResultDeclaration("router_balance", "mean", "framework"),)
    reports = tuple(ResultDeclaration(name, "none", "framework") for name in ROUTER_REPORT_NAMES)

    def capabilities_for_config(
        self,
        config: Mapping[str, object],
        context: BuildContext,
    ):
        device = getattr(context.device, "type", context.device)
        grouped = (
            config.get("backend", "auto") == "grouped"
            and device == "cuda"
            and context.dtype == torch.bfloat16
            and config.get("expert_specs") is None
        )
        expert_axis = config.get("expert_parallel_axis")
        plan = None
        if expert_axis is not None:
            plan = ParallelPlan(
                fsdp_boundary=False,
                directives=(
                    ParallelDirective(
                        "experts.distribute",
                        "expert",
                        (str(expert_axis),),
                        {"placement": "contiguous"},
                    ),
                ),
            )
        return ComponentCapabilities(
            eager=True,
            compile_fullgraph=grouped,
            dynamic_shapes=grouped,
            export=False,
            cuda_graph=grouped,
            activation_checkpoint=grouped,
            amp_fp32=True,
            amp_bf16=True,
            amp_fp16=True,
            distributed=True,
            deterministic=False,
            parallel_plan=plan,
        )

    def validate_config(self, config: Mapping[str, object]) -> None:
        allowed = {
            "hidden_dim",
            "experts",
            "top_k",
            "inner_dim",
            "router",
            "balance_coefficient",
            "normalization",
            "routing_granularity",
            "backend",
            "zero_output",
            "residual",
            "bias_update_rate",
            "expert_specs",
            "expert_parallel_axis",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown dabsn:sparse_moe configuration fields: {sorted(unknown)}")
        for key in ("hidden_dim", "experts", "top_k"):
            if _int(config[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        expert_specs = config.get("expert_specs")
        if expert_specs is None:
            if _int(config["inner_dim"]) <= 0:
                raise ValueError("inner_dim must be positive")
        else:
            if "inner_dim" in config:
                raise ValueError("inner_dim and expert_specs are mutually exclusive")
            if "backend" in config or "zero_output" in config:
                raise ValueError("backend and zero_output apply only to built-in ReLU2 experts")
            if not isinstance(expert_specs, list) or len(expert_specs) != _int(config["experts"]):
                raise ValueError("expert_specs must contain exactly one spec per expert")
            for index, raw in enumerate(expert_specs):
                if not isinstance(raw, dict):
                    raise ValueError(f"expert_specs[{index}] must be an object")
                required = {
                    "provider_key",
                    "provider_distribution",
                    "provider_version",
                    "component_abi_version",
                    "config_schema_version",
                    "config",
                }
                missing = required - raw.keys()
                if missing:
                    raise ValueError(f"expert_specs[{index}] is missing {sorted(missing)}")
        if _int(config["top_k"]) > _int(config["experts"]):
            raise ValueError("top_k cannot exceed experts")
        if config.get("router", "switch") not in {"switch", "aux_loss_free"}:
            raise ValueError("router must be switch or aux_loss_free")
        if config.get("normalization", "none") not in {"none", "rmsnorm"}:
            raise ValueError("normalization must be none or rmsnorm")
        if config.get("router", "switch") == "switch":
            if "balance_coefficient" not in config:
                raise ValueError("switch routing requires explicit balance_coefficient")
            if _float(config["balance_coefficient"]) < 0:
                raise ValueError("balance_coefficient must be non-negative")
            if "bias_update_rate" in config:
                raise ValueError("bias_update_rate is valid only for aux_loss_free routing")
        elif "balance_coefficient" in config:
            raise ValueError("balance_coefficient is valid only for switch routing")
        if config.get("routing_granularity", "individual_h") != "individual_h":
            raise ValueError("built-in sparse MoE routing granularity is individual_h")
        expert_axis = config.get("expert_parallel_axis")
        if expert_axis is not None and (not isinstance(expert_axis, str) or not expert_axis):
            raise ValueError("expert_parallel_axis must be a non-empty axis name")

    def contract(self, config: Mapping[str, object]) -> ComponentContract:
        world = _world(_int(config["hidden_dim"]))
        return ComponentContract(world, world)

    def build(self, config: Mapping[str, object], context: BuildContext):
        hidden = _int(config["hidden_dim"])
        experts = _int(config["experts"])
        top_k = _int(config["top_k"])
        policy = str(config.get("router", "switch"))
        router: SwitchTopKRouter | AuxLossFreeTopKRouter
        if policy == "switch":
            router = SwitchTopKRouter(
                hidden,
                experts,
                top_k,
                balance_coefficient=_float(config["balance_coefficient"]),
            )
        else:
            router = AuxLossFreeTopKRouter(
                hidden,
                experts,
                top_k,
                bias_update_rate=_float(config.get("bias_update_rate", 1.0e-3)),
            )
        raw_expert_specs = config.get("expert_specs")
        group: ReLU2MLPExpertGroup | GenericExpertGroup
        if raw_expert_specs is None:
            group = ReLU2MLPExpertGroup(
                experts,
                hidden,
                _int(config["inner_dim"]),
                zero_output=bool(config.get("zero_output", False)),
                backend=str(config.get("backend", "auto")),
            )
        else:
            modules = []
            expected = _expert_item(hidden)
            expert_specs = cast(list[dict[str, object]], raw_expert_specs)
            for index, raw in enumerate(expert_specs):
                spec = ComponentSpec(
                    component_id=f"expert.{index}",
                    provider_key=str(raw["provider_key"]),
                    config=dict(cast(Mapping[str, object], raw["config"])),
                    component_abi_version=_int(raw["component_abi_version"]),
                    config_schema_version=_int(raw["config_schema_version"]),
                    provider_distribution=str(raw["provider_distribution"]),
                    provider_version=str(raw["provider_version"]),
                )
                binding = component_registry.build(spec, context)
                input_errors = expected.incompatibilities(binding.contract.input)
                output_errors = expected.incompatibilities(binding.contract.output)
                if input_errors or output_errors:
                    raise ValueError(
                        f"expert {index} contract must preserve routed [N,H] worlds; "
                        f"input={input_errors} output={output_errors}"
                    )
                modules.append(binding.module)
            group = GenericExpertGroup(modules)
        return SparseMoEComponent(
            hidden,
            router,
            group,
            residual=bool(config.get("residual", False)),
            normalization=(
                MLPRMSNorm(hidden) if config.get("normalization", "none") == "rmsnorm" else None
            ),
            routing_granularity="individual_h",
        )

    @staticmethod
    def _local_expert_group(
        group: GenericExpertGroup | ReLU2MLPExpertGroup,
        *,
        start: int,
        stop: int,
    ) -> GenericExpertGroup | ReLU2MLPExpertGroup:
        local_experts = stop - start
        if isinstance(group, ReLU2MLPExpertGroup):
            local = ReLU2MLPExpertGroup(
                local_experts,
                group.hidden_dim,
                group.inner_dim,
                accumulation_dtype=group.accumulation_dtype,
                backend=group.backend,
            ).to(device=group.w1.device, dtype=group.w1.dtype)
            local.w1 = torch.nn.Parameter(group.w1.detach()[start:stop].clone())
            local.w2 = torch.nn.Parameter(group.w2.detach()[start:stop].clone())
            return local
        return GenericExpertGroup(tuple(group.experts[start:stop]))

    def parallelize(
        self,
        module: torch.nn.Module,
        config: Mapping[str, object],
        plan: ParallelPlan,
        context: ParallelExecutionContext,
    ) -> ParallelExecutionReceipt:
        if not isinstance(module, SparseMoEComponent):
            raise TypeError("dabsn:sparse_moe parallel executor received the wrong module type")
        if len(plan.directives) != 1 or plan.directives[0].kind != "expert":
            raise ValueError("dabsn:sparse_moe supports exactly one expert directive")
        directive = plan.directives[0]
        if len(directive.mesh_axes) != 1:
            raise ValueError("dabsn:sparse_moe expert distribution requires exactly one axis")
        axis = directive.mesh_axes[0]
        workers = context.topology.axis_size(axis)
        worker = int(context.coordinate[axis])
        if module.experts % workers:
            raise ValueError(
                f"{module.experts} experts cannot be divided evenly across "
                f"{workers} workers on axis {axis!r}"
            )
        local_count = module.experts // workers
        start = worker * local_count
        stop = start + local_count
        if not isinstance(module.expert_group, (GenericExpertGroup, ReLU2MLPExpertGroup)):
            raise TypeError("sparse mixture-of-experts component is already parallelized")
        local_group = self._local_expert_group(module.expert_group, start=start, stop=stop)
        process_group = context.group(axis)
        module.expert_group = ExpertParallelExpertGroup(
            local_group,
            process_group=process_group,
            world_size=workers,
            rank=worker,
        )

        if isinstance(module.router, AuxLossFreeTopKRouter):
            module.router.process_group = context.groups.get("data")
        handles = []
        if workers > 1 and torch.distributed.is_initialized():

            def average_gradient(gradient, *, group=process_group, divisor=workers):
                synchronized = gradient.clone()
                torch.distributed.all_reduce(synchronized, group=group)
                return synchronized.div_(divisor)

            expert_parameter_ids = {id(parameter) for parameter in module.expert_group.parameters()}
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                if id(parameter) in expert_parameter_ids:
                    handles.append(
                        parameter.register_hook(
                            lambda gradient, divisor=workers: gradient.div(divisor)
                        )
                    )
                else:
                    handles.append(parameter.register_hook(average_gradient))
        module._expert_parallel_gradient_hooks = handles
        expert_parameter_ids = {id(parameter) for parameter in module.expert_group.parameters()}
        placements = {
            name: ((axis,) if id(parameter) in expert_parameter_ids else ())
            for name, parameter in module.named_parameters()
        }
        return ParallelExecutionReceipt(
            (directive.directive_id,),
            parameter_sharding=placements,
        )

    def consolidate_parallel_state(
        self,
        module: torch.nn.Module,
        config: Mapping[str, object],
        local_state: Mapping[str, torch.Tensor],
        context: ParallelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(module, SparseMoEComponent):
            raise TypeError("dabsn:sparse_moe consolidator received the wrong module type")
        group = module.expert_group
        if not isinstance(group, ExpertParallelExpertGroup):
            return dict(local_state)
        axis_value = config.get("expert_parallel_axis")
        if not isinstance(axis_value, str) or not axis_value:
            raise ValueError("parallel sparse mixture-of-experts config has no expert axis")
        axis = axis_value
        workers = context.topology.axis_size(axis)
        worker = int(context.coordinate[axis])
        local_count = module.experts // workers
        start = worker * local_count
        local_prefix = "expert_group.local_group."
        result = {
            name: tensor
            for name, tensor in local_state.items()
            if not name.startswith(local_prefix)
        }
        process_group = context.group(axis)
        if isinstance(group.local_group, ReLU2MLPExpertGroup):
            sizes = (local_count,) * workers
            for parameter_name in ("w1", "w2"):
                local_name = local_prefix + parameter_name
                if local_name not in local_state:
                    raise KeyError(f"expert shard state is missing {local_name!r}")
                result["expert_group." + parameter_name] = _gather_first_dimension(
                    local_state[local_name],
                    sizes=sizes,
                    group=process_group,
                    device=context.device,
                )
            return result

        if not isinstance(group.local_group, GenericExpertGroup):
            raise TypeError("unknown local expert group type during portable consolidation")
        local_tensors: dict[str, torch.Tensor] = {}
        expert_prefix = local_prefix + "experts."
        for name, tensor in local_state.items():
            if not name.startswith(expert_prefix):
                continue
            suffix = name[len(expert_prefix) :]
            local_index_text, separator, remainder = suffix.partition(".")
            if not separator:
                raise ValueError(f"invalid local expert state name {name!r}")
            local_index = int(local_index_text)
            global_name = f"expert_group.experts.{start + local_index}.{remainder}"
            local_tensors[global_name] = tensor
        local_metadata = [
            (name, tuple(tensor.shape), tensor.dtype)
            for name, tensor in sorted(local_tensors.items())
        ]
        gathered_metadata: list[object | None] = [None for _ in range(workers)]
        torch.distributed.all_gather_object(
            gathered_metadata,
            local_metadata,
            group=process_group,
        )
        global_ranks = context.group_ranks[axis]
        for owner, raw_metadata in enumerate(gathered_metadata):
            if not isinstance(raw_metadata, list):
                raise RuntimeError("expert state metadata collective returned an invalid value")
            for raw_item in raw_metadata:
                if (
                    not isinstance(raw_item, tuple)
                    or len(raw_item) != 3
                    or not isinstance(raw_item[0], str)
                    or not isinstance(raw_item[1], tuple)
                    or not isinstance(raw_item[2], torch.dtype)
                ):
                    raise RuntimeError("expert state metadata entry is invalid")
                name, shape, dtype = raw_item
                if owner == worker:
                    tensor = local_tensors[name].to(device=cast(Any, context.device))
                else:
                    tensor = torch.empty(shape, dtype=dtype, device=cast(Any, context.device))
                torch.distributed.broadcast(
                    tensor,
                    src=global_ranks[owner],
                    group=cast(Any, process_group),
                )
                result[name] = tensor.cpu()
        return result

    def migrate_config(self, old_version: int, config: Mapping[str, object]):
        if old_version != self.config_schema_version:
            raise ValueError(f"no dabsn:sparse_moe migration from schema {old_version}")
        return dict(config)


def register_builtin_components() -> None:
    for provider in (
        DABSNBlockProvider(),
        ResidualMLPProvider(),
        SparseMoEProvider(),
    ):
        if provider.provider_key not in component_registry.keys():
            component_registry.register(
                provider,
                trusted=True,
                distribution="dabsn",
                version="2.0.0",
            )


__all__ = [
    "DABSNBlockProvider",
    "ResidualMLPProvider",
    "SparseMoEProvider",
    "register_builtin_components",
]
