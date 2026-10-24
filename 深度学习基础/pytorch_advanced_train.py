"""可运行的 PyTorch 进阶训练示例，只使用固定的合成标量回归数据。

单进程 CPU 默认命令：
    python outputs/pytorch_advanced_train.py --output-dir outputs/pytorch_run
同一个六轮计划在第三轮边界停止，再恢复：
    python outputs/pytorch_advanced_train.py --epochs 6 --stop-after-epoch 3 --output-dir outputs/pytorch_split
    python outputs/pytorch_advanced_train.py --epochs 6 --resume outputs/pytorch_split/last.pt --output-dir outputs/pytorch_resume
多进程示例：
    torchrun --standalone --nproc-per-node=2 outputs/pytorch_advanced_train.py --device cuda --precision fp16
    torchrun --standalone --nproc-per-node=2 outputs/pytorch_advanced_train.py --device cpu --precision fp32

CUDA 多进程使用 NCCL，一进程一张卡；Windows 通常没有 NCCL，应使用 CPU/Gloo，
或在具有 NCCL 的环境中训练。CPU 训练只允许 fp32。Windows 的 Gloo/TCPStore
若没有 libuv 支持，可在启动 torchrun 前设置环境变量 USE_LIBUV=0。

DDP 的 DistributedSampler(shuffle=True, drop_last=False) 会补齐样本，使每个 rank
有相同的 microbatch 数、尾批大小和 collective 顺序。样本数不能整除 world_size 时，
补齐会重复部分样本；优化目标包含这些重复项，不能把它说成无偏的唯一训练样本平均。
每组先取至多 K 个 CPU microbatch，再对实际全局样本数 N 求和。DDP 默认平均梯度，
因此各 microbatch 的 sum-MSE 乘 world_size/N，不再除以 K。

训练集全量 MSE 使用原始唯一样本和 fp32 前向，仅诊断拟合情况，不是泛化验证。
这里只支持 epoch 边界恢复：num_workers=0，无预取增广，不保存 .grad，不恢复到
microbatch 中间。总 epochs 决定完整 schedule，stop_after_epoch 不改变这个计划。
调度器由成功 optimizer update 驱动；FP16 overflow 跳步不会推进调度器，也不会补跑
额外 epoch。lr_used 是本次尝试使用的学习率，lr_next 是成功更新后的下一次学习率。
LambdaLR 初始化即应用 u=0，所以有 warmup 时第一次学习率为 base_lr/W，而不是零。

检查点只含 tensor 和原生 Python 类型，用 weights_only=True、map_location='cpu'
加载，绝不自动回退到不受限 pickle。即使如此，也只应加载可信来源的检查点。
恢复顺序为构造模型/优化器/调度器/scaler，再加载模型/调度器/优化器/scaler，最后恢复
逐 rank RNG，且在创建下一轮 DataLoader iterator 之前完成。逐 rank 保存 Python、
CPU、当前 CUDA 设备以及独立 loader generator 的状态。复现要求同一版本、配置与
world_size；即使启用确定性算法，也不保证跨运行环境的精确确定性或 CUDA 位级相同。

明确限制：不支持多机弹性训练、改变 world_size/rank 映射或中途重分片；固定多机拓扑
也未经本示例验证。只保存当前 rank 使用的那张 CUDA 卡的 RNG，不保存其他设备状态。
num_workers=0 是恢复契约的一部分：不保存 worker 随机态、预取队列或随机数据增广；
改成多 worker 后，本例的恢复保证不成立。checkpoint 状态边界是整轮完成、梯度已清空、
scheduler/scaler 已更新之后；不是任意代码行、batch 或累积组中间的快照。
DistributedSampler padding 会重复索引；若照搬到验证集，重复项会污染验证指标的权重，
必须去重或采用不补齐且正确聚合的验证方案。这里没有独立验证集，不能据训练 MSE
宣称泛化能力。SyncBatchNorm 只作为有 BatchNorm 的 CUDA/DDP 模型的可选讨论，
不用于这个无 BN/Dropout 的回归模型；梯度累积也不等价于合并 BN 的批统计。

依赖限制：能 import torch 不代表能构造 AdamW；某些版本会经 Dynamo 导入 sympy 等。
缺依赖时应安装完整依赖后再测，不能模拟 optimizer 或把跳过的测试宣称通过。
NumPy 缺失警告不影响本例不经 NumPy 的基础 tensor 运算，但其他入口仍可能被依赖阻塞。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
import itertools
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


INPUT_DIM = 6
HIDDEN_DIM = 32
MIN_LR_RATIO = 0.1
CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cpu"
    precision: str = "fp32"
    epochs: int = 6
    stop_after_epoch: int | None = None
    resume: str | None = None
    output_dir: str = "outputs/pytorch_advanced_run"
    batch_size: int = 8
    accum_steps: int = 3
    samples: int = 101
    seed: int = 42
    warmup_updates: int = 3
    lr: float = 0.01
    grad_clip: float = 1.0


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


class SyntheticRegressionDataset(Dataset):
    """独立 generator 保证数据构造不消耗训练的全局 RNG。"""

    def __init__(self, samples: int = 101, seed: int = 42) -> None:
        if samples < 1:
            raise ValueError("samples 必须为正整数。")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.features = torch.randn(samples, INPUT_DIM, generator=generator)
        coefficients = torch.tensor([1.5, -2.0, 0.75, 1.0, -0.5, 0.25])
        noise = 0.05 * torch.randn(samples, 1, generator=generator)
        self.targets = self.features @ coefficients[:, None] + 0.4 + noise

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class RegressionMLP(nn.Module):
    """ModuleList 注册子模块，buffer 进入 state_dict 但不是可训练参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(INPUT_DIM, HIDDEN_DIM),
             nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
             nn.Linear(HIDDEN_DIM, 1)]
        )
        self.register_buffer("input_scale", torch.ones(INPUT_DIM))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features / self.input_scale
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))
        return self.layers[-1](hidden)


def iter_microbatch_groups(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]], accum_steps: int
) -> Iterator[list[tuple[torch.Tensor, torch.Tensor]]]:
    if accum_steps < 1:
        raise ValueError("accum_steps 必须为正整数。")
    iterator = iter(loader)
    while True:
        group = list(itertools.islice(iterator, accum_steps))
        if not group:
            return
        yield group


def sample_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    world_size: int,
    global_samples: int,
) -> torch.Tensor:
    return F.mse_loss(prediction, target, reduction="sum") * (
        world_size / global_samples
    )


def lr_factor(
    update_index: int,
    total_updates: int,
    warmup_updates: int,
    min_ratio: float = MIN_LR_RATIO,
) -> float:
    """u 从零开始，表示将进行的第 u 个成功更新位置。

    warmup 使用 (u+1)/W。余下离散位置的首点为 1，末点为 min_ratio；
    如果余弦段只有一个位置，直接使用 min_ratio。计划结束后的值固定在末点。
    """
    if total_updates < 1 or not 0 <= warmup_updates < total_updates:
        raise ValueError("必须满足 total_updates > warmup_updates >= 0。")
    if update_index < 0 or not 0.0 <= min_ratio <= 1.0:
        raise ValueError("更新位置必须非负，min_ratio 必须位于 [0, 1]。")
    u = min(update_index, total_updates - 1)
    if u < warmup_updates:
        return (u + 1) / max(1, warmup_updates)
    remaining = total_updates - warmup_updates
    if remaining == 1:
        return min_ratio
    position = (u - warmup_updates) / max(1, remaining - 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * position)
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_updates: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    lr_factor(0, total_updates, warmup_updates)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda u: lr_factor(u, total_updates, warmup_updates),
    )


def finish_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> dict[str, Any]:
    """真实 overflow 交给 GradScaler 处理，不能在 FP16 跳步前抛梯度非有限异常。"""
    lr_used = float(optimizer.param_groups[0]["lr"])
    scale_before = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=grad_clip, error_if_nonfinite=False
    )
    norm_finite = bool(torch.isfinite(norm).item())
    if not scaler.is_enabled() and not norm_finite:
        raise FloatingPointError("未启用 scaler 时梯度范数非有限，拒绝执行 optimizer.step。")
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    stepped = scale_after >= scale_before
    if stepped:
        if not norm_finite:
            raise FloatingPointError("梯度范数非有限但 scaler 未跳步，终止并拒绝保存异常状态。")
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "stepped": stepped,
        "grad_norm": float(norm.item()) if norm_finite else None,
        "grad_norm_finite": norm_finite,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "lr_used": lr_used,
        "lr_next": float(optimizer.param_groups[0]["lr"]),
    }


def validate_config(config: TrainConfig) -> None:
    for name in ("epochs", "batch_size", "accum_steps", "samples"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} 必须为正整数。")
    if config.device not in ("cpu", "cuda"):
        raise ValueError("device 仅支持 cpu/cuda。")
    if config.precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision 仅支持 fp32/fp16/bf16。")
    if config.device == "cpu" and config.precision != "fp32":
        raise ValueError("本示例的 CPU 训练只支持 fp32。")
    if config.warmup_updates < 0 or not 0 <= config.seed < 2**63:
        raise ValueError("warmup_updates 必须非负，seed 必须位于 [0, 2**63)。")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr 必须是有限正数。")
    if not math.isfinite(config.grad_clip) or config.grad_clip <= 0:
        raise ValueError("grad_clip 必须是有限正数。")
    if config.stop_after_epoch is not None and not (
        1 <= config.stop_after_epoch <= config.epochs
    ):
        raise ValueError("stop_after_epoch 是累计轮数，必须位于 [1, epochs]。")


def setup_distributed(device_name: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("torchrun 的 rank/world_size 环境变量不合法。")
    if dist.is_initialized():
        raise RuntimeError("请在未初始化 process group 的进程中调用 train。")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA。")
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK 超出 CUDA 设备数；必须一进程一卡。")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        backend = "nccl" if device_name == "cuda" else "gloo"
        available = (
            dist.is_nccl_available() if backend == "nccl" else dist.is_gloo_available()
        )
        if not dist.is_available() or not available:
            raise RuntimeError(
                f"当前构建没有 {backend}；Windows 通常没有 NCCL，可改用 CPU/Gloo。"
            )
        dist.init_process_group(
            backend=backend, init_method="env://", timeout=timedelta(seconds=90)
        )
    return DistributedContext(device, rank, local_rank, world_size)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def capture_rng(
    context: DistributedContext, loader_generator: torch.Generator
) -> dict[str, Any]:
    return {
        "rank": context.rank,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(context.device)
            if context.device.type == "cuda" else None
        ),
        "cuda_device_index": context.device.index,
    }


def restore_rng(
    state: dict[str, Any],
    context: DistributedContext,
    loader_generator: torch.Generator,
) -> None:
    if state["rank"] != context.rank or state["cuda_device_index"] != context.device.index:
        raise ValueError("检查点 RNG 的 rank/CUDA 设备不匹配。")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    loader_generator.set_state(state["loader_generator"])
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device=context.device)


def full_training_mse(
    model: nn.Module, dataset: SyntheticRegressionDataset, device: torch.device
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(dataset), 256):
            features = dataset.features[start:start + 256].to(device)
            targets = dataset.targets[start:start + 256].to(device)
            total += F.mse_loss(model(features), targets, reduction="sum").item()
    model.train(was_training)
    result = total / len(dataset)
    if not math.isfinite(result):
        raise FloatingPointError("训练集诊断 MSE 非有限；不保存异常状态。")
    return result


def ensure_finite_loss(loss: torch.Tensor, context: DistributedContext) -> None:
    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
    if context.distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(
            "前向 loss 非有限，所有 rank 在 backward 前终止。请检查数据或降低 lr；"
            "有限 loss 对应的 FP16 梯度 overflow 则由 GradScaler 正常跳步恢复。"
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    *,
    context: DistributedContext,
    loader_generator: torch.Generator,
    config: dict[str, Any],
    next_epoch: int,
    global_update: int,
    global_group: int,
    skipped_updates: int,
    microbatches_seen: int,
) -> None:
    local_rng = capture_rng(context, loader_generator)
    rng_by_rank: list[Any] = [None] * context.world_size
    if context.distributed:
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank[0] = local_rng
    if context.rank == 0:
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "torch_version": str(torch.__version__),
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "next_epoch": next_epoch,
            "global_update": global_update,
            "global_group": global_group,
            "skipped_updates": skipped_updates,
            "microbatches_seen": microbatches_seen,
            "config": config,
            "rng_by_rank": rng_by_rank,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
    if context.distributed:
        dist.barrier()


def validate_checkpoint(
    state: dict[str, Any], expected_config: dict[str, Any]
) -> None:
    if state["format_version"] != CHECKPOINT_FORMAT:
        raise ValueError("检查点格式版本不匹配。")
    if state["torch_version"] != str(torch.__version__):
        raise ValueError("严格恢复要求 PyTorch 版本一致。")
    saved_config = state["config"]
    mismatches = [
        key for key in sorted(set(saved_config) | set(expected_config))
        if saved_config.get(key) != expected_config.get(key)
    ]
    if mismatches:
        raise ValueError("续训配置不一致: " + ", ".join(mismatches))
    next_epoch = state["next_epoch"]
    if not 0 <= next_epoch <= expected_config["epochs"]:
        raise ValueError("检查点 next_epoch 不合法。")
    if state["global_group"] != next_epoch * expected_config["updates_per_epoch"]:
        raise ValueError("只接受完整 epoch 边界的检查点。")
    if state["microbatches_seen"] != next_epoch * expected_config["loader_batches"]:
        raise ValueError("检查点 microbatch 计数与 epoch 边界不一致。")
    if state["global_update"] + state["skipped_updates"] != state["global_group"]:
        raise ValueError("成功更新/跳步/组计数不一致。")
    if state["scheduler"]["last_epoch"] != state["global_update"]:
        raise ValueError("调度器位置与成功更新计数不一致。")
    if len(state["rng_by_rank"]) != expected_config["world_size"]:
        raise ValueError("逐 rank RNG 数量不匹配。")
    if [item["rank"] for item in state["rng_by_rank"]] != list(
        range(expected_config["world_size"])
    ):
        raise ValueError("逐 rank RNG 顺序不匹配。")


def train(
    config: TrainConfig,
    *,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    context = setup_distributed(config.device)
    try:
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 bf16。")
        if context.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        random.seed(config.seed + context.rank)
        torch.manual_seed(config.seed + context.rank)
        dataset = SyntheticRegressionDataset(config.samples, config.seed)
        loader_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + 100_000 + context.rank
        )
        sampler = (
            DistributedSampler(
                dataset, num_replicas=context.world_size, rank=context.rank,
                shuffle=True, seed=config.seed, drop_last=False,
            ) if context.distributed else None
        )
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=sampler is None,
            sampler=sampler, num_workers=0, drop_last=False,
            generator=loader_generator,
        )
        updates_per_epoch = math.ceil(len(loader) / config.accum_steps)
        total_updates = config.epochs * updates_per_epoch
        lr_factor(0, total_updates, config.warmup_updates)
        run_config = {
            key: value for key, value in asdict(config).items()
            if key not in ("resume", "output_dir", "stop_after_epoch")
        }
        run_config.update({
            "world_size": context.world_size,
            "num_workers": 0,
            "num_threads": 1,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "min_lr_ratio": MIN_LR_RATIO,
            "weight_decay": 0.01,
            "loader_batches": len(loader),
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "dataset_version": "fixed_generator_scalar_regression_v1",
            "schedule": "warmup_cosine_discrete_v1",
        })
        if context.distributed:
            rank_configs: list[Any] = [None] * context.world_size
            dist.all_gather_object(rank_configs, run_config)
            if any(item != run_config for item in rank_configs):
                raise ValueError("各 rank 的训练配置必须完全相同。")
        model: nn.Module = RegressionMLP().to(context.device)
        if context.distributed:
            model = DDP(
                model,
                device_ids=[context.local_rank] if config.device == "cuda" else None,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=0.01, foreach=False
        )
        # 必须在加载 optimizer 状态之前构造 scheduler，避免覆盖已恢复的学习率。
        scheduler = build_scheduler(optimizer, total_updates, config.warmup_updates)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=config.device == "cuda" and config.precision == "fp16"
        )
        next_epoch = global_update = global_group = skipped_updates = 0
        microbatches_seen = 0
        if config.resume is not None:
            state = torch.load(config.resume, map_location="cpu", weights_only=True)
            validate_checkpoint(state, run_config)
            unwrap_model(model).load_state_dict(state["model"], strict=True)
            scheduler.load_state_dict(state["scheduler"])
            # 参数已在目标设备；load_state_dict 按参数设备映射 AdamW 的动量状态。
            # 不盲目把所有状态搬到 CUDA：非 capturable AdamW 的 step 张量可留在 CPU。
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            if optimizer.param_groups[0]["lr"] != scheduler.get_last_lr()[0]:
                raise ValueError("optimizer 和 scheduler 保存的学习率不一致。")
            next_epoch = state["next_epoch"]
            global_update = state["global_update"]
            global_group = state["global_group"]
            skipped_updates = state["skipped_updates"]
            microbatches_seen = state["microbatches_seen"]
            # 所有构造和状态加载都已完成；尚未创建任何 DataLoader iterator。
            restore_rng(state["rng_by_rank"][context.rank], context, loader_generator)
        stop_epoch = config.stop_after_epoch or config.epochs
        if stop_epoch < next_epoch:
            raise ValueError("stop_after_epoch 不能早于已恢复的 next_epoch。")
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "last.pt"

        def emit(event: dict[str, Any]) -> None:
            if context.rank == 0:
                if log_fn is None:
                    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
                else:
                    log_fn(event)

        initial_mse = (
            full_training_mse(unwrap_model(model), dataset, context.device)
            if context.rank == 0 else None
        )
        final_mse = initial_mse
        emit({
            "event": "start", "rank": context.rank,
            "world_size": context.world_size, "next_epoch": next_epoch,
            "planned_epochs": config.epochs, "stop_after_epoch": stop_epoch,
            "total_updates": total_updates, "initial_train_mse": initial_mse,
            "lr_next": float(optimizer.param_groups[0]["lr"]),
            "diagnostic": "全量训练集 MSE，不是泛化验证；DDP 补齐样本可能重复。",
        })
        optimizer.zero_grad(set_to_none=True)
        autocast_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
        for epoch in range(next_epoch, stop_epoch):
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
            for group_index, group in enumerate(
                iter_microbatch_groups(loader, config.accum_steps), start=1
            ):
                local_samples = sum(target.shape[0] for _, target in group)
                count = torch.tensor(local_samples, device=context.device, dtype=torch.int64)
                if context.distributed:
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)
                global_samples = int(count.item())
                local_loss_sum = 0.0
                for micro_index, (features, target) in enumerate(group):
                    sync = (
                        model.no_sync()
                        if context.distributed and micro_index < len(group) - 1
                        else nullcontext()
                    )
                    # no_sync 必须同时覆盖前向和反向，而 autocast 仅覆盖前向与 loss。
                    with sync:
                        features = features.to(context.device)
                        target = target.to(context.device)
                        with torch.autocast(
                            device_type=context.device.type, dtype=autocast_dtype,
                            enabled=config.precision != "fp32",
                        ):
                            prediction = model(features)
                            loss = sample_weighted_mse(
                                prediction, target, world_size=context.world_size,
                                global_samples=global_samples,
                            )
                        ensure_finite_loss(loss, context)
                        local_loss_sum += float(loss.detach().item()) * (
                            global_samples / context.world_size
                        )
                        scaler.scale(loss).backward()
                update = finish_update(model, optimizer, scheduler, scaler, config.grad_clip)
                global_group += 1
                microbatches_seen += len(group)
                if update["stepped"]:
                    global_update += 1
                else:
                    skipped_updates += 1
                loss_total = torch.tensor(
                    local_loss_sum, device=context.device, dtype=torch.float64
                )
                if context.distributed:
                    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
                emit({
                    "event": "update", "epoch": epoch + 1,
                    "group_in_epoch": group_index, "global_group": global_group,
                    "global_update": global_update, "skipped_updates": skipped_updates,
                    "microbatches": len(group), "microbatches_seen": microbatches_seen,
                    "local_samples": local_samples, "global_samples": global_samples,
                    "group_mse": float(loss_total.item()) / global_samples,
                    **update,
                })
            next_epoch = epoch + 1
            if context.rank == 0:
                final_mse = full_training_mse(unwrap_model(model), dataset, context.device)
            emit({
                "event": "epoch", "completed_epochs": next_epoch,
                "train_mse": final_mse, "global_update": global_update,
                "global_group": global_group,
            })
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                context=context, loader_generator=loader_generator, config=run_config,
                next_epoch=next_epoch, global_update=global_update,
                global_group=global_group, skipped_updates=skipped_updates,
                microbatches_seen=microbatches_seen,
            )
        summary = {
            "event": "finished", "completed_epochs": next_epoch,
            "global_update": global_update, "global_group": global_group,
            "skipped_updates": skipped_updates, "total_updates": total_updates,
            "initial_train_mse": initial_mse, "final_train_mse": final_mse,
            "checkpoint": str(checkpoint_path),
        }
        emit(summary)
        return summary
    finally:
        if context.distributed and dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="可恢复的合成标量回归训练，默认 CPU/fp32。")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--epochs", type=int, default=6, help="完整训练计划，不因早停而改变。")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="完成该累计轮数后停止。")
    parser.add_argument("--resume", default=None, help="可信的 epoch 边界 last.pt。")
    parser.add_argument("--output-dir", default="outputs/pytorch_advanced_run")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return TrainConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
