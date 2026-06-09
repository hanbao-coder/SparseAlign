"""剪枝算法: Wanda, L1/L2 范数, 和 SparseGPT."""

import math
import time
import torch
import torch.nn as nn
import gc


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ActivationCollector:
    """收集激活值的 L2 范数，用于 Wanda 重要性计算。"""

    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()


class HessianTracker:
    """Hessian 矩阵追踪器，用于 SparseGPT 剪枝。"""

    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone().float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)

            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def find_linear_layers(module, name=''):
    """递归查找模块中所有 nn.Linear 层。"""
    if isinstance(module, nn.Linear):
        return {name: module}
    result = {}
    for child_name, child in module.named_children():
        result.update(find_linear_layers(child, name=name + '.' + child_name if name else child_name))
    return result


class SparsityConstraint:
    """存储稀疏 mask 并应用到权重/梯度上。"""

    def __init__(self, layer):
        self.weight_masks = {}
        self._build_masks(layer)

    def _build_masks(self, module):
        for name, m in module.named_modules():
            if isinstance(m, nn.Linear):
                mask = (m.weight.data == 0)
                self.weight_masks[name] = mask

    def apply(self, tensor, name=None):
        """将张量投影到稀疏约束上。"""
        if name is not None and name in self.weight_masks:
            return tensor * (~self.weight_masks[name]).to(tensor.device)
        return tensor

    def count_pruned(self):
        """已剪枝元素总数。"""
        return sum(mask.sum().item() for mask in self.weight_masks.values())

    def count_total(self):
        """元素总数。"""
        return sum(mask.numel() for mask in self.weight_masks.values())

    def apply_mask_to_gradients(self, module):
        """将剪枝位置的梯度清零。"""
        for name, m in module.named_modules():
            if isinstance(m, nn.Linear) and name in self.weight_masks:
                if m.weight.grad is not None:
                    mask = self.weight_masks[name].to(m.weight.grad.device)
                    m.weight.grad.masked_fill_(mask, 0.0)


@torch.no_grad()
def compress_wanda(layer, device, args, input_states, output_states, attention_mask, position_ids):
    """Wanda 剪枝: 基于 |W| * sqrt(激活范数) 的重要性排序."""
    prune_m = getattr(args, 'prune_m', 0)
    prune_n = getattr(args, 'prune_n', 0)
    subset = find_linear_layers(layer)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)

    wrapped = {}
    for name in subset:
        wrapped[name] = ActivationCollector(subset[name])

    def add_batch(collector_name):
        def tmp(_, inp, out):
            wrapped[collector_name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in wrapped:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    # 第一次前向: 收集激活统计
    for j in range(128):
        current_inp = input_states[j].unsqueeze(0).to(device)
        out = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]
        output_states[j] = out.cpu()
        del current_inp, out
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # 应用 Wanda 剪枝 mask
    for name in subset:
        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped[name].scaler_row.reshape((1, -1)))
        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

        if prune_n != 0:
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        else:
            sorted_indices = torch.sort(W_metric, dim=-1, stable=True)[1]
            W_mask.scatter_(1, sorted_indices[:, :int(W_metric.shape[1] * args.sparsity_ratio)], True)

        subset[name].weight.data[W_mask] = 0

    # 第二次前向: 验证剪枝
    for j in range(128):
        current_inp = input_states[j].unsqueeze(0).to(device)
        out = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]
        output_states[j] = out.cpu()
        del current_inp, out
        torch.cuda.empty_cache()

    for name in wrapped:
        wrapped[name].free()

    input_states, output_states = output_states, input_states

    if attention_mask is not None:
        del attention_mask
    if position_ids is not None:
        del position_ids

    return layer, input_states, output_states, SparsityConstraint(layer)


@torch.no_grad()
def compress_sparsegpt(layer, device, args, input_states, output_states, attention_mask, position_ids):
    """SparseGPT 剪枝: 基于 Hessian 的二阶重要性排序."""
    prune_m = getattr(args, 'prune_m', 0)
    prune_n = getattr(args, 'prune_n', 0)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)

    subset = find_linear_layers(layer)

    hessians = {}
    for name in subset:
        hessians[name] = HessianTracker(subset[name])

    def add_batch(tracker_name):
        def tmp(_, inp, out):
            hessians[tracker_name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in hessians:
        handles.append(subset[name].register_forward_hook(add_batch(name)))

    # 前向: 收集 Hessian 统计
    for j in range(128):
        current_inp = input_states[j].unsqueeze(0).to(device)
        out = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]
        output_states[j] = out.cpu()
        del current_inp, out
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # 应用 SparseGPT 剪枝
    for name in hessians:
        hessians[name].fasterprune(
            args.sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            percdamp=0.01,
            blocksize=128
        )
        hessians[name].free()

    # 验证前向
    for j in range(128):
        current_inp = input_states[j].unsqueeze(0).to(device)
        out = layer(current_inp, attention_mask=attention_mask, position_ids=position_ids)[0]
        output_states[j] = out.cpu()
        del current_inp, out
        torch.cuda.empty_cache()

    input_states, output_states = output_states, input_states

    if attention_mask is not None:
        del attention_mask
    if position_ids is not None:
        del position_ids

    gc.collect()
    torch.cuda.empty_cache()

    return layer, input_states, output_states, SparsityConstraint(layer)


@torch.no_grad()
def compress_magnitude(layer, device, args, input_states, output_states,
                       attention_mask, position_ids, norm="l1_norm"):
    """基于 L1/L2 范数的幅值剪枝: 直接按权重绝对值排序，剪掉最小的。

    相比 Wanda 更简单，不需要激活统计。
    - l1_norm: 按 |W| 排序
    - l2_norm: 按 W^2 排序
    """
    prune_m = getattr(args, 'prune_m', 0)
    prune_n = getattr(args, 'prune_n', 0)
    subset = find_linear_layers(layer)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)

    # 计算原始输出
    for j in range(128):
        current_inp = input_states[j].unsqueeze(0).to(device)
        out = layer(current_inp, attention_mask=attention_mask,
                    position_ids=position_ids)[0]
        output_states[j] = out.cpu()
        del current_inp, out
        torch.cuda.empty_cache()

    # 幅值剪枝
    for name, m in subset.items():
        weight = m.weight.data.float()
        if norm == "l2_norm":
            importance = weight ** 2
        else:  # l1_norm
            importance = torch.abs(weight)

        mask = torch.zeros_like(importance, dtype=torch.bool)

        if prune_n != 0:
            # N:M 稀疏
            for ii in range(importance.shape[1]):
                if ii % prune_m == 0:
                    tmp = importance[:, ii:(ii + prune_m)]
                    mask.scatter_(
                        1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )
        else:
            sorted_indices = torch.sort(importance, dim=-1, stable=True)[1]
            k = int(importance.shape[1] * args.sparsity_ratio)
            mask.scatter_(1, sorted_indices[:, :k], True)

        m.weight.data[mask] = 0

    # 验证剪枝后输出
    for j in range(128):
        current_inp = input_states[j].unsqueeze(0).to(device)
        out = layer(current_inp, attention_mask=attention_mask,
                    position_ids=position_ids)[0]
        output_states[j] = out.cpu()
        del current_inp, out
        torch.cuda.empty_cache()

    input_states, output_states = output_states, input_states

    if attention_mask is not None:
        del attention_mask
    if position_ids is not None:
        del position_ids

    return layer, input_states, output_states, SparsityConstraint(layer)
