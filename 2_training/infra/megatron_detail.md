Megatron分享

# Megatron-LM SFT与知识蒸馏训练流程详解
本文档详细讲解基于 Megatron-LM Core v0.14.0rc7 的 SFT（监督微调）和知识蒸馏训练的完整流程，从启动脚本到核心训练循环，涵盖所有关键代码路径。

---

## 目录
1. [整体架构概览](#1-整体架构概览)
2. [完整调用链路图](#2-完整调用链路图)
3. [启动流程：run.sh](#3-启动流程runsh)
4. [配置解析：config.env](#4-配置解析configenv)
5. [训练脚本：run_mcore_deepseek.sh](#5-训练脚本run_mcore_deepseeksh)
6. [training_deepseek.py - Python训练入口](#6-training_deepseekpy---python训练入口)
7. [megatron_patch/training.py - 核心训练循环](#7-megatron_patchtrainingpy---核心训练循环)
8. [megatron_patch/helper/helper.py - 辅助函数](#8-megatron_patchhelperhelperpy---辅助函数)
9. [megatron_patch/model/lightning_model.py - 模型定义](#9-megatron_patchmodellightning_modelpy---模型定义)
10. [已知问题：MoE Aux Loss Scaling Bug](#10-已知问题moe-aux-loss-scaling-bug)

---

## 1. 整体架构概览
Megatron-LM 是一个用于大规模语言模型训练的框架，支持多种并行策略（TP/PP/CP/EP/SP）和先进特性（Flash Attention、激活检查点、MTP等）。本项目在其基础上实现了：

* **SFT（监督微调）**：支持多轮对话格式，带 Packing 优化
* **知识蒸馏**：从教师模型获取 Top-K logits 进行蒸馏
* **MTP（Multi-Token Prediction）**：多 token 预测辅助任务

**核心代码结构：**

```
baidu/ps/SearchLighting/
├── run.sh                           # 主启动脚本
├── config
│   └── config.env                   # 配置文件
├── projects/deepseek/
│   ├── run_mcore_deepseek.sh       # DeepSeek 模型训练脚本
│   └── training_deepseek.py        # 训练入口
├── megatron_patch/                 # 自定义补丁代码
│   ├── training.py                 # 覆盖 Megatron 的 pretrain() 函数
│   ├── helper/helper.py            # 核心辅助函数（forward_step, loss_func）
│   ├── model/lightning_model.py    # 自定义模型（支持蒸馏和 MTP）
│   ├── data/                       # 数据处理
│   │   └── real_json_sft_packing.py # Packing SFT 数据集
│   └── teacher/client.py           # 教师模型客户端
└── Megatron-LM-core_v0.14.0rc7/    # Megatron-LM 核心库
    └── megatron/
        ├── training/training.py    # 原生训练循环
        └── core/                   # 核心组件
```
---

## 2. 完整调用链路图
### 2.1 整体流程概览
```
run.sh
  └─> run_mcore_deepseek.sh
       └─> torchrun training_deepseek.py
            └─> pretrain()  [megatron_patch/training.py]
                 ├─> initialize_megatron()
                 ├─> setup_model_and_optimizer()
                 │    ├─> model_provider()
                 │    │    └─> LightningModel(...) [megatron_patch/model/lightning_model.py]
                 │    │         └─> wrap_with_ddp(model)  [distributed_data_parallel.py]
                 │    └─> get_megatron_optimizer()
                 ├─> build_train_valid_test_data_iterators()
                 │    └─> train_valid_test_datasets_provider()
                 │         └─> REALJSONPACKINGSFTDataset(...)
                 └─> train()  [megatron/training/training.py]
                      └─> [for iteration in range(train_iters)]
                           ├─> train_step()  [详见 2.2]
                           └─> training_log()  [详见 2.4]
```
### 2.2 单个训练步骤详解（train_step）
![](1f1b.png)
**源码位置**: `megatron/training/training.py:train_step()`

```
train_step()
 │
 ├─> [1] 前向+反向传播
 │    └─> forward_backward_func()  
 │         └─> forward_backward_pipelining_without_interleaving()  [schedules.py]
 │              │
 │              ├─> [Warmup Phase] 前向传播预热
 │              │    └─> for i in range(num_warmup_microbatches):
 │              │         └─> forward_step()  [详见 2.2.1]
 │              │
 │              ├─> [Steady State] 1F1B 阶段
 │              │    └─> for i in range(num_microbatches_remaining):
 │              │         ├─> forward_step()  [详见 2.2.1]
 │              │         └─> backward_step()  [详见 2.2.2]
 │              │
 │              ├─> [Cooldown Phase] 反向传播收尾
 │              │    └─> for i in range(num_warmup_microbatches):
 │              │         └─> backward_step()  [详见 2.2.2]
 │              │
 │              └─> [Finalize] 梯度最终化
 │                   └─> finalize_model_grads()  [详见 2.3]
 │
 ├─> [2] 梯度裁剪
 │    └─> clip_grad_norm_fp32()
 │
 ├─> [3] 参数更新
 │    ├─> optimizer.step()
 │    └─> optimizer.zero_grad()
 │
 └─> [4] 学习率调度
      └─> opt_param_scheduler.step()
```
### 2.2.1 前向传播详解（forward_step）
**源码位置**: `megatron/core/pipeline_parallel/schedules.py:forward_step()`

```
forward_step()
 │
 ├─> forward_step_func()  [megatron_patch/helper/helper.py:forward_step]
 │    │
 │    ├─> [数据获取] get_batch_with_teacher_knowledge()
 │    │    │
 │    │    ├─> get_batch_base()
 │    │    │    └─> next(data_iterator)
 │    │    │         └─> REALJSONPACKINGSFTDataset.__getitem__()
 │    │    │              └─> 返回: (tokens, labels, sequence_order, ...)
 │    │    │
 │    │    ├─> [蒸馏] TeacherClient.submit()  [异步提交]
 │    │    │    └─> ZeroMQ 发送请求到教师服务
 │    │    │
 │    │    └─> [蒸馏] broadcast_teacher_knowledge()
 │    │         └─> torch.distributed.broadcast()  [同步等待教师响应]
 │    │              └─> 返回: teacher_topk_logps, teacher_topk_indices
 │    │
 │    ├─> [模型前向] model.forward()  [LightningModel]
 │    │    │
 │    │    ├─> embedding_forward()
 │    │    │    └─> word_embeddings + position_embeddings
 │    │    │
 │    │    ├─> decoder_forward()  [Transformer layers]
 │    │    │    └─> for layer in layers:
 │    │    │         ├─> attention()  [MLA/MQA]
 │    │    │         └─> mlp() 或 moe_layer()
 │    │    │              └─> [MoE] router.forward()
 │    │    │                   ├─> compute routing scores
 │    │    │                   ├─> _apply_aux_loss()  [计算 aux loss]
 │    │    │                   │    └─> MoEAuxLossAutoScaler.apply(activation, aux_loss)
 │    │    │                   │         └─> [Trick] 保存 aux_loss 到 context
 │    │    │                   └─> token_dispatcher.dispatch()
 │    │    │
 │    │    ├─> [MTP] mtp_forward()  [if enabled]
 │    │    │    ├─> mtp_decoder_forward()
 │    │    │    └─> compute_language_model_loss()
 │    │    │         └─> MTPLossCalculator.__call__()
 │    │    │              ├─> cross_entropy_loss * loss_mask  [只计算有效token]
 │    │    │              └─> MTPLossAutoScaler.apply(output, mtp_loss)
 │    │    │                   └─> [Trick] 保存 mtp_loss 到 context
 │    │    │
 │    │    ├─> output_layer_forward()
 │    │    │    └─> linear projection to vocab size
 │    │    │
 │    │    └─> [Loss计算] fused_distill_vocab_parallel_cros_entropy_v2()
 │    │         ├─> student_logits → cross_entropy(labels) → lm_loss
 │    │         └─> [蒸馏] kl_divergence(student_logits, teacher_logits) → distill_loss
 │    │
 │    └─> loss_func()  [megatron_patch/helper/helper.py]
 │         │
 │         ├─> 提取损失: lm_loss, distill_loss, mtp_loss (from model output)
 │         ├─> 计算 reduced_loss (使用 loss_mask)
 │         ├─> [归一化] 按 sequence 数量归一化
 │         │    └─> lm_loss_avg = sum(lm_loss / valid_tokens_per_seq)
 │         └─> 返回: (total_loss, turn_num, loss_dict)
 │
 └─> [Loss缩放] forward_step_calc_loss()  [schedules.py:230-237]
      │
      ├─> if calculate_per_token_loss == False:
      │    ├─> output_tensor /= num_tokens        [除以本 microbatch 的 token 数]
      │    └─> output_tensor /= num_microbatches  [除以 microbatch 总数]
      │
      └─> [MoE/MTP Loss Scale] 设置 AutoScaler 的 scale
           ├─> if calculate_per_token_loss:
           │    ├─> MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
           │    └─> MTPLossAutoScaler.set_loss_scale(loss_scale)
           └─> else:
                ├─> MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)
                └─> MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)
```
### 2.2.2 反向传播详解（backward_step）
**源码位置**: `megatron/core/pipeline_parallel/schedules.py:backward_step()`

```
backward_step()
 │
 ├─> [准备阶段]
 │    ├─> input_tensor.retain_grad()  [保留输入梯度]
 │    └─> if output_tensor_grad is None:  [最后一个 PP stage]
 │         └─> output_tensor = grad_scale_func(output_tensor)  [应用 FP16 loss scale]
 │
 ├─> [反向传播触发]
 │    └─> torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
 │         │
 │         ├─> [主 Loss 梯度] ∂(lm_loss + distill_loss) / ∂(params)
 │         │    └─> 梯度累积到 param.main_grad 或 grad_buffer
 │         │
 │         ├─> [MoE Aux Loss 梯度] MoEAuxLossAutoScaler.backward()
 │         │    │   [源码: moe/moe_utils.py:185-202]
 │         │    │
 │         │    ├─> scaled_aux_loss_grad = ones_like(aux_loss) * main_loss_backward_scale
 │         │    │    └─> main_loss_backward_scale 由 set_loss_scale() 设置
 │         │    │         ├─> calculate_per_token_loss=True:  loss_scale
 │         │    │         └─> calculate_per_token_loss=False: loss_scale / num_microbatches
 │         │    │
 │         │    └─> torch.autograd.backward(aux_loss, grad_tensors=scaled_aux_loss_grad)
 │         │         └─> ∂(aux_loss) / ∂(router_params) 累积到 grad_buffer
 │         │
 │         └─> [MTP Loss 梯度] MTPLossAutoScaler.backward()
 │              │   [源码: multi_token_prediction.py:539-552]
 │              │
 │              ├─> scaled_mtp_loss_grad = ones_like(mtp_loss) * main_loss_backward_scale
 │              │    └─> main_loss_backward_scale 由 set_loss_scale() 设置
 │              │
 │              └─> torch.autograd.backward(mtp_loss, grad_tensors=scaled_mtp_loss_grad)
 │                   └─> ∂(mtp_loss) / ∂(decoder_params) 累积到 grad_buffer
 │
 ├─> [梯度预缩放] DDP 的 start_grad_sync()
 │    │   [源码: distributed_data_parallel.py:566-576]
 │    │
 │    └─> for bucket in buckets:
 │         └─> if gradient_scaling_factor != 1.0:
 │              └─> bucket.grad_data *= gradient_scaling_factor
 │                   └─> calculate_per_token_loss=False: 
 │                        └─> grad *= (1.0 / dp_size)  [预缩放]
 │
 ├─> [梯度通信] DDP 的 all-reduce
 │    │
 │    ├─> torch.distributed.all_reduce(grad_data, group=dp_group, op=SUM)
 │    │    └─> 跨 DP ranks 求和梯度
 │    │
 │    └─> [Expert Parallel] all-reduce on expert_parallel_group
 │         └─> 专家参数在 EP group 内 all-reduce
 │
 └─> [返回] input_tensor_grad
      └─> 传递给前一个 PP stage 作为 output_tensor_grad
```
### 2.3 梯度最终化详解（finalize_model_grads）
**源码位置**: `megatron/core/distributed/finalize_model_grads.py:finalize_model_grads()`

**调用时机**: 所有 microbatches 的前向+反向传播完成后，optimizer.step() 之前

```
finalize_model_grads(model, num_tokens, grad_finalize_pgs)
 │
 ├─> [1] Embedding 梯度 all-reduce (跨 PP stages)
 │    │   [源码: finalize_model_grads.py:444-450]
 │    │
 │    ├─> _allreduce_word_embedding_grads()
 │    │    └─> torch.distributed.all_reduce(word_emb_grad, group=embd_group)
 │    │         └─> PP 第一个 stage 和最后一个 stage 共享 embedding
 │    │
 │    └─> _allreduce_position_embedding_grads()
 │         └─> torch.distributed.all_reduce(pos_emb_grad, group=pos_emb_group)
 │
 ├─> [2] LayerNorm 梯度 all-reduce (用于 Sequence Parallelism)
 │    │   [源码: finalize_model_grads.py:298-367]
 │    │
 │    └─> _allreduce_non_tensor_model_parallel_grads()
 │         └─> for param in layernorm_params:
 │              └─> torch.distributed.all_reduce(param.grad, group=tp_group)
 │
 ├─> [3] 条件 Embedding 梯度 all-reduce (用于 Diffusion 模型)
 │    │   [源码: finalize_model_grads.py:89-129]
 │    │
 │    └─> _allreduce_conditional_embedding_grads()
 │         └─> torch.distributed.all_reduce(cond_emb_grad, group=pp_group)
 │
 ├─> [4] MoE Router Expert Bias 更新
 │    │   [源码: finalize_model_grads.py:270-295]
 │    │
 │    └─> if config.moe_router_enable_expert_bias:
 │         └─> _update_router_expert_bias()
 │              ├─> torch.distributed.all_reduce(local_tokens_per_expert)
 │              └─> expert_bias = f(tokens_per_expert, old_bias)
 │
 └─> [5] Per-Token Loss 归一化 (关键!)
      │   [源码: finalize_model_grads.py:458-474]
      │
      └─> if num_tokens is not None:  [calculate_per_token_loss=True]
           │
           ├─> [Step 1] Broadcast num_tokens from PP last stage
           │    └─> torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)
           │         └─> 将 total_num_tokens 从最后一个 PP stage 广播到所有 stages
           │              └─> 只有最后 stage 累积了所有 microbatches 的 token 数
           │
           ├─> [Step 2] All-reduce num_tokens across DP ranks
           │    └─> torch.distributed.all_reduce(num_tokens, group=dp_cp_group, op=SUM)
           │         └─> 跨所有 DP replicas 求和，得到全局总 token 数
           │              └─> Example: DP=2 时，8192 + 8192 = 16384
           │
           └─> [Step 3] 统一缩放所有梯度
                └─> for model_chunk in model:
                     └─> scaling = 1.0 / num_tokens  [1.0 / 全局总token数]
                          └─> model_chunk.scale_gradients(scaling)
                               │   [源码: distributed_data_parallel.py:590-593]
                               │
                               └─> for buffer in buffers + expert_parallel_buffers:
                                    └─> buffer.grad_data *= scaling
                                         │
                                         └─> ⭐ 所有梯度统一缩放，包括:
                                              ├─> 主 Loss 的梯度
                                              ├─> MoE Aux Loss 的梯度
                                              └─>MTP Loss 的梯度
```
**梯度缩放总结**:

|场景|Loss 预缩放|DDP 预缩放|finalize 缩放|最终等效|
|-|-|-|-|-|
|`calculate_per_token_loss=False`|✓ (÷tokens ÷mbs)|✓ (÷dp_size)|✗|1/(tokens×mbs×dp)|
|`calculate_per_token_loss=True`|✗|✗|✓ (÷global_tokens)|1/global_tokens|

### 2.4 日志记录详解（training_log）
**源码位置**: `megatron/training/training.py:training_log()`

**调用时机**: 每个 iteration 完成后，每 `log_interval` 步记录一次

```
training_log(loss_dict, total_loss_dict, learning_rate, iteration, ...)
 │
 ├─> [1] 累积损失统计
 │    │   [源码: training.py:1345-1369]
 │    │
 │    ├─> total_loss_dict['advanced iterations'] += 1  [成功的 iteration]
 │    ├─> total_loss_dict['skipped iterations'] += skipped_iter  [跳过的 iteration]
 │    ├─> total_loss_dict['nan iterations'] += int(got_nan)  [NaN 的 iteration]
 │    └─> for key in loss_dict:
 │         └─> total_loss_dict[key] += loss_dict[key]
 │              └─> 累积: lm loss, distill loss, load_balancing_loss 等
 │
 ├─> [2] 计时器日志
 │    │   [源码: training.py:1372-1403]
 │    │
 │    └─> for timer_name in ['forward-backward', 'forward-compute', ...]:
 │         ├─> elapsed = timers(timer_name).elapsed()
 │         └─> log_string += f'{timer_name}: {elapsed:.2f}ms |'
 │
 ├─> [3] 梯度范数日志
 │    │   [源码: training.py:1406-1425]
 │    │
 │    ├─> if grad_norm is not None:
 │    │    └─> log_string += f'grad norm: {grad_norm:.3f} |'
 │    └─> if num_zeros_in_grad is not None:
 │         └─> log_string += f'num zeros: {num_zeros_in_grad:.1f} |'
 │
 ├─> [4] 学习率日志
 │    │   [源码: training.py:1426-1442]
 │    │
 │    ├─> log_string += f'learning rate: {learning_rate:.6E} |'
 │    ├─> if writer:  [TensorBoard]
 │    │    └─> writer.add_scalar('learning-rate', learning_rate, iteration)
 │    └─> if wandb_writer:  [W&B]
 │         └─> wandb_writer.log({'learning-rate': learning_rate}, iteration)
 │
 ├─> [5] Loss Scale 日志 (FP16/BF16)
 │    │   [源码: training.py:1443-1464]
 │    │
 │    └─> if loss_scale is not None:
 │         ├─> log_string += f'loss scale: {loss_scale:.1f} |'
 │         └─> if writer:
 │              └─> writer.add_scalar('loss-scale', loss_scale, iteration)
 │
 ├─> [6] 每 log_interval 步的详细日志
 │    │   [源码: training.py:1501-1654]
 │    │
 │    └─> if iteration % args.log_interval == 0:
 │         │
 │         ├─> [计算吞吐量]
 │         │    ├─> elapsed_time = timers('interval-time').elapsed()
 │         │    ├─> throughput = FLOP / (elapsed_time * world_size)  [TFLOP/s/GPU]
 │         │    └─> log_string += f'throughput: {throughput:.1f} TFLOP/s/GPU |'
 │         │
 │         ├─> [打印训练进度]
 │         │    └─> log_string = f'iteration {iteration}/{train_iters} | '
 │         │         f'consumed samples: {consumed_train_samples} | '
 │         │         f'elapsed time: {elapsed_time_per_iteration:.1f}ms |'
 │         │
 │         ├─> [平均损失]
 │         │    └─> for key in total_loss_dict:
 │         │         └─> avg_loss = total_loss_dict[key] / log_interval
 │         │              └─> log_string += f'{key}: {avg_loss:.6E} |'
 │         │
 │         ├─> [写入 TensorBoard]
 │         │    └─> if writer:
 │         │         ├─> writer.add_scalar('lm loss', avg_lm_loss, iteration)
 │         │         ├─> writer.add_scalar('distill loss', avg_distill_loss, iteration)
 │         │         ├─> writer.add_scalar('learning-rate', learning_rate, iteration)
 │         │         ├─> writer.add_scalar('throughput', throughput, iteration)
 │         │         └─> writer.add_scalar('grad-norm', grad_norm, iteration)
 │         │
 │         ├─> [写入 W&B]
 │         │    └─> if wandb_writer:
 │         │         └─> wandb_writer.log({
 │         │                  'lm loss': avg_lm_loss,
 │         │                  'distill loss': avg_distill_loss,
 │         │                  'learning-rate': learning_rate,
 │         │                  'throughput': throughput,
 │         │                  'grad-norm': grad_norm,
 │         │              }, iteration)
 │         │
 │         ├─> [MoE 指标日志]
 │         │    │   [源码: moe/moe_utils.py:754-813]
 │         │    │
 │         │    └─> if config.num_moe_experts:
 │         │         └─> track_moe_metrics()
 │         │              ├─> reduce_aux_losses_tracker_across_ranks()
 │         │              │    └─> all-reduce aux loss across TP/CP/DP groups
 │         │              └─> for aux_loss_name in ['load_balancing_loss', ...]:
 │         │                   ├─> avg_aux_loss = total_aux_loss / num_layers
 │         │                   └─> writer.add_scalar(aux_loss_name, avg_aux_loss, iteration)
 │         │
 │         ├─> [内存统计]
 │         │    └─> if report_memory_flag:
 │         │         ├─> torch.cuda.max_memory_allocated()
 │         │         └─> log_string += f'max memory: {max_mem / 1e9:.2f} GB |'
 │         │
 │         ├─> [能耗监控]
 │         │    └─> if args.log_energy:
 │         │         └─> energy = energy_monitor.get_energy()
 │         │              └─> log_string += f'energy: {energy:.2f} J |'
 │         │
 │         └─> [打印日志]
 │              └─> print_rank_0(log_string)
 │                   └─> 只在 rank 0 打印
 │
 └─> [7] 重置累积计数器
      └─> total_loss_dict = {}  [每 log_interval 步重置]
```
---

## 3. 启动流程：run.sh
**文件位置：** `baidu/ps/SearchLighting/run.sh`

### 3.1 功能说明
`run.sh` 是训练的总入口，负责：

1. 环境初始化（单机/多机 MPI 配置）
2. 生成数据配置文件
3. 调用数据准备脚本
4. 应用代码补丁
5. 启动真正的训练脚本

### 3.2 关键代码片段
```
#!/bin/bash
# 用于启动分布式深度学习训练任务，支持单机和多机模式
set -x  # 开启命令执行跟踪，便于调试
queue_h_flag=$1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source ./config.env

# 判断运行模式：独立模式 vs 分布式模式
if [[ ${IS_STANDALONE:-1} -eq 1 ]]; then
    # 独立模式：单机运行，不使用 mpirun
```
### 3.3 执行流程
**步骤 1：加载配置**

```
source ./config.env
```
从 `config.env` 加载所有训练配置（模型大小、并行策略、学习率等）。

**步骤 2：生成数据配置 JSON**

```
${mpirun} cat > "./domain_config/data_config.json" << EOF
{
  "training": [
    {
      "name": "50w", 
      "file_path": "${TRAIN_DATA_PATH}"
    }
  ],
  "validation": [
    {
      "name": "vaildation", 
      "val_file_path": "${VALID_DATA_PATH}"
    }
  ]
}
EOF
```
**步骤 3：数据准备**

```
${mpirun} sh prepare_data.sh  ${DATA_CONFIG[@]}
```
**步骤 4：应用补丁**

```
${mpirun} cp "patch_code/utils.py" "/usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention/dot_product_attention/utils.py"
```
修复 Transformer Engine 的注意力机制代码。

**步骤 5：启动训练**

```
${mpirun} sh ./projects/deepseek/run_mcore_deepseek.sh  \
    ${BASE_CONFIG[@]} \
    ${PARALLE_AND_BOOL_OPTION[@]} \
    ${OTHERS[@]}
```
将所有配置参数传递给 DeepSeek 训练脚本。

---

## 4. 配置解析：config.env
**文件位置：** `baidu/ps/SearchLighting/config.env`

### 4.1 关键配置项
```
# 运行环境: "dsw" or "dlc"
ENVIRONMENT="dsw"

# 模型规模
MODEL_SIZE="V100B"

# 单个GPU的 Micro Batch Size
MICRO_BATCH_SIZE=1

# 全局 Global Batch Size (跨所有GPU)
GLOBAL_BATCH_SIZE=16

# 学习率
MAX_LR=5e-6
MIN_LR=1e-6

# 序列长度
SEQ_LENGTH=12288
PADDING_LENGTH=12288

# 混合精度: "bf16" or "fp16"
PRECISION="bf16"


# ----------------------------------------------------------------
# 4. 并行策略与布尔选项 (Parallelism & Boolean Options)
# (来自 PARALLE_AND_BOOL_OPTION 数组)
# ----------------------------------------------------------------
# 并行策略
TP=1 # 张量并行 (Tensor Parallelism)
PP=6 # 流水线并行 (Pipeline Parallelism)
CP=1 # 上下文并行 (Context Parallelism)
EP=8 # 专家并行 (Expert Parallelism, for MoE)

# 布尔选项
SP=true # 序列并行 (Sequence Parallelism)
DO=true # 分布式优化器 (Distributed Optimizer)
USE_FLASH_ATTENTION=true # (FL: true = Flash Attn, false = Fused Attn)
IS_SFT=true # (SFT: Supervised Fine-Tuning mode)
```
### 4.2 蒸馏相关配置
```
# --- 知识蒸馏 (Knowledge Distillation) ---
KD_MODE="real" # (real/fake/none)
TEACHER_IP="10.192.188.79"
TEACHER_PORT=15555

# --- Loss 权重 ---
LM_LOSS_WEIGHT=0
DISTILL_LOSS_WEIGHT=1
MTP_LOSS_WEIGHT=0
MTP_DISTILL_LOSS_WEIGHT=0.1
```
* `KD_MODE="real"`：使用真实教师模型
* `LM_LOSS_WEIGHT=0`：不使用语言模型损失（纯蒸馏模式）
* `DISTILL_LOSS_WEIGHT=1`：蒸馏损失权重

---

## 5. 训练脚本：run_mcore_deepseek.sh
**文件位置：** `baidu/ps/SearchLighting/projects/deepseek/run_mcore_deepseek.sh`

### 5.1 功能说明
该脚本负责：

1. 设置分布式训练环境变量（MASTER_ADDR、NCCL 等）
2. 解析命令行参数
3. 根据模型大小配置模型超参数
4. 构建完整的训练命令
5. 启动 `torchrun`

### 5.2 关键代码片段
**分布式环境配置：**

```
if [ $ENV = dsw ]; then
    MASTER_ADDR=$POD_0_IP
    MASTER_PORT=6001
    NNODES=${PADDLE_TRAINERS_NUM}
    NODE_RANK=${PADDLE_TRAINER_ID}
    GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
```
**模型配置（DeepSeek V3 100B）：**

```
elif [ $MODEL_SIZE = V100B ]; then
    MP_PP0_LAYERS=11
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=128
    NUM_LAYERS=61
    INTERMEDIATE_SIZE=9216
    MOE_INTERMEDIATE_SIZE=1024
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
    EXTRA_VOCAB_SIZE=467
    PADDED_VOCAB_SIZE=128000
    Q_LORA_RANK=1536
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    NUM_SHARED_EXPERTS=1
```
**蒸馏配置：**

```
if [ $KD != none ]; then
    training_options="$training_options \
        --teacher-ip ${KD_IP} \
        --teacher-port ${KD_PORT}"
fi
if [ $KD = real ]; then
    training_options="$training_options \
        --use-distillation \
        --teacher-type real"
elif [ $KD = fake ]; then
    training_options="$training_options \
        --use-distillation \
        --teacher-type fake"
fi

training_weight_options=" \
        --lm-loss-weight ${LM_LOSS_WEIGHT} \
        --distill-loss-weight ${KD_LOSS_WEIGHT} \
        --mtp-loss-weight ${MTP_LOSS_WEIGHT} \
        --mtp-distill-loss-weight ${MTP_KD_LOSS_WEIGHT}"
```
**数据集配置：**

```
elif [ ${MP_DATASET_TYPE} = "real-raw-packing" ]; then
    dataset_option=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset REAL-JSON-SFT-PACKING"
```
**最终启动命令：**

```
run_cmd="torchrun $DISTRIBUTED_ARGS projects/deepseek/training_deepseek.py
 ${megatron_options} ${training_options} ${training_weight_options} ${dataset_option} ${pr_options} ${load_options} ${load_options_extra} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} ${packing_options} ${uneven_split_option} \
 ${attn_backend_option} ${eval_options} "

ps aux | grep "training_deepseek.py" | awk -F ' ' '{print $2}' | xargs kill 2>/dev/null || echo "Killing running processes of training_deepseek.py"
ps aux | grep "upload_ckpts.py" | awk -F ' ' '{print $2}' | xargs kill 2>/dev/null || echo "Killing running processes of upload_ckpts.py"
ps aux | grep "fake_utilization.py" | awk -F ' ' '{print $2}' | xargs kill 2>/dev/null || echo "Killing running processes of fake_utilization.py"

CKPT_AFS_PATH="output_path/Exp_L2_SFT/model_save/new_exp/${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-$(date +%Y%m%d%H%M)"
setsid nohup python tools/upload_ckpts.py $SAVED_PRETRAIN_CHECKPOINT_PATH $CKPT_AFS_PATH $NODE_RANK 2 &

echo ${run_cmd}
eval ${run_cmd}
```
---

## 6. training_deepseek.py - Python训练入口
**文件位置：** `baidu/ps/SearchLighting/projects/deepseek/training_deepseek.py`

### 6.1 文件说明
这是 Python 训练代码的入口文件，包含：

1. **主函数** - 启动训练流程
2. **model_provider()** - 构建模型
3. **train_valid_test_datasets_provider()** - 构建数据集

### 6.2 主函数
```
if __name__ == "__main__":
    # 在 Megatron 初始化之前，替换掉原生的参数验证函数
    from megatron.training import initialize
    initialize.validate_args = validate_args_patched

    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
```
**关键点：**

* **Line 235**：替换 Megatron 原生的参数验证函数，使用自定义的 `validate_args_patched`
* **Line 242-247**：调用 `pretrain()` 函数，传入：
    * `train_valid_test_datasets_provider`：数据集构建函数
    * `model_provider`：模型构建函数
    * `forward_step`：前向传播函数（来自 `megatron_patch.helper`）


### 6.3 model_provider() - 模型构建
`model_provider()` 负责构建模型实例，返回一个 `LightningModel` 对象（继承自 `GPTModel`）。

```
def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel]:

    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(args, DeepSeekV2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.num_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config, use_transformer_engine=use_te, 
                normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, 
                vp_stage=vp_stage
            )
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                    normalization=args.normalization)

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
            transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
        else:
            transformer_layer_spec_for_mtp = transformer_layer_spec
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
        )
    
    model = LightningModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )
    
    return model
```
**关键点：**

* **Line 92**：构建 Transformer 配置（`DeepSeekV2TransformerConfig`）
* **Line 116-123**：如果启用 MTP，构建 MTP block spec
* **Line 125-143**：返回 `LightningModel` 实例

### 6.4 train_valid_test_datasets_provider() - 数据集构建
根据 `args.dataset` 类型构建训练、验证、测试数据集。

```
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    
    if args.dataset == "PRETRAIN-WITH-WEIGHT":
        print_rank_0("> Building weighted BlendedDataset...")
        train_ds, valid_ds, test_ds = build_pretrain_weighted_datasets(args)
        print_rank_0("> finished creating weighted BlendedDataset datasets ...")
    elif args.dataset == 'JSON-SFT':
        train_ds = JSONSFTDataset(args.train_data_path, args.max_padding_length)
        valid_ds = JSONSFTDataset(args.valid_data_path, args.max_padding_length)
        test_ds = JSONSFTDataset(args.valid_data_path, args.max_padding_length)
    elif args.dataset == 'JSON-SFT-PACKING':
        train_ds = JSONPACKINGSFTDataset(args.train_data_path, args.max_padding_length)
        valid_ds = JSONPACKINGSFTDataset(args.valid_data_path, args.max_padding_length)
        test_ds = JSONPACKINGSFTDataset(args.valid_data_path, args.max_padding_length)
    elif args.dataset == 'REAL-JSON-SFT-PACKING':
        train_ds = REALJSONPACKINGSFTDataset(args.train_data_path, args.max_padding_length)
        valid_ds = REALJSONPACKINGSFTDataset(args.valid_data_path, args.max_padding_length)
        test_ds = REALJSONPACKINGSFTDataset(args.valid_data_path, args.max_padding_length)
    elif args.dataset == 'JSON-PREFER':
        train_ds = JSONPreferDataset(args.train_data_path, args.seq_length)
        if args.valid_data_path:
            valid_ds = JSONPreferDataset(args.valid_data_path, args.seq_length)
        else:
            valid_ds = None
        test_ds = None
    elif args.dataset == 'MMAP':
        if args.sft:
            dataset_type = SFTDataset
        else:
            if args.mock_data:
                dataset_type = MockGPTDataset
            else:
                dataset_type = GPTDataset

        print_rank_0("> building train, validation, and test datasets for GPT ...")
        config = core_gpt_dataset_config_from_args(args)
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is no longer supported")
    return train_ds, valid_ds, test_ds
```
**本项目使用的数据集：** `REAL-JSON-SFT-PACKING`，对应 `REALJSONPACKINGSFTDataset` 类（详见第10节）。

---

## 7. megatron_patch/training.py - 核心训练循环
**文件位置：** `baidu/ps/SearchLighting/megatron_patch/training.py`

### 7.1 功能说明
`pretrain()` 是整个训练的核心入口，负责：

1. 初始化 Megatron（分布式环境、参数解析）
2. 构建模型、优化器、学习率调度器
3. 加载 checkpoint
4. 构建数据集和数据迭代器
5. 调用 `train()` 函数进入训练循环

### 7.2 函数签名
```
def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper: Optional[CallWrapper] = None,
):
```
### 7.2 pretrain() - 训练入口
**功能:** 是整个训练的核心入口。

**关键步骤:**

**步骤 1：初始化 Megatron**

```
initialize_megatron(
    extra_args_provider=extra_args_provider,
    args_defaults=args_defaults,
    get_embedding_ranks=get_embedding_ranks,
    get_position_embedding_ranks=get_position_embedding_ranks,
    store=store,
)
```
初始化分布式环境、解析命令行参数、设置随机种子。

**步骤 2：构建模型和优化器**

```
model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    model_provider, model_type, teacher=False
)
```
**步骤 3：加载 checkpoint**

```
if args.load:
    timers('load-checkpoint', log_level=0).start(barrier=True)
    args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler, ...)
    timers('load-checkpoint').stop(barrier=True)
```
**步骤 4：构建数据集**

```
train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
    train_valid_test_dataset_provider
)
```
**步骤 5：进入训练循环**

```
if not args.skip_train:
    print_rank_0('training ...')

    if args.dataloader_type == 'cyclic' and args.retro_project_dir:
        assert args.retro_cyclic_train_iters is not None
        args.train_iters = args.retro_cyclic_train_iters
        print_rank_0("retro cyclic train iters : %d" % args.train_iters)

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration, num_floating_point_operations_so_far = train(
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            process_non_loss_data_func,
            config,
            checkpointing_context,
            non_loss_data_func,
        )
```
### 7.3 train_step() - 单步训练
**文件位置：** `baidu/ps/SearchLighting/megatron_patch/training.py`

`train_step()` 执行单次训练迭代：

1. 清空梯度
2. 调用 `forward_backward_func()` 进行前向和反向传播
3. 更新优化器

**关键代码：**

```
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        if has_nvidia_modelopt:
            # [ModelOpt]: Pipeline-parallel Distillation stacks student and teacher tensors
            adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
                model, args.seq_length, args.micro_batch_size, args.decoder_seq_length
            )
        else:
            adjust_tensor_shapes_fn = None

        # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
        # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
        # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            for optim_instance in optimizer.chained_optimizers:
                if isinstance(optim_instance, DistributedOptimizer):
                    optim_instance._copy_main_params_to_param_buffer()

        # Forward pass.
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )
```
`forward_backward_func()`** 来自：**

```
forward_backward_func = get_forward_backward_func()
```
这是 Megatron Core 的流水线并行实现，会根据 PP stage 自动处理前向/反向传播的调度。

---

## 8. megatron_patch/helper/helper.py - 辅助函数
**文件位置：** `baidu/ps/SearchLighting/megatron_patch/helper/helper.py`

### 8.1 文件说明
这个文件包含训练过程中的核心辅助函数：

1. **forward_step()** - 前向传播与蒸馏
2. **get_batch_with_teacher_knowledge()** - 获取教师知识
3. **loss_func()** - 损失计算

### 8.2 forward_step() - 前向传播
**功能:** 每个训练 step 的前向传播函数。

**文件位置：** `baidu/ps/SearchLighting/megatron_patch/helper/helper.py`

`forward_step()` 是每个训练 step 的前向传播函数，负责：

1. 从 data iterator 获取 batch 数据
2. 如果启用蒸馏，调用 `get_batch_with_teacher_knowledge()` 获取教师知识
3. 调用模型的 `forward()` 方法
4. 返回 output 和 loss 函数

```
def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel ): The GPT Model
    """
    timers = get_timers()
    args = get_args()
    # print(type(data_iterator))
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    teacher_topk_logps = None
    teach_topk_indices = None
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1 and args.dataset == 'REAL-JSON-SFT-PACKING':
        assert args.micro_batch_size == 1, "当前场景下只支持bs = 1, 否则可能会出错"
    if args.use_distillation:
        start_time = time.time()
        # debug_all_rank()
        (tokens, labels, loss_mask, attention_mask, position_ids, num_seqs,
         packed_seq_params, teacher_topk_logps, teach_topk_indices, turn_num, loss_token_num) = get_batch_with_teacher_knowledge(
            data_iterator,not model.training)
        if teacher_topk_logps is not None:
            teacher_topk_logps = teacher_topk_logps.view(args.micro_batch_size, args.seq_length // cp_size, 256)
        if teach_topk_indices is not None:
            teach_topk_indices = teach_topk_indices.view(args.micro_batch_size, args.seq_length // cp_size, 256)
        end_time = time.time()
        
        if mpu.is_pipeline_last_stage() and end_time - start_time >= 0.01:
            print("[WARN] [RANK: {}] get_batch_with_teacher_knowledge耗时: {:.2f}秒".format(torch.distributed.get_rank(), end_time - start_time))
    else:
        # debug_all_rank()
        tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, turn_num, loss_token_num  = get_batch_base(
            data_iterator)
        # position_ids = None
        # print(attention_mask)
    timers("batch-generator").stop()
    # debug_all_rank()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params,loss_token_num=loss_token_num,
                          teacher_topk_logits=teacher_topk_logps,
                          teacher_topk_indices=teach_topk_indices)
    
    del teacher_topk_logps, teach_topk_indices

    # return output_tensor, partial(loss_func, loss_mask, num_seqs)
    return output_tensor, partial(loss_func, loss_mask, turn_num, loss_token_num)
```
**关键点：**

* **Line 697-710**：如果启用蒸馏（`args.use_distillation`），调用 `get_batch_with_teacher_knowledge()` 获取教师的 top-k logits
* **Line 718-720**：调用模型的 `forward()` 方法，传入 `teacher_topk_logits` 和 `teacher_topk_indices`
* **Line 725**：返回 loss 函数（`partial(loss_func, ...)`）

---

### 8.3 get_batch_with_teacher_knowledge() - 获取教师知识
**功能:** 蒸馏的核心函数，负责从教师模型获取 top-k logits。

**文件位置：** `baidu/ps/SearchLighting/megatron_patch/helper/helper.py`

这是蒸馏的核心函数，负责：

1. 初始化教师模型客户端（`TeacherClient`）
2. 预填充 batch buffer（异步请求教师模型）
3. 从 buffer 中取出 batch 和对应的教师知识
4. 广播教师知识到所有 TP ranks

**初始化教师客户端：**

```
def get_batch_with_teacher_knowledge(data_iterator,is_eval):
    # torch.cuda.empty_cache()
    global teacher_client
    args = get_args()
    # Get the batch.
    if is_teacher_client_rank() and teacher_client is None:
        if args.teacher_type == "fake":
            teacher_client = PlaceholderTeacherClient()
        elif args.teacher_type == "real":
            teacher_client = TeacherClient(server_ip=args.teacher_ip, server_port=args.teacher_port,
                                num_microbatches=1, n_server_workers=8, temperature=1)
        else:
            assert False, "蒸馏时必须指定教师"
    batch_buffer = train_batch_buffer
```
**预填充 batch buffer：**

```
n_prefills = 1 if batch_buffer else 4 * get_num_microbatches()
for _ in range(n_prefills):
    # ! 注意如果是json sft 格式，这里的position id 实际上返回的是 sequence_order
    assert args.dataset == 'REAL-JSON-SFT-PACKING', "当前代码只在这个情况下可以正常运行， get_batch 接口被改了"
    # debug_all_rank()
    tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, turn_num, loss_token_num = get_batch_base(data_iterator)
    # debug_all_rank()
    # if torch.distributed.get_rank() == 0:
    #     import ipdb; ipdb.set_trace()
    # decompose tokens into sub sequences using order info
    if is_teacher_client_rank():
        # debug_all_rank()
        # tokens.shape = [1, 4096]
        if args.dataset == 'REAL-JSON-SFT-PACKING':
            sequence_order = position_ids # 用它实际的内容避免混乱
        
            seqs_to_submit = []
            assert tokens.shape[0] == 1, "only consider batch_size = 1"
            teacher_knowledge = []
            # assert teacher_client.num_microbatches == 1
            # debug_all_rank()
            for seq_tokens, sequence_order in zip(tokens, position_ids): # 这里position id实际上存的是seq order
                sub_seq_tokens = extract_sequences_tensor(seq_tokens, sequence_order, padding=True)
                # debug_all_rank()
                for t in sub_seq_tokens:
                    sub_teacher_knowledge = teacher_client.submit([t])
                    teacher_knowledge.append(sub_teacher_knowledge)
```
**关键逻辑：**

1. **Line 434-438**：对于 `REAL-JSON-SFT-PACKING` 数据集，`position_ids` 实际存储的是 `sequence_order`（packing 信息）
2. **Line 445-450**：使用 `extract_sequences_tensor()` 将 packed tokens 分解为多个子序列
3. **Line 449**：为每个子序列提交异步请求到教师模型

**重建教师知识：**

```
tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, loss_token_num, turn_num, teacher_knowledge = batch_buffer.popleft()
if teacher_knowledge:
    # rebuild tokens
    # if  mpu.is_pipeline_last_stage():
    #     debug_all_rank()
    if args.dataset == 'REAL-JSON-SFT-PACKING':
        bs = position_ids.shape[0]
        seq_nums = [torch.max(x) for x in position_ids]
        concated_teacher_topk_logps = []
        concated_teacher_topk_indices = []

        assert bs == 1, "当前代码只支持bs为1的情况, 其余的处于未测试状态"
        for bs_id, seq_num in enumerate(seq_nums):
            vals, counts = torch.unique(position_ids[bs_id], return_counts=True)
            seq_id_to_len = dict(zip(vals.tolist(), counts.tolist()))
            cur_teacher_topk_logs = []
            cur_teacher_topk_indices = []
            for seq_idx in range(seq_num):
                sub_teacher_knowledge = teacher_knowledge[seq_idx].result()
                _, teacher_topk_logps, teacher_topk_indices = sub_teacher_knowledge
                # debug_all_rank()
                try:
                    seq_len = seq_id_to_len[seq_idx + 1] # seq id start from 1
                    cur_teacher_topk_logs.append(teacher_topk_logps[0][:seq_len])
                    cur_teacher_topk_indices.append(teacher_topk_indices[0][:seq_len])
                except:
                    debug_all_rank()
            # debug_all_rank()
            cur_teacher_topk_logs = torch.cat(cur_teacher_topk_logs, dim=0)
            cur_teacher_topk_indices = torch.cat(cur_teacher_topk_indices, dim=0)
            concated_teacher_topk_logps.append(cur_teacher_topk_logs)
            concated_teacher_topk_indices.append(cur_teacher_topk_indices)
        # 如果 bs > 1这里要做额外处理
        teacher_topk_logps = concated_teacher_topk_logps 
        teacher_topk_indices = concated_teacher_topk_indices
    else:
        _, teacher_topk_logps, teacher_topk_indices = teacher_knowledge.result()

    # teacher_topk_logps [tensor(4096, 256)] 外面这层len就是batch size
    # 这里将之前拿到的结果拼回去
    # debug_all_rank()
    teacher_topk_logps = torch.stack(teacher_topk_logps)
    teacher_topk_logps = teacher_topk_logps.cuda().to(torch.bfloat16)
    teacher_topk_indices = torch.stack(teacher_topk_indices)
    teacher_topk_indices = teacher_topk_indices.cuda()
```
**广播教师知识：**

```
teacher_topk_logps, teacher_topk_indices = broadcast_teacher_knowledge(teacher_topk_logps, teacher_topk_indices)

# 对数据进行 cp 的拆解, teacher 部分目前是当没有cp广播的，广播完了再拆
cp_size = mpu.get_context_parallel_world_size()
if cp_size > 1:
    batch = {"tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "teacher_topk_logps": teacher_topk_logps,
            "teacher_topk_indices": teacher_topk_indices,
            }
    if "sequence_order" in batch.keys():
        cu_seqlens = prepare_cu_seqlens(batch)
        batch_cur_cp = get_batch_on_this_cp_rank_varlen(batch, cu_seqlens)
    else:
        batch_cur_cp = get_batch_on_this_cp_rank(batch)
    # 重新获取cp后的数据
    tokens = batch_cur_cp["tokens"]
    labels = batch_cur_cp["labels"]
    loss_mask = batch_cur_cp["loss_mask"]
    attention_mask = batch_cur_cp["attention_mask"]
    position_ids = batch_cur_cp["position_ids"]
    teacher_topk_logps = batch_cur_cp["teacher_topk_logps"]
    teacher_topk_indices = batch_cur_cp["teacher_topk_indices"]

# debug_all_rank()
return tokens, labels, loss_mask, attention_mask, position_ids, num_seqs, packed_seq_params, teacher_topk_logps, teacher_topk_indices, turn_num, loss_token_num
```
**文件位置：** `baidu/ps/SearchLighting/megatron_patch/teacher/client.py`

**功能：** 异步与教师模型服务器通信，获取 top-k logits。

```
class TeacherClient:
    def __init__(self, server_ip, server_port, num_microbatches=1, max_tokens=1,
                 n_server_workers=1, temperature=1, only_response=False, max_seq_len=None) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.num_microbatches = num_microbatches
        self.n_server_workers = n_server_workers
        self.max_tokens = max_tokens
        self.task_queue = queue.Queue()
        self.mutex = threading.Lock() if n_server_workers > 1 else nullcontext()
        self.context = zmq.Context()
        self.temperature = temperature
        self.only_response = only_response
        self.max_seq_len = max_seq_len
        self._run()

    def bg_task(self):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_ip}:{self.server_port}")

        while True:
            futures = []
            batch = []
            with self.mutex:
                for _ in range(self.num_microbatches):
                    future, data = self.task_queue.get()
                    futures.append(future)
                    batch.extend(data.tolist() if isinstance(data, torch.Tensor) else data)

            if self.max_seq_len:
                max_tokens = [min(self.max_tokens, self.max_seq_len - len(prompt)) for prompt in batch]
                request = {"prompt_token_ids": batch, "max_tokens": max_tokens}
            else:
                request = {"prompt_token_ids": batch, "max_tokens": self.max_tokens}

            if self.temperature:
                request["temperature"] = self.temperature
            if self.only_response:
                request["only_response"] = True

            socket.send(serialize(request))
            # with Timer(name="recv_and_unpack", initial_text=True):
            # buffer = io.BytesIO(self.socket.recv())
            # response = msgpack.unpackb(self.socket.recv())
            # response = torch.load(buffer)
            response = deserialize(socket.recv())
            # print(len(response["teacher_topk_logprobs"]), len(response["teacher_topk_logprobs"][0]), len(response["teacher_topk_logprobs"][1]))
            mbs = len(response["teacher_topk_logprobs"]) // self.num_microbatches
            for i, future in enumerate(futures):
                responses = response["responses"][i * mbs: (i + 1) * mbs]
                teacher_topk_logps = response["teacher_topk_logprobs"][i * mbs: (i + 1) * mbs]
                # teacher_topk_logps = teacher_topk_logps.to(torch.float32)
                # if torch.cuda.is_available():
                #     teacher_topk_logps = teacher_topk_logps.pin_memory()
                # teacher_topk_logps = torch.tensor(teacher_topk_logps, dtype=torch.float32, pin_memory=True)

                # print(teacher_topk_logps.shape)
                teacher_topk_indices = response["teacher_topk_indices"][i * mbs: (i + 1) * mbs]
                # teacher_topk_indices = teacher_topk_indices.to(torch.int32)
                # if torch.cuda.is_available():
                #     teacher_topk_indices = teacher_topk_indices.pin_memory()
                # teacher_topk_indices = torch.tensor(teacher_topk_indices, dtype=torch.int32, pin_memory=True)
                future.set_result((responses, teacher_topk_logps, teacher_topk_indices))

    def _run(self):
        for _ in range(self.n_server_workers):
            threading.Thread(target=self.bg_task, daemon=True).start()

    def submit(self, data):
        future = Future()
        self.task_queue.put((future, data))
        return future
```
---

## 9. megatron_patch/model/lightning_model.py - 模型定义
**文件位置：** `baidu/ps/SearchLighting/megatron_patch/model/lightning_model.py`

### 9.1 功能说明
`LightningModel` 继承自 `GPTModel`，重写了 `forward()` 方法以支持蒸馏和 MTP。

### 9.2 关键代码
```
def forward(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input: torch.Tensor = None,
    labels: torch.Tensor = None,
    teacher_topk_logits: torch.Tensor = None,
    teacher_topk_indices: torch.Tensor = None,
    loss_mask: torch.Tensor = None,
    packed_seq_params = None,
    loss_token_num: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    
    decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = self._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        packed_seq_params=packed_seq_params,
        inference_context=kwargs.get('inference_context'),
    )

    # --- Decoder Step ---
    # debug_all_rank()
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=kwargs.get('inference_context'),
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        sequence_len_offset=sequence_len_offset,
        packed_seq_params=packed_seq_params,
        **kwargs.get('extra_block_kwargs', {})
    )

    # If not on the last stage, just return the hidden_states.
    if not self.post_process:
        return hidden_states

    # --- Postprocessing and Loss Calculation on the LAST stage ---
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if self.mtp_process:
        mtp_loss_fn = self._get_mtp_loss_function()

        mtp_teacher_logits = teacher_topk_logits.transpose(0, 1).contiguous() if teacher_topk_logits is not None else None
        mtp_teacher_indices = teacher_topk_indices.transpose(0, 1).contiguous() if teacher_topk_indices is not None else None

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        hidden_states_after_mtp = self.mtp(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            loss_token_num=loss_token_num,
            teacher_topk_logits=mtp_teacher_logits,
            teacher_topk_indices=mtp_teacher_indices,
            compute_language_model_loss=mtp_loss_fn,
            embedding=self.embedding,
            output_layer=self.output_layer,
            output_weight=output_weight
        )
```
**后处理和损失计算：**

```
else:
    # 没有 MTP 的情况
    logits = self.output_layer(hidden_states, weight=output_weight)
    
    if labels is None:
        return logits
    
    if not self.args.use_distillation:
        # 标准交叉熵损失
        lm_loss = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous(), labels.transpose(0, 1).contiguous()
        )
        distill_loss = None
    else:
        # 融合蒸馏损失
        lm_loss, distill_loss = fused_distill_vocab_parallel_cross_entropy_v2(
            logits.contiguous(),
            labels.transpose(0, 1).contiguous(),
            teacher_topk_logits,
            teacher_topk_indices,
        )
    
    return lm_loss, distill_loss, labels, logits, input_ids, packed_seq_params
```
---

### 8.4 loss_func() - 损失计算
**功能:** 统一处理 LM loss 和 distillation loss。

**文件位置：** `baidu/ps/SearchLighting/megatron_patch/helper/helper.py`

`loss_func()` 统一处理 LM loss 和 distillation loss，支持序列级平均。

```
def loss_func(loss_mask: torch.Tensor, num_seqs: torch.Tensor,loss_token_num: torch.Tensor, output_from_model: tuple):
    """
    Unified loss function with correct return signature for both
    token-level and sequence-level averaging.
    """
    args = get_args()
    # debug_all_rank()
    # 1. Unpack losses from model output
    lm_loss, distill_loss, labels, lm_logits, input_ids, packed_seq_params = output_from_model

    # 2. Prepare masks for LM loss
    loss_mask_view = loss_mask.view(-1).float()
    loss_mask_sum = loss_mask_view.sum()
    # Handle case with zero valid tokens to prevent division by zero
    if loss_mask_sum == 0:
        # Create a zero tensor for the loss to ensure type consistency
        zero_loss = torch.tensor(0.0, device=lm_loss.device if lm_loss is not None else 'cuda')
        # Return format depends on num_seqs
        if num_seqs is None:
            return zero_loss, {"lm loss": torch.tensor(0.0)}
        else:
            return zero_loss, torch.tensor(0, device=num_seqs.device), {"lm loss": torch.tensor(0.0)}

    losses_dict = {}
    # 3. Calculate local sum of losses and token counts
    reduced_lm_loss = torch.stack([lm_loss.view(-1) * loss_mask_view, loss_token_num.view(-1)])
    
    reduced_distill_loss = None
    if args.use_distillation and distill_loss is not None:
        safe_distill_loss = torch.nan_to_num(distill_loss.view(-1))
        distill_loss = safe_distill_loss * loss_mask_view
        reduced_distill_loss = torch.stack([distill_loss, loss_token_num.view(-1)])

    # 没有反向kl
    # reduced_rev_kl_loss = None
    # if args.use_distillation and rev_kl_loss is not None and args.rev_kl_loss_weight > 0:
    #     safe_rev_kl_loss = torch.nan_to_num(rev_kl_loss.view(-1))
    #     rev_kl_loss_sum = torch.sum(safe_rev_kl_loss * loss_mask_view)
    #     reduced_rev_kl_loss = torch.stack([rev_kl_loss_sum, loss_mask_sum])

    # 4. Context Parallelism Reduction
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(reduced_lm_loss, group=mpu.get_context_parallel_group())
        if reduced_distill_loss is not None:
            torch.distributed.all_reduce(reduced_distill_loss, group=mpu.get_context_parallel_group())
        # if reduced_rev_kl_loss is not None:
        #     torch.distributed.all_reduce(reduced_rev_kl_loss, group=mpu.get_context_parallel_group())

    # 5. Calculate losses for logging (globally averaged)
    # LM Loss for logging
    # avg_lm_loss_and_count = average_losses_across_data_parallel_group(
    #     [reduced_lm_loss[0], reduced_lm_loss[1]]    # [avg_loss_sum, avg_token_count]
    # )
    # debug_all_rank()
    losses_dict["lm loss"] = torch.sum(reduced_lm_loss[0] / (reduced_lm_loss[1] + 1e-8)) / num_seqs[0]
    
    # Distillation Loss for logging
    if args.use_distillation and reduced_distill_loss is not None:
        # avg_distill_loss_and_count = average_losses_across_data_parallel_group(
        #     [reduced_distill_loss[0], reduced_distill_loss[1]]
        # )
        losses_dict["distill loss"] = torch.sum(reduced_distill_loss[0] / (reduced_distill_loss[1] + 1e-8))  / num_seqs[0]
    
    # if args.use_distillation and reduced_rev_kl_loss is not None:
    #     # avg_rev_kl_loss_and_count = average_losses_across_data_parallel_group(
    #     #     [reduced_rev_kl_loss[0], reduced_rev_kl_loss[1]]
    #     # )
    #     losses_dict["reverse kl loss"] = reduced_rev_kl_loss[0] / (reduced_rev_kl_loss[1] + 1e-8)
        
    # MTP Losses for logging (populated from the helper)
    if args.mtp_num_layers is not None:
        MTPLossLoggingHelper.track_mtp_metrics(loss_scale=1.0, iteration=None, writer=None, 
                                               wandb_writer=None, total_loss_dict=losses_dict)
    
    # 6. Calculate total loss for backward dedepass (using local, pre-DP-averaged values)
    # 计算总 loss 用于反向传播
    total_loss_sum = torch.tensor(0.0, device=loss_mask.device)
    # 只有当权重>0且loss存在时，才将其加入总loss
    if args.lm_loss_weight > 0 and reduced_lm_loss is not None:
        total_loss_sum += (torch.sum(reduced_lm_loss[0] / (reduced_lm_loss[1] + 1e-8))) * args.lm_loss_weight
    if args.use_distillation and args.distill_loss_weight > 0 and reduced_distill_loss is not None:
        total_loss_sum += (torch.sum(reduced_distill_loss[0] / (reduced_distill_loss[1] + 1e-8))) * args.distill_loss_weight
    # if args.use_distillation and args.rev_kl_loss_weight > 0 and reduced_rev_kl_loss is not None:
    #     total_loss_sum += reduced_rev_kl_loss[0] * args.rev_kl_loss_weight
    # if total_loss_sum.item() > 3:
    #     debug_all_rank()

    # MTP losses are NOT added here. Their gradients are injected by MTPLossAutoScaler.

    # 7. Apply the correct return format based on num_seqs
    # Sequence-level averaging: return the sum of losses on this rank
    return total_loss_sum * args.context_parallel_size, num_seqs.sum(), losses_dict
```
**关键点：**

* **Line 609**：计算 LM loss 和对应的 token 数量
* **Line 612-615**：如果启用蒸馏，计算 distillation loss
* **Line 624-628**：如果使用 CP，做 all-reduce
* **Line 638**：计算用于日志的 LM loss（序列级平均）
* **Line 662-665**：计算用于反向传播的总 loss（加权求和）

---

## 10. 已知问题：MoE Aux Loss Scaling Bug
### 10.1 问题描述
在 Megatron-LM Core v0.14.0rc7 中，当同时满足以下条件时，MoE aux loss 的梯度缩放存在一个隐藏bug：

**触发条件：**

1. 启用 `calculate_per_token_loss=True`
2. 使用 MoE 模型
3. 进行 SFT（存在大量 padding tokens）

### 10.2 Bug 根源
**源码位置：** `megatron/core/transformer/moe/router.py:320-331`

```
if self.calculate_per_token_loss:
    # Scale the aux_loss by the number of tokens.
    # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
    # After commit 02648000, Megatron started using the number of total tokens to scale
    # gradients under the argument of calculate_per_token_loss,
    # which scales both the main_loss gradient and aux_loss gradient by
    # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
    # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
    # 用calculate_per_token_loss = False
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
else:
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
```
**问题分析：**

1. `activation.shape[0]`** 的实际含义：**

```
activation.shape[0] = seq_length * batch_size  # 包含所有tokens（含padding）
```
2. **预期行为：**

    * 在 `finalize_model_grads` 中，会用 `1.0 / global_total_valid_tokens` 缩放所有梯度
    * 所以前向中需要先乘以 **实际有效token数**，以便在反向时正确归一化

3. **实际行为：**

```
# 前向: aux_loss * (seq_length * batch_size)  [包含padding]
# 反向: grad * (1.0 / global_valid_tokens)    [不包含padding]

# 结果: 梯度被 over-scaled 了！
scaling_error_ratio = (seq_length * batch_size) / seq_nums
```
### 10.3 影响范围
**不受影响场景：**

* ✓ `calculate_per_token_loss=False` - 使用不同的缩放路径
* ✓ 非MoE模型 - 没有aux loss

**实际影响：**

* MoE aux loss 的梯度被**放大**了 `(total_tokens / valid_tokens)` 倍
* 可能导致 load balancing loss 对训练的影响**过大**
* Router 参数更新过于激进

### 10.4 建议的修复方案
**方案 1：传入实际有效token数**

```
# router.py
if self.calculate_per_token_loss:
    # 计算实际有效token数（需要传入loss_mask）
    valid_tokens = loss_mask.sum() if loss_mask is not None else activation.shape[0]
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * valid_tokens)
else:
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
```
## 总结
本文档详细讲解了基于 Megatron-LM 的 SFT 和知识蒸馏训练流程，涵盖了从启动脚本到核心训练循环的所有关键代码路径。关键特性包括：

1. **多维并行策略**：TP/PP/CP/EP/SP 组合
2. **Sequence Packing**：提高 GPU 利用率
3. **知识蒸馏**：异步获取教师 top-k logits
4. **MTP**：多 token 预测辅助任务
5. **MLA**：低秩注意力机制
6. **MoE**：专家混合模型