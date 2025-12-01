```
算法：Actor-Critic PPO (Proximal Policy Optimization) 用于RLHF

输入:
    - 预训练语言模型 π_θ (Actor，策略网络，参数为θ)
    - 值函数网络 V_φ (Critic，参数为φ)
    - 参考模型 π_ref (通常是π_θ的副本)
    - 奖励模型 R
    - 提示数据集 D
    - 迭代次数 N
    - 每轮样本数 M
    - PPO截断参数 ε (通常为0.1或0.2)
    - KL散度目标值 β_target
    - Actor学习率 α_π
    - Critic学习率 α_v
    - 折扣因子 γ
    - GAE参数 λ_GAE

输出:
    - 优化后的语言模型 π_θ 和值函数网络 V_φ

过程:
    初始化参考模型 π_ref ← π_θ
    
    对于 iteration = 1 到 N:
        // 1. 数据收集阶段
        样本集 S = {}
        从数据集D中随机采样M个提示 {x_1, x_2, ..., x_M}
        
        对于每个提示 x_i:
            使用当前策略模型π_θ生成响应序列 y_i = (y_i,1, y_i,2, ..., y_i,T)
            对于每个时间步t:[这儿是prefill推理，是一次推理，而不是token by token的推理得到结果]
                计算状态值 V_φ(x_i, y_i,1:t-1)
                使用参考模型计算动作概率 p_ref,t = π_ref(y_i,t | x_i, y_i,1:t-1)
            使用奖励模型计算最终奖励 r_i = R(x_i, y_i)
            S = S ∪ {(x_i, y_i, r_i, {p_ref,t}, {V_φ(x_i, y_i,1:t-1)})}
        
        // 2. 计算GAE优势估计
        对于每个样本(x_i, y_i, r_i, {p_ref,t}, {V_t})在S中:
            // 计算每个时间步的奖励（简化为最终奖励分配到每个token）
            r_i,t = r_i / T 为每个时间步t
            
            // 计算TD残差
            δ_t = r_i,t + γ*V_t+1 - V_t  (假设最后一步V_T+1 = 0)
            
            // 使用GAE计算优势值
            A_i,t = 0
            for t = T 到 1 (倒序):
                A_i,t = δ_t + γ*λ_GAE*A_i,t+1
            
            // 计算回报目标
            G_i,t = V_t + A_i,t
        
        // 3. 策略(Actor)和值函数(Critic)优化
        对于 k = 1 到 K(PPO更新次数):
            对于每个小批量(x_i, y_i, {r_i,t}, {p_ref,t}, {V_t}, {A_i,t}, {G_i,t})在S中:
                // 3.1 Actor更新
                对于每个时间步t:
                    // 计算当前策略下的概率
                    p_θ,t = π_θ(y_i,t | x_i, y_i,1:t-1)
                    
                    // 计算重要性权重比例
                    ratio_i,t = p_θ,t / p_ref,t
                    
                    // 计算PPO截断目标
                    objective1 = ratio_i,t * A_i,t
                    objective2 = clip(ratio_i,t, 1-ε, 1+ε) * A_i,t
                    actor_loss = -平均(min(objective1, objective2))
                    
                    // 计算KL散度惩罚
                    KL = KL散度(p_ref,t, p_θ,t)
                    KL惩罚 = max(0, KL - β_target)
                    
                    // 总Actor损失
                    L_actor = actor_loss + λ * KL惩罚
                
                // 3.2 Critic更新
                对于每个时间步t:
                    // 计算当前值函数估计
                    V_current = V_φ(x_i, y_i,1:t-1)
                    
                    // 计算值函数损失(MSE)
                    critic_loss = 平均((V_current - G_i,t)²)
                
                // 3.3 更新网络参数
                使用梯度下降更新θ: θ ← θ - α_π * ∇_θL_actor
                使用梯度下降更新φ: φ ← φ - α_v * ∇_φcritic_loss
                
                // 3.4 检查KL散度是否过大
                if 平均(KL) > 2 * β_target:
                    提前结束当前iteration
        
        // 4. 定期更新参考模型
        if iteration % update_frequency == 0:
            π_ref ← π_θ
    
    返回优化后的模型 π_θ 和 V_φ

```