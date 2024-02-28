# 定义目标函数
def objective(x):  # 优化变量分别为：支付向量、带宽分配向量
    theta = x[:n]
    gamma = x[n:2*n]
    f = x[2*n:]
    t_cm = M / (gamma * B * np.log2(1 + theta * gamma / Z0))
    t_cp = mu * AD * f / f
    t_max = np.max(t_cm + t_cp)
    return t_max * np.sum((theta**2) * gamma * AD * f)