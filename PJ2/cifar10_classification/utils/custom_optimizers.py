"""
手写优化器
"""
import torch

class CustomSGD:
    """手写SGD优化器"""
    def __init__(self, params, lr=0.05, momentum=0.9, weight_decay=1e-4):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data

            if self.velocity[i].device != p.data.device:
                self.velocity[i] = self.velocity[i].to(p.data.device)

            if self.weight_decay != 0:
                grad.add_(p.data, alpha=self.weight_decay)
            
            # 动量更新：velocity = momentum * velocity + grad（原地）
            self.velocity[i].mul_(self.momentum).add_(grad)
            
            # 参数更新：p = p - lr * velocity（原地）
            p.data.add_(self.velocity[i], alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self):
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'velocity': [v.clone() for v in self.velocity]
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.weight_decay = state_dict['weight_decay']
        self.velocity = state_dict['velocity']

class CustomAdam:
    """手写Adam优化器"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # 初始化一阶/二阶动量缓冲区
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0  # 全局迭代步数

    def step(self):
        """执行一步参数更新"""
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            
            # 自动同步动量缓冲区到参数所在设备
            if self.m[i].device != p.data.device:
                self.m[i] = self.m[i].to(p.data.device)
                self.v[i] = self.v[i].to(p.data.device)
            
            # 权重衰减应该在动量更新之前应用
            if self.weight_decay != 0:
                grad = grad.add(p.data, alpha=self.weight_decay)
            
            # 一阶动量更新：m = beta1*m + (1-beta1)*grad
            self.m[i] = torch.add(
                torch.mul(self.m[i], self.beta1),
                torch.mul(grad, 1 - self.beta1)
            )
            
            # 二阶动量更新：v = beta2*v + (1-beta2)*grad^2
            self.v[i] = torch.add(
                torch.mul(self.v[i], self.beta2),
                torch.mul(torch.square(grad), 1 - self.beta2)
            )
            
            # 偏差修正
            m_hat = torch.div(self.m[i], 1 - self.beta1 ** self.t)
            v_hat = torch.div(self.v[i], 1 - self.beta2 ** self.t)
            
            # 参数更新：p = p - lr * m_hat / (sqrt(v_hat) + eps)
            p.data = torch.sub(
                p.data,
                torch.mul(
                    self.lr,
                    torch.div(m_hat, torch.add(torch.sqrt(v_hat), self.eps))
                )
            )

    def zero_grad(self):
        """清空所有梯度"""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self):
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': [m.clone() for m in self.m],
            'v': [v.clone() for v in self.v],
            't': self.t
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']