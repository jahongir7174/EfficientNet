import math

import torch


def round_width(filters, divisor=8):
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_depth(repeats):
    return int(math.ceil(repeats))


def init_weight(self):
    for m in self.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            size = m.weight.size()[0]
            init_range = 1.0 / math.sqrt(size)
            m.weight.data.uniform_(-init_range, init_range)
            if m.bias is not None:
                m.bias.data.zero_()


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class SE(torch.nn.Module):
    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      torch.nn.SiLU(),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, s, r):
        super().__init__()
        identity = torch.nn.Identity()
        features = [Conv(in_ch, r * in_ch, torch.nn.SiLU()) if r != 1 else identity,
                    Conv(r * in_ch, r * in_ch, torch.nn.SiLU(), k, s, r * in_ch),
                    SE(r * in_ch, r),
                    Conv(r * in_ch, out_ch, identity)]

        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        y = self.res(x)
        return x + y if self.add else y


class EfficientNet(torch.nn.Module):
    """
                         w,   d,   r,  rate
     efficientnet-b0 : (1.0, 1.0, 224, 0.2),
     efficientnet-b1 : (1.0, 1.1, 240, 0.2),
     efficientnet-b2 : (1.1, 1.2, 260, 0.3),
     efficientnet-b3 : (1.2, 1.4, 300, 0.3),
     efficientnet-b4 : (1.4, 1.8, 380, 0.4),
     efficientnet-b5 : (1.6, 2.2, 456, 0.4),
     efficientnet-b6 : (1.8, 2.6, 528, 0.5),
     efficientnet-b7 : (2.0, 3.1, 600, 0.5),
     efficientnet-b8 : (2.2, 3.6, 672, 0.5),
    """

    def __init__(self, num_class=1000, w=1.0, d=1.0, rate=0.2):
        super().__init__()
        num_dep = [1, 2, 2, 3, 3, 4, 1]
        filters = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        num_dep = [round_depth(depth * d) for depth in num_dep]
        filters = [round_width(width * w) for width in filters]

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        for i in range(num_dep[0]):
            if i == 0:
                self.p1.append(Conv(3, filters[0], torch.nn.SiLU(), 3, 2))
                self.p1.append(Residual(filters[0], filters[1], 3, 1, 1))
            else:
                self.p1.append(Residual(filters[1], filters[1], 3, 1, 1))
        # p2/4
        for i in range(num_dep[1]):
            if i == 0:
                self.p2.append(Residual(filters[1], filters[2], 3, 2, 6))
            else:
                self.p2.append(Residual(filters[2], filters[2], 3, 1, 6))
        # p3/8
        for i in range(num_dep[2]):
            if i == 0:
                self.p3.append(Residual(filters[2], filters[3], 5, 2, 6))
            else:
                self.p3.append(Residual(filters[3], filters[3], 5, 1, 6))
        # p4/16
        for i in range(num_dep[3]):
            if i == 0:
                self.p4.append(Residual(filters[3], filters[4], 3, 2, 6))
            else:
                self.p4.append(Residual(filters[4], filters[4], 3, 1, 6))
        for i in range(num_dep[4]):
            if i == 0:
                self.p4.append(Residual(filters[4], filters[5], 5, 1, 6))
            else:
                self.p4.append(Residual(filters[5], filters[5], 5, 1, 6))
        # p5/5
        for i in range(num_dep[5]):
            if i == 0:
                self.p5.append(Residual(filters[5], filters[6], 5, 2, 6))
            else:
                self.p5.append(Residual(filters[6], filters[6], 5, 1, 6))
        for i in range(num_dep[6]):
            if i == 0:
                self.p5.append(Residual(filters[6], filters[7], 3, 1, 6))
            else:
                self.p5.append(Residual(filters[7], filters[7], 3, 1, 6))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc = torch.nn.Sequential(Conv(filters[7], filters[8], torch.nn.SiLU()),
                                      torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Dropout(rate),
                                      torch.nn.Linear(filters[8], num_class))

        init_weight(self)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        return self.fc(x)

    def export(self):
        from timm.models.layers import Swish
        for n, m in self.named_modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                if isinstance(m.relu, torch.nn.SiLU):
                    m.relu = Swish()
                m.conv = fuse_conv(m.conv, m.norm)
                delattr(m, 'norm')
                m.forward = m.fuse_forward
            elif type(m) is SE:
                m.se[2] = Swish()
        return self

    def fuse(self):
        for n, m in self.named_modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                delattr(m, 'norm')
                m.forward = m.fuse_forward
        return self


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.base_values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class EMA(torch.nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model)

        self.model.eval()

    def update_fn(self, model, fn):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(fn(e, m))

    def update(self, model):
        self.update_fn(model, fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0.,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        prob = self.softmax(x)
        mean = -prob.mean(dim=-1)
        nll_loss = -prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        return ((1. - self.epsilon) * nll_loss + self.epsilon * mean).mean()
