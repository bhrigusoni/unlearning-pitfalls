"""SAM over the explicitly supplied trainable parameter scope.

v2 runs SAM twice per step, once per objective, each with its own radius. Between
the two the finished forget gradient is parked on the CPU so the retain pass can
build its ascent direction from a clean retain gradient.
"""
import torch


def gradient_norm(parameters):
    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    if not norms:
        raise RuntimeError('No gradients in optimization scope.')
    norm = torch.stack(norms).norm()
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm.')
    return norm


class SAM:
    def __init__(self, parameters, rho):
        self.parameters = list(parameters)
        self.rho = rho
        self.originals = []

    @torch.no_grad()
    def perturb(self):
        if self.originals:
            raise RuntimeError('SAM parameters already perturbed.')
        scale = self.rho / (gradient_norm(self.parameters).item() + 1e-12)
        effective_squared = 0.0
        try:
            for p in self.parameters:
                if p.grad is None:
                    continue
                # CPU snapshots avoid another model-sized GPU allocation and ensure
                # bit-exact restoration, including after BF16 rounding or exceptions.
                old = p.detach().cpu().clone()
                self.originals.append((p, old))
                p.copy_((p.float() + p.grad.float() * scale).to(p.dtype))
                effective_squared += (p.detach().float().cpu() - old.float()).square().sum().item()
        except BaseException:
            self.restore()
            raise
        return effective_squared ** 0.5

    @torch.no_grad()
    def restore(self):
        for p, old in self.originals:
            p.copy_(old)
        self.originals.clear()


@torch.no_grad()
def take_gradients(parameters):
    """Move the accumulated gradient off the parameters and onto the CPU.

    Returned so a later put_gradients adds it back. The next backward therefore
    allocates fresh gradient buffers and the objective that runs in between sees its
    own gradient alone, which is what its SAM direction has to be built from.
    Snapshots are kept in the parameter dtype on the CPU, matching SAM's weight
    snapshots, so no second model-sized GPU allocation is needed.
    """
    saved = [(p, p.grad.detach().cpu().clone()) for p in parameters if p.grad is not None]
    for p, _ in saved:
        p.grad = None
    return saved


@torch.no_grad()
def put_gradients(saved):
    """Add a take_gradients snapshot back onto whatever the parameters now hold."""
    for p, value in saved:
        value = value.to(p.device, p.dtype)
        p.grad = value if p.grad is None else p.grad.add_(value)


class MasterSGD(torch.optim.Optimizer):
    """Plain SGD, FP32 master weights on CPU; no momentum or weight decay.

    Full BF16 weights + gradients fit much more easily than Adam states. CPU
    masters preserve accumulated small updates that BF16 would otherwise lose.
    ponytail: CPU transfers trade speed for memory; add sharding only if needed.
    """
    def __init__(self, parameters, lr):
        super().__init__(parameters, {'lr': lr})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Pass an explicit backward closure to the training loop.')
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'master' not in state:
                    state['master'] = p.detach().float().cpu().clone()
                state['master'].add_(p.grad.detach().float().cpu(), alpha=-group['lr'])
                p.copy_(state['master'])

    def load_state_dict(self, payload):
        # Optimizer.load_state_dict normally casts states to the parameter's
        # dtype/device. Keep master weights FP32/CPU on resume.
        saved = payload['state']
        current = [p for group in self.param_groups for p in group['params']]
        indices = [i for group in payload['param_groups'] for i in group['params']]
        if len(current) != len(indices) or len(payload['param_groups']) != len(self.param_groups):
            raise ValueError('Optimizer scope changed.')
        self.state.clear()
        for p, i in zip(current, indices):
            if i in saved:
                master = saved[i]['master']
                if master.shape != p.shape:
                    raise ValueError('Master weight shape mismatch.')
                self.state[p]['master'] = master.float().cpu()
        for group, old in zip(self.param_groups, payload['param_groups']):
            group['lr'] = old['lr']
