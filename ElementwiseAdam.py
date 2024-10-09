from torch.optim import Optimizer
import torch

class ElementWiseAdam(Optimizer):
    """
    A custom implementation of the Adam optimizer. Defaults used are as recommended in https://arxiv.org/abs/1412.6980 
    See the paper or visit Optimizer_Experimentation.ipynb for more information on how exactly Adam works + mathematics behind it.

    Params:
    lr (float): Learing rate for parameter update
    betas (Tuple[float, float]): coefficients used for calculating momentum of gradients
    eps (float): small number to prevent divided by zero
    bias_correction (bool): whether the optimizer should correct for the specified biases when taking a step. DEFAULT - TRUE.
    filters (List[Tensor]): element-wise filter tensors for adjusting parameter updates. Must have the same shape as parameters.
    """
    # Initialize optimizer with parameters
    def __init__(self, params, filters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        if lr < 0:
            raise ValueError("Invalid learning rate [{}]. Choose a positive learning rate".format(lr))
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError("Invalid beta parameters [{}, {}]. Choose positive beta parameters.".format(betas[0], betas[1]))
        params = list(params)
        if len(params) != len(filters):
            raise ValueError("The length of filters must match the length of params.")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, bias_correction=bias_correction, filters=filters)
        super(ElementWiseAdam, self).__init__(params, defaults)

    # Step method (for updating parameters)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            filters = group['filters']
            for idx, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                gradients = param.grad.data

                # State initialization
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param.data)
                    state['exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Increment step
                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(group['betas'][0]).add_(gradients, alpha=(1.0 - group['betas'][0]))

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(group['betas'][1]).addcmul_(gradients, gradients, value=(1.0 - group['betas'][1]))

                # Compute bias-corrected first moment estimate
                if group['bias_correction']:
                    bias_correction1 = 1 - group['betas'][0] ** state['step']
                    bias_correction2 = 1 - group['betas'][1] ** state['step']
                    first_unbiased = exp_avg / bias_correction1
                    second_unbiased = exp_avg_sq / bias_correction2
                else:
                    first_unbiased = exp_avg
                    second_unbiased = exp_avg_sq

                # Update parameters with filter applied
                denom = second_unbiased.sqrt().add_(group['eps'])
                step_size = group['lr']
                update = first_unbiased / denom
                param.data.add_(-step_size * update * filters[idx])

        return loss
    
    # Load state_dict from PyTorch Adam
    def load_state_dict(self, state_dict, filters=None):
        # Load state dictionary into CustomAdam
        super(ElementWiseAdam, self).load_state_dict(state_dict)
        
        # Update filters if provided
        if filters is not None:
            for group in self.param_groups:
                group['filters'] = filters