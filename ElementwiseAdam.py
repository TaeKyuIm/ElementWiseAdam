from torch.optim import Optimizer
import torch

class ElementWiseAdam(Optimizer):
    """
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
            params = group['params']
            betas = group['betas']
            eps = group['eps']
            lr = group['lr']
            bias_correction = group.get('bias_correction', True)

            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            grads = []
            filtered_params = []

            for idx, p in enumerate(params):
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Increment state step
                state['step'] += 1

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])
                grads.append(grad)
                filtered_params.append((p, filters[idx]))

            if not grads:
                continue

            # Convert lists to tuples for foreach functions
            exp_avgs = tuple(exp_avgs)
            exp_avg_sqs = tuple(exp_avg_sqs)
            grads = tuple(grads)
            # Note: state_steps remains a list because we don't need it to be a tuple

            beta1, beta2 = betas

            # Decay the first and second moment running average coefficient
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

            if bias_correction:
                bias_correction1 = [1 - beta1 ** step for step in state_steps]
                bias_correction2 = [1 - beta2 ** step for step in state_steps]
                # Compute step sizes
                step_sizes = [lr / bc1 for bc1 in bias_correction1]
            else:
                step_sizes = [lr] * len(state_steps)

            # Compute the denominator
            denom = [exp_avg_sq.sqrt().add_(eps) for exp_avg_sq in exp_avg_sqs]

            # Compute the step
            steps = [(-step_size * (exp_avg / d)) for step_size, exp_avg, d in zip(step_sizes, exp_avgs, denom)]

            # Apply filters and update parameters
            for (param, filter_tensor), step in zip(filtered_params, steps):
                param.data.add_(step * filter_tensor)

        return loss

    
    # Load state_dict from PyTorch Adam
    def load_state_dict(self, state_dict, filters=None):
        # Load state dictionary into CustomAdam
        super(ElementWiseAdam, self).load_state_dict(state_dict)
        
        # Update filters if provided
        if filters is not None:
            for group in self.param_groups:
                group['filters'] = filters

        for group in self.param_groups:
            if 'bias_correction' not in group:
                group['bias_correction'] = self.defaults.get('bias_correction', True)