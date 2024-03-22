from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass

@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t

@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        outs = {'norm': norm, 'grad_norm': norm_grad.norm().cpu().item()}
        return x_t, norm, outs

@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        outs = {'norm': norm, 'grad_norm': norm_grad.norm().cpu().item()}

        return x_t, norm, outs

@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        outs = {'norm': norm, 'grad_norm': 0.}
        return x_t, norm, outs

@register_conditioning_method(name='lgd-mc')
class LGDMCSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.mc = kwargs.get('mc', 20)
        self.scale = kwargs.get('scale', 1.0)

    def grad_and_value(self, x_prev, x_0_hat, measurement, r, **kwargs):
        _, C, H, W = x_prev.shape
        x0 = r * torch.randn((self.mc, C, H, W), device=x_0_hat.device) + x_0_hat
        x0.detach_()

        with torch.no_grad():
            x_ref = x_0_hat.detach()
            if self.noiser.__name__ == 'gaussian':
                Z = measurement - self.operator.forward(x_ref, **kwargs)
                Z = torch.linalg.norm(Z) ** 2
            else:
                Z = measurement - self.operator.forward(x_ref, **kwargs)
                Z = torch.sum(Z.abs())

        with torch.no_grad():
            if self.mc <= 1000: difference = measurement - self.operator.forward(x0, **kwargs)
            else: difference = torch.cat([(measurement - self.operator.forward(x_batch, **kwargs)) for x_batch in x0.chunk(10)])
            if self.noiser.__name__ == 'gaussian':
                p_0l = torch.exp(- (torch.linalg.norm(difference, dim=(1, 2, 3))**2) / Z) # for Gaussian Distribution
            else:
                p_0l = torch.exp(- torch.sum(difference.abs(), dim=(1, 2, 3)) / Z)
        q0t = torch.exp(-torch.linalg.norm((x0 - x_0_hat), dim=(1, 2, 3))**2)
        loss = torch.log(torch.mean(p_0l * q0t))
        x_t_grad = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]
        del x0, difference, p_0l, loss

        # grad = x_t_grad / (x_t_grad.norm()+1e-7)
        grad = x_t_grad

        with torch.no_grad():
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # norm = torch.linalg.norm(difference)
            norm = torch.linalg.norm(difference, dim=(1, 2, 3)).mean()

        return {'x_t_grad': grad, 'norm': norm,
                'Z': Z,
                'grad_norm': x_t_grad.norm().item(),
                'r': r}

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, r, eta, **kwargs):
        outs = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, r=r, **kwargs)
        x_t += self.scale * eta * outs['x_t_grad']
        norm = outs['norm']
        return x_t, norm, outs

@register_conditioning_method(name='pg')
class PGSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.mc = kwargs.get('mc', 100)
        self.scale = kwargs.get('scale', 1.0)
        self.prev_grad = None
        self.beta = kwargs.get('beta', 0.1)
        self.maxloss = -1
        self.Zscale = kwargs.get('Z', 1.)
        self.bias = kwargs.get('bias', 0.)
        self.rmin = kwargs.get('rmin', 0.)
        self.Znorm = kwargs.get('Znorm', 196608.)
        self.clamp_r = kwargs.get('clamp_r', 10.)
        self.beta_2 = 0.9 # track rounding factor
        self.prev_Z = None

    def grad_and_value(self, x_prev, x_0_hat, measurement, beta, **kwargs):

        with torch.no_grad():
            x_ref = x_0_hat.detach()
            if self.noiser.__name__ == 'gaussian':
                Z = measurement - self.operator.forward(x_ref, **kwargs)
                Z = torch.linalg.norm(Z) ** 2
                norm_base = Z.detach_()
                if self.prev_Z == None:
                    self.prev_Z = Z
                Z = self.beta_2 * self.prev_Z + (1 - self.beta_2) * Z
                self.prev_Z = Z
                # r = torch.clamp(torch.sqrt(Z / self.Znorm), min=self.rmin)--super resolution*4
                r = torch.clamp(torch.sqrt(Z / self.Znorm) / 2, min=self.rmin)
            else:
                Z = measurement - self.operator.forward(x_ref, **kwargs)
                Z = torch.sum(Z.abs())
                r = torch.clamp(Z / self.Znorm, min = self.rmin)

        # method 3: policy gradient
        _, C, H, W = x_prev.shape
        x0 = r * torch.randn((self.mc, C, H, W), device=x_0_hat.device) + x_0_hat
        x0.detach_()

        with torch.no_grad():
            if self.mc <= 1000: difference = measurement - self.operator.forward(x0, **kwargs)
            else: difference = torch.cat([(measurement - self.operator.forward(x_batch, **kwargs)) for x_batch in x0.chunk(20)])
            if self.noiser.__name__ == 'gaussian':
                err_norms = torch.linalg.norm(difference, dim=(1, 2, 3)) ** 2 - norm_base
                p_0l = torch.exp( - err_norms / Z )
            else:
                p_0l = torch.exp(- torch.sum(difference.abs(), dim=(1, 2, 3)) / Z)
            pB = (torch.sum(p_0l) - p_0l) / (self.mc - 1)
            pB.detach_()
            b_norm = torch.mean(p_0l)
        log_q0t = torch.linalg.norm((x0 - x_0_hat), dim=(1, 2, 3))**2 / (2 * r * r)
        loss = torch.mean((p_0l-pB) * log_q0t)
        x_t_grad = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]

        if self.prev_grad == None:
            self.prev_grad = x_t_grad
            self.grad_scale = x_t_grad.norm()
            tmp_grad = x_t_grad
        else:
            tmp_grad = x_t_grad
            x_t_grad = self.beta * beta * self.prev_grad + (1 - self.beta * beta) * tmp_grad
            self.prev_grad = tmp_grad

        # for AISTATS submission, rounding using C
        grad = x_t_grad / (x_t_grad.norm()+1e-7)
        # for AISTATS rebuttal, normalizing using Ep_l
        with torch.no_grad():
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference, dim=(1, 2, 3)).mean()

        return {'x_t_grad': grad, 'norm': norm,
                'Z': Z,
                'grad_norm': grad.norm().item(),
                'r': r}

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, eta, beta, r, **kwargs):
        outs = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, beta=beta, **kwargs)
        x_t -= self.scale * eta * outs['x_t_grad']
        norm = outs['norm']
        return x_t, norm, outs
