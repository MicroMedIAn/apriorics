import numpy as np
from pathaia.util.basic import ifnone
import torch
from apriorics.model_components.hooks import Hooks
from torchmetrics import Metric


def named_leaf_modules(model, name=""):
    named_children = list(model.named_children())
    if named_children == []:
        model.name = name
        return [model]
    else:
        res = []
        for n, m in named_children:
            if not (isinstance(m, torch.jit.ScriptModule) or isinstance(m, Metric)):
                pref = name + "." if name != "" else ""
                res += named_leaf_modules(m, pref + n)
        return res


def get_sizes(model, input_shape=(3, 224, 224), leaf_modules=None):
    leaf_modules = ifnone(leaf_modules, named_leaf_modules(model))

    class Count:
        def __init__(self):
            self.k = 0

    count = Count()

    def _hook(model, input, output):
        model.k = count.k
        count.k += 1
        return model, output.shape

    with Hooks(leaf_modules, _hook) as hooks:
        x = torch.rand(1, *input_shape)
        model.cpu().eval()(x)
        sizes = [list(hook.stored[1]) for hook in hooks if hook.stored is not None]
        mods = [hook.stored[0] for hook in hooks if hook.stored is not None]
    idxs = np.argsort([mod.k for mod in mods])
    return np.array(sizes, dtype=object)[idxs], np.array(mods, dtype=object)[idxs]
