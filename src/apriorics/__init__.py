from apriorics.models import unet, logo, med_t, axialunet, gated
from timm.models.registry import register_model

for func in (unet, logo, med_t, axialunet, gated):
    register_model(func)
