from timm.models import register_model

from apriorics.models import axialunet, gated, logo, med_t, unet

for func in (unet, logo, med_t, axialunet, gated):
    register_model(func)
