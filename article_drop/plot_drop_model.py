import matplotlib.pyplot as plt
from math import radians
from drop_model import plot_drop


plot_drop(
    # LiquidDrop
    base=2,
    eccentricity=.75,
    contact_angle=(radians(45), radians(45)),
    light_angle=60,
    shadow_bands=4,
    shadow_color=.1,
    # TableShow
    table_color=.2,
    table_length=(.05, .5),
    blur_length=.15,
    # NoiseShow
    noise_color=1,
    noise_band=0,
    # AxisShow
    xlim=(-2, 2),
    ylim=(-1.5, 1.5),
    axis_off=True
)
plt.show()
