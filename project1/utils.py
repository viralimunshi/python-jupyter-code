import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import ipywidgets

def plot_tangent_line(
    *,
    f,
    f_prime,
    interval_size,
    domain,
    y_label='MSE'
):
    def plot(h):
        fig, ax = plt.subplots()
        plt.plot(domain, f(domain))

        left_y = f(h) - f_prime(h) * interval_size/2
        right_y = f(h) + f_prime(h) * interval_size/2
        left_x = h - interval_size/2
        right_x = h + interval_size/2
        plt.plot([left_x, right_x], [left_y, right_y], color='black', linestyle='--')

        plt.scatter(h, f(h), zorder=10)
        
        plt.xlim([25_000, 200_000])
        plt.ylim([np.min(f(domain))*.98, np.max(f(domain)) * 1.02])
        plt.xlabel('Prediction');
        plt.ylabel(y_label);

    return ipywidgets.interact(
        plot, 
        h=ipywidgets.FloatSlider(min=25_000, max=200_000, step=2_000, continuous_update=False)
    )
