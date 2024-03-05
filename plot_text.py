import matplotlib.pyplot as plt

def get_text(ax, x_horz, y_vert):
    for p in ax.patches:
        percentage = f'{p.get_height() : .2f}%'
        x = p.get_x() + p.get_width() / x_horz
        y = p.get_y() + p.get_height() + y_vert
        ax.annotate(percentage, (x, y), size=10, ha='center', va='center')


def get_text_horz(ax, x_add, n):
    i = 0
    for p in ax.patches:
        x_val = p.get_width() + x_add
        if i < n:
            plt.text(x=x_val, y=i, s=f'{x_val : .2f}%')
            i += 1