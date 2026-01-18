import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

df = pd.read_csv("task2_sweep_seat_times.csv")

conf_params =  ["p_luggage", "walk_time_per_row", "base_seat_time",
              "interference_penalty", "luggage_add_time"]

df = df.loc[df['policy']=='random']
df_mean_time = df.groupby(conf_params, as_index=False)["finish_time_s"].mean()


params_ordered = []
#order parameters so we display them sequentially
for i, y_param in enumerate(conf_params):
    for x_param in conf_params[i+1:]:
        params_ordered.append((x_param, y_param)) 

# Comon min max colorbar limit
vmin = df_mean_time["finish_time_s"].min()
vmax = df_mean_time["finish_time_s"].max()


n_pairs = len(params_ordered)
ncols = 2
nrows = (n_pairs + 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows*5))
axes = axes.flatten()

# heatmap for each combination
for i, (x_param, y_param) in enumerate(params_ordered):
    ax = axes[i]
    
    pivot = df_mean_time.pivot_table(index=y_param,
                                     columns=x_param,
                                     values="finish_time_s",
                                     aggfunc="mean")
    sns.heatmap(pivot,
                ax=ax,
                cmap="viridis",
                annot=True,    
                fmt=".2f",            
                annot_kws={"color": "white", "fontsize": 8},  
                linewidths=0.3,
                cbar=True,
                cbar_kws={'label': 'Mean boarding time'},       
                vmin=vmin,
                vmax=vmax)
    
    # fontsize of color map label
    cbar = ax.collections[0].colorbar  
    cbar.ax.tick_params(labelsize=7)   
    cbar.ax.yaxis.label.set_size(7.5) 
    
    ax.invert_yaxis()
    ax.set_ylabel(y_param, fontsize=7.5)
    ax.set_xlabel("")
    
    # Tick labels
    ax.tick_params(axis='x', rotation=0, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    
  
    ax.text(x=0.5,
            y=-0.1,
            s=x_param,
            ha='center',
            va='top',
            fontsize=7,
            rotation=0,
            transform=ax.transAxes)

plt.tight_layout(h_pad=2, w_pad=2)
plt.show()
