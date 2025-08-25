# %%
"""
Learnings:
- 1B   (n=28 | 67,500 steps x 3 seeds, 17,339 steps x 2 seeds): seeds 1 and 2 are short runs with only valppl.  For seeds 0, 3, 4 thefull run is provided for both ppl and olmes but there are some random missing steps for olmes.
- 750M (n=51 | 62,500 ppl x 1 seed, 26,250 x 3 seeds), seed 0 has full run valppl but only 26,250 steps olmes, seeds 2 and 3 have only the first 28,257 ppl and 26250 olmes respectively.
- 530M (n=42 | 51,250 steps x 3 seeds): seed 0 has some random NaNs in the main ppl run, but has full 57776 steps for ppl and similar 51250 for olmes. Seeds 1 & 2 have 51,411 for ppl and 51,250 for olmes. -> 3 seeds at ~51,250 steps for both.
- 300M (n=37 | 45,000 steps x 3 seeds): no random NaNs. Seed 0 and 2 have 45787 for ppl and 45000 for olmes. Seed 1 has 46250 ppl and 45000 for olmes.
- 150M (n=31 | 37,500 steps x 3 seeds): a few random nans, ppl goes a few thousand steps farther than olmes and has one more data point as with prev models.
- 90M  (n=25 | 29,901 steps x 3 seeds): some nans, ppl and olmes are exactly the same.
- 60M  (n=25 | 29,042 steps x 3 seeds): some nans in olmes, none in ppl,l olmes and ppl same len.
- 20M  (n=13 | 14,584 steps x 3 seeds): very few nans, ppl and olmes same len.
- 16M  (n=20 | 24,432 steps x 3 seeds): nans, ppl and olmes same len.
- 14M  (n=18 | 21,953 steps x 3 seeds): nans, ppl and olmes same len.
- 10M  (n=14 | 15,117 steps x 3 seeds): nans, ppl and olmes same len.
- 8M   (n=12 | 13,039 steps x 3 seeds): nans, ppl and olmes same len.
- 6M   (n=9  |  9,182 steps x 3 seeds): nans, first half of boolq_correct_prob nan, ppl and olmes same len.
- 4M   (n=6  |  5,725 steps x 3 seeds): nans, ppl and olmes same len.

QUESTION 1: Which models have 3 seeds for full run? (all but 750M)
Question 2: How many data points are included in each model's curve? (8M+ have >10)
"""

#%%
# type: ignore
%load_ext autoreload
%autoreload 2
%matplotlib inline
from hf_utils import DataDecide

# These imports require ddpred package to be installed separately
# NOTE: This notebook requires ddpred for data processing and feature extraction
try:
    from ddpred.data import DataParamConfig
    import ddpred.data as dd_data
    import ddpred.features.lrs as dd_lrs
except ImportError as e:
    print("Error: ddpred package required but not found.")
    print("Please install ddpred package: pip install -e /path/to/ddpred")
    print(f"Import error: {e}")
    raise
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# %%
dd = DataDecide()
dc = DataParamConfig(dd)
dc.use_all_data()
dc.use_all_params()
print(dc)

# %%
all_data = dd_data.get_config_data(dd, dc, use_mean=True, verbose=True)
all_data = dd.filter_by_max_step_to_use(all_data)
all_data = dd.merge_in_ds_and_model_details(all_data)
all_data = dd_lrs.add_lr_cols(all_data)
all_data.head()

# %%

all_data.head()

# %%
def prep_for_plotting(
    df, x_col, y_col,
    data_param_combos=[],
    keep_cols=[],
):
    df = df.copy()
    if len(data_param_combos) > 0:
        df = dd_data.select_by_data_param_combos(df, data_param_combos)
    df = df[[x_col, y_col] + keep_cols]
    df = df[df.isna().sum(axis=1) == 0]
    df['x_col'] = df[x_col]
    df['y_col'] = df[y_col]
    return df

# %%
def plot_model_size_curve(
    df, data, x_col, y_col, params_list,
    line_lambda=None,
    xlog=False,
    ylog=False,
    xlim=None, ylim=None,
    keep_cols=[],
):
    cmap = plt.get_cmap('tab10')
    colour_map = {param: cmap(i % 10) for i, param in enumerate(params_list)}

    plt.figure(figsize=(8,5))
    for params in params_list:
        p_df = prep_for_plotting(
            all_data, 
            data_param_combos=[(data, params)],
            x_col=x_col,
            y_col=y_col,
            keep_cols=keep_cols,
        )
        plt.plot(p_df['x_col'], p_df['y_col'], marker='o', label=f'{params}', color=colour_map[params]) 
        if line_lambda is not None:
            plt.axvline(x=line_lambda(p_df, params, x_col), linestyle='--', alpha=0.7, color=colour_map[params])
    xlabel = x_col
    ylabel = y_col
    if xlog:
        plt.xscale('log')
        xlabel = f"{xlabel} (log scale)"
    if ylog:
        plt.yscale('log')
        ylabel = f"{ylabel} (log scale)"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.title(f'{data} {y_col} vs {x_col}')
    plt.grid(True)
    plt.show()

def end_of_warmup_line_lambda(df, params, x_col):
    just_params = df[(df['params'] == params)]
    warmup_steps = just_params['warmup_steps'].values[0]
    if x_col == 'step':
        return warmup_steps
    elif x_col == 'cumulative_lr_at_step':
        return dd_lrs.calculate_cumulative_lr(
            warmup_steps,
            just_params['lr_warmup_start'].values[0],
            just_params['lr_max'].values[0],
            just_params['lr_final'].values[0],
            just_params['warmup_steps'].values[0],
            just_params['lr_decay_steps'].values[0],
        )
    elif x_col == 'tokens':
        return warmup_steps * just_params['theoretical_tokens_per_step'].values[0]
    else:
        raise ValueError(f"Invalid x_col: {x_col}")

# %%
plot_model_size_curve(
    all_data,
    data='DCLM-Baseline',
    #x_col='step',
    #x_col='tokens',
    x_col='cumulative_lr_at_step',
    y_col='c4_en-valppl',
    #y_col='lr_at_step',
    keep_cols=['params', 'warmup_steps', 'lr_warmup_start', 'lr_max', 'lr_final', 'lr_decay_steps', 'theoretical_tokens_per_step'],
    #params_list=['60M', '90M', '150M', '300M', '530M', '750M', '1B'],
    params_list=['60M', '300M'],
    #params_list=['1B'],
    line_lambda=end_of_warmup_line_lambda,
    #xlog=True, ylog=True,
    #xlog=True, ylog=False,
    #xlog=False, ylog=True,
    xlim=(60, 77.5),
    ylim=(22, 30),
)

# %% Question 3: Can I plot the full curve for a single (DS, Model, Seed, Metric)?
plot_model_size_curve(
    all_data,
    data='DCLM-Baseline',
    x_col='tokens',
    y_col='lr_at_step',
    params_list=['60M', '90M', '150M', '300M', '530M', '750M', '1B'],
    #params_list=['1B'],
    xlog=True, ylog=True,
)
# %%
plot_model_size_curve(
    all_data,
    data='DCLM-Baseline',
    x_col='cumulative_lr_at_step',
    y_col='c4_en-valppl',
    params_list=['60M', '90M', '150M', '300M', '530M', '750M', '1B'],
    #params_list=['1B'],
    #xlog=True, ylog=True,
)
#%%
pprint([col for col in all_data.columns if 'lr' in col or 'warmup' in col])



# %% Question 4: Can I plot all model sizes for a single (DS, Seed, Metric)?


def plot_lr_schedule_and_cumulative(lr_warmup_start=1e-6, lr_max=1e-3, lr_final=1e-5,
                                   warmup_steps=1000, lr_decay_steps=10000):
    """Plot both the LR schedule and cumulative LR."""
    
    total_steps = warmup_steps + lr_decay_steps
    steps = np.arange(0, total_steps + 1, 10)
    
    # Calculate LR and cumulative LR at each step
    lrs = [dd_lrs.get_lr_at_step(s, lr_warmup_start, lr_max, lr_final, 
                         warmup_steps, lr_decay_steps) for s in steps]
    cumulative_lrs = [dd_lrs.calculate_cumulative_lr(s, lr_warmup_start, lr_max, lr_final,
                                             warmup_steps, lr_decay_steps) for s in steps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot learning rate schedule
    ax1.plot(steps, lrs, 'b-', linewidth=2, label='Learning Rate')
    ax1.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='End of Warmup')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate Schedule')
    #ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot cumulative learning rate
    ax2.plot(steps, cumulative_lrs, 'g-', linewidth=2, label='Cumulative LR')
    ax2.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='End of Warmup')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Cumulative Learning Rate')
    ax2.set_title('Cumulative Learning Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return steps, lrs, cumulative_lrs

# %%
_ = plot_lr_schedule_and_cumulative()






# %%
def sklearn_polyfit(x, y, degree=3, verbose=False):
    """Fit polynomial using scikit-learn."""
    # Convert pandas Series to numpy arrays if needed
    if hasattr(x, 'values'):  # Check if it's a pandas Series/DataFrame
        x = x.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Ensure x is 2D for sklearn
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Create polynomial features and linear regression pipeline
    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=True)),
        ('linear_reg', LinearRegression())
    ])
    
    # Fit the model
    poly_pipeline.fit(x, y)
    
    # Make predictions
    y_pred = poly_pipeline.predict(x)
    
    # Get coefficients (note: order is different from numpy)
    poly_features = poly_pipeline.named_steps['poly_features']
    linear_reg = poly_pipeline.named_steps['linear_reg']
    
    feature_names = poly_features.get_feature_names_out(['x'])
    coefficients = linear_reg.coef_
    intercept = linear_reg.intercept_
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    if verbose:
        print(f"Degree {degree} polynomial:")
        print(f"  Intercept: {intercept:.4f}")
        for name, coeff in zip(feature_names[1:], coefficients[1:]):  # Skip intercept
            print(f"  {name}: {coeff:.4f}")
        print(f"R² score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
    
    return poly_pipeline, coefficients, r2, mse

# %%
p_df.head(10)
# %%
def thefit(x):
    return 105.1857 + (-19.8574 * x) + (3.3725*x**2) + (-0.2402 * x**3) + (0.0059*x**4)

print(thefit(0.733221))
print(thefit(2.932882))
print(thefit(6.598985))
print(thefit(11.731530))
print(thefit(18.329042))
print(thefit(25.575370))
print(thefit(32.711453))
print(thefit(39.635051))
print(thefit(46.250189))
print(thefit(52.469991))
# %%
def polyfit_eq(x, intercept, coefficients):
    return intercept + sum(coeff * x**(degree-i) for i, coeff in enumerate(coefficients))



# %%
p_df = prep_for_plotting(
    all_data, 
    data_param_combos=[('DCLM-Baseline', '60M')],
    x_col='cumulative_lr_at_step',
    y_col='c4_en-valppl',
    #keep_cols=['params', 'warmup_steps', 'lr_warmup_start', 'lr_max', 'lr_final', 'lr_decay_steps', 'theoretical_tokens_per_step'],
)
start_idx = -11
end_idx = -1
degree = 2
sklearn_result = sklearn_polyfit(
    x=p_df['cumulative_lr_at_step'][start_idx:end_idx],
    y=p_df['c4_en-valppl'][start_idx:end_idx],
    degree=degree,
)

# %%
color_map = {2: 'orange', 4: 'blue', 3: 'green'}
plt.figure(figsize=(10, 6))
for start_idx in [2, 6, 12]:
    #plt.figure(figsize=(10, 6))
    for degree in [2, 3, 4]:
        x = []
        y = []
        for i in range(3, len(p_df['cumulative_lr_at_step'][start_idx:])):
            x.append(p_df['cumulative_lr_at_step'].values[start_idx +i])
            sklr = sklearn_polyfit(
                x=p_df['cumulative_lr_at_step'][start_idx:start_idx+i],
                y=p_df['c4_en-valppl'][start_idx:start_idx+i],
                degree=degree,
            )
            y.append(sklr[3])
            print(start_idx + i, x[-1], sklr[3])
        plt.plot(x, y, label=f'{p_df["cumulative_lr_at_step"].values[start_idx]:0.2f} to [{p_df["cumulative_lr_at_step"].values[start_idx+3]:0.2f}-{p_df["cumulative_lr_at_step"].values[start_idx+i]:0.2f}] degree {degree}', color=color_map[degree], alpha=0.5, marker = 'o')
        print('---')
plt.title(f'MSE vs segment fit')
plt.xlabel('max cumulative_lr_at_step fit')
plt.xlim(0, 90)
plt.ylim(-0.002, 0.02)
plt.ylabel('MSE')
plt.legend()
plt.show()
    
#%%
# Create plot
plt.figure(figsize=(10, 6))

# Plot original data
plt.scatter(p_df['cumulative_lr_at_step'], p_df['c4_en-valppl'], alpha=0.7, s=50, 
            color='blue', label='Original Data')

# Plot fitted line (on original points)
plt.plot(p_df['cumulative_lr_at_step'][start_idx:end_idx], sklearn_result[0].predict(p_df[['cumulative_lr_at_step']][start_idx:end_idx]), 'r-', linewidth=2, 
            label=f'Polynomial Fit (degree {sklearn_result[1]})')
plt.plot(p_df['cumulative_lr_at_step'], sklearn_result[0].predict(p_df[['cumulative_lr_at_step']]), 'r-', linewidth=2, 
            color='blue', label=f'Polynomial Fit (degree {sklearn_result[1]})')

plt.xlabel('cumulative_lr_at_step')
plt.ylabel('c4_en-valppl')
plt.title(f'Polynomial Fit: R² = {sklearn_result[2]:.4f}')
plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(25, 55)
plt.xlim(40,90)
plt.grid(True, alpha=0.3)
plt.show()








# %%
# %%
# %%
# %%