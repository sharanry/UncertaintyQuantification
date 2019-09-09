import os

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import jax.random as random
from jax import vmap
from jax.config import config as jax_config
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc
from numpyro import sample
import numpyro
from numpy import linalg as LA
from jax import device_get
from sklearn.utils import shuffle

from utils import *

# CONFIG
args = {
    "num_samples" : 1000, #def: 1000
    "num_warmup" : 3000, #def: 3000
    "num_data" : 100, #def: 100
    "num_hidden" : 10, #def: 10
    "device" : 'cpu', #def: cpu
    "save_directory": "./results",
}

# PREPARE TO SAVE RESULTS
try:
    os.stat(args["save_directory"])
except:
    os.mkdir(args["save_directory"]) 

sigmas = [1/5, 1, 5] # def: [1/5, 1, 5]

jax_config.update('jax_platform_name', args["device"])
N, D_X, D_H = args["num_data"], 1, args["num_hidden"]


# GENERATE ARTIFICIAL DATA
X, Y, X_test = get_data(functions, ranges, num_samples=500)
mean = X.mean()
X = X/mean
X, Y = shuffle(X, Y)


X_test=onp.arange(0,2,0.01).reshape(-1,1)


# PLOTTING
plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window
# make plots
fig, ax = plt.subplots(1, len(sigmas), sharey=True)
fig.set_figheight(5)
fig.set_figwidth(len(sigmas)*7)

samples_collected = []



# INFERENCE
for i, sigma in enumerate(sigmas):
    print("Model with weights prior sigma ", sigma)
    rng, rng_predict = random.split(random.PRNGKey(0));
    samples =  run_inference(model, args, rng, X, Y, D_H, sigma);
    samples_collected.append((sigma, samples))
    
    
    
    # predict Y_test at inputs X_test
    vmap_args = (samples, random.split(rng_predict, args["num_samples"]));
    predictions = vmap(lambda samples, rng: predict(model, rng, samples, X_test, D_H, sigma))(*vmap_args)
    predictions = predictions[..., 0]
    1
    train_predictions = vmap(lambda samples, rng: predict(model, rng, samples, X, D_H, sigma))(*vmap_args)
    train_predictions = train_predictions[..., 0]
    
    
    
    # compute mean prediction and 95% confidence interval around median
    mean_prediction = np.mean(predictions, axis=0)
    percentiles = onp.percentile(predictions, [2.5, 97.5], axis=0)
    
    # compute mean prediction and confidence interval around median
    train_mean_prediction = np.mean(train_predictions, axis=0)
      
    # plot training data
    ax[i].plot(X, Y, 'kx', c="red", alpha=0.3, label="Data samples")
    # plot 90% confidence level of predictions
    ax[i].fill_between(X_test[:,0], percentiles[0, :], percentiles[1, :], color='lightblue', label="95% CI", step='mid')
    # plot mean prediction
    ax[i].plot(X_test, mean_prediction, c='blue', alpha=0.6, label="Predicted")
    
    ax[i].plot(X_test[:100], [data_gen_func(x, normalizing_mean=mean) for x in X_test[:100]], c='purple', alpha=0.6, label="True")
    ax[i].plot(X_test[100:], [data_gen_func(x, normalizing_mean=mean) for x in X_test[100:]], c='purple', alpha=0.6)
    
    ax[i].set(xlabel="X", ylabel="Y", title="σ = " + str(sigma))
    ax[i].title.set_size(30)
    ax[i].xaxis.label.set_size(30)
    ax[i].yaxis.label.set_size(30)
    ax[i].set_ylim([-2,3])
    ax[i].tick_params(labelsize=30)
    if(i==len(samples_collected)-1):
        ax[i].legend(fontsize=15, loc="lower left")
    

print("Saving sigma analysis confidence interval plot...") 
plt.savefig(os.path.join(args["save_directory"], "sigma_ci.png"))


plt.cla()   # Clear axis
plt.clf()   # Clear figure
plt.close() # Close a figure window
fig, ax = plt.subplots(1,len(samples_collected), sharey=True)
fig.set_figheight(5)
fig.set_figwidth(len(samples_collected)*7)
for i  in range(len(samples_collected)):
    to_plot = []
    for name, value in samples_collected[i][1].items():
        value = device_get(value)
        neffs = numpyro.diagnostics.effective_sample_size(value[None, ...])
        
        if isinstance(neffs, onp.ndarray):
            to_plot.append(onp.log(neffs.flatten()))
    bplot = ax[i].boxplot(
                            to_plot, labels=list(samples_collected[i][1].keys()),
                            patch_artist=True,
                            
                        )
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bplot[element], color="black")
    for patch in bplot['boxes']:
        patch.set_facecolor("lightblue")
    
    ax[i].set(ylabel="log ESS", title="σ = " + str(samples_collected[i][0]))
    ax[i].title.set_size(30)
    ax[i].xaxis.label.set_size(30)
    ax[i].yaxis.set_label("neff")
    ax[i].yaxis.label.set_size(30)
    ax[i].tick_params(labelsize=25.0)

print("Saving sigma analysis's effective sample size box plot...") 
plt.savefig(os.path.join(args["save_directory"], "sigma_ess.png"))
        
