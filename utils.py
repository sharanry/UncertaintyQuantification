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
from numpy import linalg as LA
from jax import device_get
from sklearn.utils import shuffle


def safe_softplus(x):
    inRanges = (x < 100)
    return np.log(1 + np.exp(x*inRanges))*inRanges + x*(1-inRanges)

# the non-linearity we use in our neural network
def nonlin(x):
    return np.tanh(x)

# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H, sigma):
    D_X, D_Y = X.shape[1], 1

    # sample first layer (we put unit normal priors on all weights)
    w1 = sample("w1", dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))*sigma))  # D_X D_H
    z1 = nonlin(np.matmul(X, w1))   # N D_H  <= first layer of activations

    # sample second layer
    w2 = sample("w2", dist.Normal(np.zeros((D_H, D_H)), np.ones((D_H, D_H))*sigma))  # D_H D_H
    z2 = nonlin(np.matmul(z1, w2))  # N D_H  <= second layer of activations

    # sample final layer of weights and neural network output
    w3_mu = sample("w3_mu", dist.Normal(np.zeros((D_H, D_Y)), np.ones((D_H, D_Y))*sigma))  # D_H D_Y
    z3_mu = np.matmul(z2, w3_mu)  # N D_Y  <= output of the neural network
    
    w3_sig = sample("w3_sig", dist.Normal(np.zeros((D_H, D_Y)), np.ones((D_H, D_Y))*sigma))  # D_H D_Y
    z3_sig = np.matmul(z2, w3_sig)  # N D_Y  <= output of the neural network

    sigma_obs = safe_softplus(z3_sig)

    # observe data
    sample("Y", dist.Normal(z3_mu, sigma_obs), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng, X, Y, D_H, sigma):
    init_params, potential_fn, constrain_fn = initialize_model(rng, model, X, Y, D_H, sigma)
    samples = mcmc(args["num_warmup"], args["num_samples"], init_params,
                   sampler='hmc', potential_fn=potential_fn, constrain_fn=constrain_fn)
    return samples


# helper function for prediction
def predict(model, rng, samples, X, D_H, sigma):
    model = substitute(seed(model, rng), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = trace(model).get_trace(X=X, Y=None, D_H=D_H, sigma=sigma)
    return model_trace['Y']['value']

# create artificial regression dataset
def get_data(functions, ranges, num_samples=50):
    import random
    random.seed(37)
    onp.random.seed(37)
    random.shuffle(functions)
    X = []
    Y = []
    for i, func in enumerate(functions):
        Xs = list(onp.random.uniform(ranges[i], ranges[i+1], size=num_samples))
        Ys = list(func(Xs) + onp.random.normal(scale=0.3, size=len(Xs)))
        X.append(Xs)
        Y.append(Ys)
    X=np.array(X).reshape(-1,1)
    Y=np.array(Y).reshape(-1,1)
    
    return X, Y, X


functions = [onp.sin, onp.log]
def split_into_parts(number, n_parts):
    return np.linspace(0, number, n_parts+1)[1:]
ranges = [0] + list(split_into_parts(20, len(functions)))


def data_gen_func(X, normalizing_mean=1.0):
    if X<1.0:
        return onp.log(X * normalizing_mean)
    else:
        return onp.sin(X * normalizing_mean)


