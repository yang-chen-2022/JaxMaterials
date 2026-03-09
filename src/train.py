"""Very simple Jax training loop"""

import jax
import torch
import optax
from jax import numpy as jnp
from flax import linen as nn
from jaxmaterials.data.data import LayeredFibresDataset


class SimpleCNN(nn.Module):
    """A very simple CNN model which downsamples the data"""

    @nn.compact
    def __call__(self, x):
        # Layer 1
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 2
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 3
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 4
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 5
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        s = x.shape
        x = jnp.reshape(x, (-1, s[-1] * s[-2] * s[-3] * s[-4]))
        # dense layer
        x = nn.Dense(features=6 * 6)(x)
        x = jnp.reshape(x, (-1, 6, 6))
        return x


# batch size
batchsize = 2
# number of epochs used for training
epochs = 32


filename = "fibres.h5"

# load data from disk
dataset = LayeredFibresDataset(filename, features_last=True)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batchsize, shuffle=True
)

# initialise the model and print it to screen
model = SimpleCNN()

X_first, _ = next(iter(train_dataloader))
X_first = jnp.asarray(X_first)

params = model.init(jax.random.key(0), X_first)
print(model.tabulate(jax.random.key(0), X_first))

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)


def loss_fn(params, X, y_true):
    """Loss function

    :arg params: model parameters
    :arg X: input tensor with Lame parameters
    :arg y_true: true value of effective elasticity tensor
    """
    y = model.apply(params, X)
    return jnp.mean(optax.l2_loss(y, y_true))


# Train model
for epoch in range(epochs):
    running_loss = 0
    for batch, (X, y_true) in enumerate(train_dataloader):
        X_ = jnp.asarray(X)
        y_true_ = jnp.asarray(y_true)
        grads = jax.grad(loss_fn)(params, X_, y_true_)
        increment, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, increment)
        loss = loss_fn(params, X_, y_true_)
        running_loss += (loss - running_loss) / (batch + 1)
    print(f"epoch {epoch:4d} training loss = {running_loss:8.4f}")
