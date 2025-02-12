import jax
import jax.numpy as jnp

class EnvironmentModel:
    def __init__(self):
        pass

    @jax.jit
    def get_sensor_output(self, state):
        pass

class OmniRe(EnvironmentModel):
    def __init__(self):
        pass

    @jax.jit
    def get_sensor_output(self, state):
        return state
