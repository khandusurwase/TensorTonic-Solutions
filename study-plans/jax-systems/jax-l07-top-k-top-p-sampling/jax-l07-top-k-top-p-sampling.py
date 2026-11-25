import jax
import jax.numpy as jnp
from jax import lax

def sample_top_k_top_p(logits: jax.Array, uniform: jax.Array, temperature: float, top_k: int, top_p: float) -> jax.Array:
    """
    Returns an int32 array of shape (B,) holding one sampled token index per row.
    """
    scaled = logits / temperature
    kth_largest = lax.top_k(scaled, top_k)[0][:, -1:]
    scaled = jnp.where(scaled >= kth_largest, scaled, -jnp.inf)
    probs = jax.nn.softmax(scaled, axis=-1)
    sorted_probs = jnp.sort(probs, axis=-1)[:, ::-1]
    mass_before = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs
    in_nucleus = mass_before < top_p
    smallest_kept = jnp.min(jnp.where(in_nucleus, sorted_probs, jnp.inf), axis=-1, keepdims=True)
    probs = jnp.where(probs >= smallest_kept, probs, 0.0)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    cdf = jnp.cumsum(probs, axis=-1)
    token = jnp.sum(cdf <= uniform[:, None], axis=-1)
    return token.astype(jnp.int32)
