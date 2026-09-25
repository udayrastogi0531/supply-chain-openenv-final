Action space

The environment accepts three non-negative order quantities, one per product.

- Shape: `[3]`
- Type: `float32`
- Minimum: `0` for each product
- Maximum: `100` for each product

Keeping actions inside this declared range avoids invalid requests and makes agent behavior consistent with the OpenEnv manifest.