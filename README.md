# Can Transformers Do Enumerative Geometry?

We introduce a Transformer-based approach to computational enumerative geometry, specifically targeting the computation of $\psi$-class intersection numbers on the moduli space of curves. Traditional methods for calculating these numbers suffer from factorial computational complexity, making them impractical to use. By reformulating the problem as a continuous optimization task, we compute intersection numbers across a wide value range from $10^{-45}$ to $10^{45}$. 
To capture the recursive and hierarchical nature inherent in these intersection numbers, we propose the Dynamic Range Activator (DRA), a new activation function that enhances the Transformer's ability to model recursive patterns and handle severe heteroscedasticity. Given precision requirements for computing $\psi$-class intersections, we quantify the uncertainty of the predictions using Conformal Prediction with a dynamic sliding window adaptive to the partitions of equivalent number of marked points. Beyond simply computing intersection numbers, we explore the enumerative "world-model" of Transformers. Our interpretability analysis reveals that the network is implicitly modeling the Virasoro constraints in a purely data-driven manner. 
Moreover, through abductive hypothesis testing, probing, and causal inference, we uncover evidence of an emergent internal representation of the the large-genus asymptotic of $\psi$-class intersection numbers. These findings suggest that the network internalizes the parameters of the asymptotic closed-form formula linearly while capturing the polynomiality phenomenon of $\psi$-class intersection numbers in a nonlinear manner.

# torch-dra

A learnable activation function `DRA` for recursive and periodic data modalities

## Installation

```bash
pip install torch-dra

