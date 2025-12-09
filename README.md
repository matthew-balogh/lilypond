<img width="425" height="105" alt="Lilypond logo" src="https://github.com/user-attachments/assets/38fe54f2-3b01-4215-a263-57a7205fd8bf" />

# Lilypond

Lilypond is a visualization tool that leverages _Self-Organizing Maps (SOM)_ to **make high-dimensional** data **intuitive** and **interpretable** for the human eye.

## Motivation

Given a high-dimensional dataset at hand. We may want to visualize the data points but cannot go beyond three dimensions without dimensionality reduction. However, by using *Self-Organizing Maps (SOM)*, an organized representation in the original feature space can be learned and then flattened into a two-dimensional plane. The expectation is that data points that are similar in the original feature space will be positioned close to each other in the two-dimensional map.

Python implementation of SOM such as [MiniSom](https://github.com/JustGlowing/minisom) already exist, yet have limitations. First, they only use colors to display the neighboring distance information between nodes. This makes sparse regions appear as consecutive clusters since sharing similar colors. Also, the human eye may find it hard to interpret different colors as distance properly. Moreover, the original concept and the implementations suggest two plots for displaying the distance and activation information separately, which calls for a cross-referencing making it even more challenging to interpret the SOM map. Moreover, these libraries call for manual construction of the plots and often generate boilerplate or non-reusable code.


## Learning a representation of the dataset via SOM

<img width="1489" height="378" alt="image" src="https://github.com/user-attachments/assets/4739184a-d31d-4508-963a-30d662bef2ae" />

When it comes to a high-dimensional dataset, the convergence can be inspected in the Principal Component space.

<img width="989" height="319" alt="image" src="https://github.com/user-attachments/assets/25f00fe2-beff-460a-9758-edeb3fc3e3eb" />

<img width="758" height="616" alt="image" src="https://github.com/user-attachments/assets/d429caf3-7bf4-49b2-8722-1c200ec17057" />

<img width="864" height="372" alt="image" src="https://github.com/user-attachments/assets/d5e4818c-7236-4852-8d99-a48e9193ea87" />


## Solution

<img width="416" height="435" alt="image" src="https://github.com/user-attachments/assets/c1c6ff94-eb80-43da-a08f-b91cfe60d7e6" />

As the plot shows, *lilypond* **combines** the *hitmap* and *distance map* into a **single, familiar and easily interpretable visual**, where:

* water is a static blue background
* lily pads shrink according to how **far** they are located **from their neighbors**
* number of petals indicate the **activation** strength,

along with providing a user-friendly _configuration_ and _styling_ interface along with helper methods.

## How to use

```bash
pip install git+https://github.com/matthew-balogh/lilypond@experimental
```

```python
from minisom import MiniSom
from lilypond.basin import Basin

# ... given data X

# ... train a MiniSOM object
som = MiniSom(...)

# prepare the pond
basin = Basin(som, X, random_seed=42).prepare()
```

```python
# create, configure, style, and visualize the pond
basin.pond() \
  .discretize_petals(n_bins=10) \
  .flood(below_activations=2) \
  .style_pad(marker="s", gap=.1) \
  .style_petal(hide=False) \
  .observe()
```

```python
# plot the pond with feature wise coloring
feature_idx = 0
basin.pond() \
  .style_pad(marker="s", gap=.1) \
  .style_petal(hide=True) \
  .set_coloring_strategy(strategy="component_map", component_idx=feature_idx) \
  .observe(title=f"Feature {feature_idx + 1}")
```

## More on *lilypond*

*Lilypond* was inspired by water lilies as they cover a pond of water, with both dense and sparse regions. Data points projected on the map with different appearances can mimic benefitial and intruding species distinguishing between normal and abnormal phenomena, such as in the field of _novelty_ or _anomaly detection_.

<img width="359" height="426" alt="image" src="https://github.com/user-attachments/assets/bd3a6fb6-e3bc-4ee1-a5f1-56ffba04cfa8" />
