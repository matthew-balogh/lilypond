<img width="425" height="105" alt="Lilypond logo" src="https://github.com/user-attachments/assets/38fe54f2-3b01-4215-a263-57a7205fd8bf" />

[![Built with Oversee](https://img.shields.io/badge/Built%20with-Oversee-4e37f5?style=flat)](https://app.overseelabs.xyz)

# lilypond

Lilypond is a visualization tool that leverages _Self-Organizing Maps (SOM)_ to **make high-dimensional** data **intuitive** and **interpretable** for the human eye.

## What is *lilypond*?

Given a high-dimensional dataset at hand. We may want to visualize the data points but cannot go beyond three dimensions without dimension reduction.

However, by using *Self-Organizing Maps (SOM)*, we can project the high-dimensional data into a two-dimensional plane considering all the original dimensions. The expectation is that data points that are similar in the original feature space will be positioned close to each other in the two-dimensional grid consisiting of representative nodes (centroids).

Implementations of SOM such as [MiniSom](https://github.com/JustGlowing/minisom) plots the data as in the first two of the following plots:

<img width="1181" height="487" alt="Comparison of traditional SOM visuals and lilypond" src="https://github.com/user-attachments/assets/65879725-6768-4114-b584-3377ef0dbf20" />

As the rightmost plot shows, *lilypond* **combines** the *hitmap* and *distance map* into a **single, familiar and easily interpretable visual**, where:

* water is a static blue background
* lily pads shrink according to how **far** they are located **from their neighbors**
* number of petals indicate the **activation** strength

Of course, map layers can be stacked using traditional techniques. However, one of the reasons lilypond was created is that the conventional visualization of the distance map is prone to misinterpretation. Darker shaded tiles suggest closely related nodes, while red-colored tiles indicate large distances from their neighbors. Without careful attention, the observer might mistake these red nodes as being similar to each other, just as in the case of the blue regions.

One of the cool features of _lilypond_ is calling `.flood()` on the pond, which will raise the water level and non or rarely activated lilies get obscured to make you focus on the other populated ones.
