# Guide to use the data

```python
# import all

DATA_DIR = "./data/synthetic-2d-3-cluster"

X = pd.read_csv(f"{DATA_DIR}/X.csv", header=None, index_col=0)
X.columns = [0, 1]
```

```python
# import normal and abnormal samples separately

DATA_DIR = "./data/synthetic-2d-3-cluster"

X_normal = pd.read_csv(f"{DATA_DIR}/X_normal.csv", header=None, index_col=0)
X_normal.columns = [0, 1]

X_abnormal = pd.read_csv(f"{DATA_DIR}/X_abnormal.csv", header=None, index_col=0)
X_abnormal.columns = [0, 1]

X = pd.concat((X_normal, X_abnormal))
```

```python
# scatter plot

plt.figure()

plt.scatter(X.loc[:, 0], X.loc[:, 1], alpha=0.5)

plt.title("Original 2D Data (3 clusters, showing normal data)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.gca().axes.set_aspect("equal")
plt.show()
```