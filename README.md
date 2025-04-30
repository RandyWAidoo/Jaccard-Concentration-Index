# Jaccard-Concentration Index

**Jaccard-Concentration Index (JCI)** is a Python library for evaluating the quality of clustering (or, more generally, classification) using a novel metric that combines the well-known Jaccard index with a custom concentration score. It provides a more nuanced view of cluster purity by not only considering the best matches between predicted and true clusters but also measuring how concentrated each predicted cluster's mass is across the true clusters.

In general, predicted clusters that distribute their mass among a minimal number of true clusters will score higher. Clusters that distribute their mass unevenly-heavily favoring one or a few true clusters-will score even higher. For example, if there are 4 true clusters, a predicted cluster that distributes its mass in a 70-30-0-0 split will score better than one with a 65-35-0-0 split, and that one will, interestingly, score better than a cluster with a 70-10-10-10 split. This behavior stems from the dual emphasis on the strength of overlap with true clusters and the focus of that overlap. Having a higher maximum overlap with a true cluster is generally preferable, but concentrating the remaining mass is important as well because it reduces uncertainty about which true class a point in the cluster belongs to-making the classification more useful. 

In essence, the Jaccard-Concentration Index provides a smooth way to balance the precision and recall of a prediction.

---

## Functions

### `concentration`

```python
concentration(
    values: np.ndarray[np.float64],
    single_index: bool = False,
    size_invariance: bool = True,
    virtual_length: int = 0
) -> float
```

**Description:**  
This function calculates how concentrated the total value of a list of numbers is across one or a few indices. It can be used to assess the quality of a clustering based on how a predicted cluster overlaps with true clusters.

#### Parameters:
- **`values`**: `np.ndarray[np.float64]` (array-like of shape `(n_samples,)`)  
  The list of values (e.g., Jaccard indices or any other metric to be evaluated).

- **`single_index`**: `bool`, optional (default=`False`)  
  If `True`, higher scores are given when the maximum value is significantly larger than all other values. If `False`, higher scores are given when the values are concentrated in a minimal number of indices.

- **`size_invariance`**: `bool`, optional (default=`True`)  
  If `True`, the function returns a score between 0 and 1 that is independent of the size of the input list. If `False`, the score will be in the range [1/(number-of-values), 1]

- **`virtual_length`**: `int`, optional (default=`0`)  
  If specified, the function computes the score as if the `values` list has `virtual_length` items, effectively padding the list with zeros. The `virtual_length` must be greater than or equal to the actual length of `values`.

#### Returns:
- **`float`**: A concentration score in the range [0, 1] (or in the range [1/(number-of-values), 1] if `size_invariance=False`). Higher scores indicate a more concentrated distribution.

---

### `jaccard_concentration_index`

```python
jaccard_concentration_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    noise_label: Union[Any, None] = None,
    return_all: bool = False,
    ordered_labels: Sequence[Any] = []
) -> Union[float, dict]
```

**Description:**  
This function computes the Jaccard-Concentration Index for clustering evaluation. It assesses how well the predicted cluster assignments match the true cluster labels by combining the Jaccard index (to measure overlap) with the concentration score (to measure the focus of that overlap).

#### Parameters:
- **`y_true`**: `np.ndarray` (array-like of shape `(n_samples,)`)  
  The true cluster labels.

- **`y_pred`**: `np.ndarray` (array-like of shape `(n_samples,)`)  
  The predicted cluster labels.

- **`noise_label`**: `Any`, optional, (default=`None`)  
  A label within `y_pred` that represents assignment to a noise cluster.
  Scores for the noise cluster will not be computed and will not affect the final averages directly.
  However, the mass of the true clusters that is deposited in the noise cluster will count against the final score.

- **`return_all`**: `bool`, optional (default=`False`)  
  If `True`, the function returns all global metrics along with detailed metrics for each cluster. If `False`, only the global JCI score is returned.

- **`ordered_labels`**: `Sequence[Any]`, optional (default=`[]`)  
  A sequence of ordered labels to assign to the clusters. Label positions within the sequence should correspond to the sorted order of labels within `y_true`. For example, if `ordered_labels=['A', 'B']`, then the value `'A'` corresponds to the smallest label within `y_true` and `'B'` corresponds to the second smallest label within `y_true`.

#### Returns:
- If `return_all=False`:  
  A `float` representing the global Jaccard-Concentration Index score.

- If `return_all=True`:  
  A `dict` containing:
  - `'score'`: The macro-average JCI score across all clusters.
  - `'macroavg_max_jaccard_index'`: The macro-average of the highest Jaccard indices for each cluster.
  - `'macroavg_concentration'`: The macro-average concentration score across clusters.
  - `'cluster_results'`: A list of dictionaries, each containing the following per-cluster results:
    - `'score'`: Jaccard-Concentration Index for the cluster.
    - `'max_jaccard_index'`: The highest Jaccard index for the cluster.
    - `'concentration'`: The concentration score for the cluster.
    - `'closest_label_index'`: The index of the true label most closely matching the cluster.
    - `'closest_label'`: The true label most closely matching the cluster (if applicable, depending on `ordered_labels`).
    - `'size_proportion'`: The size of the cluster divided by the dataset size excluding the noise cluster.  
  
  Note that the noise cluster, if present, will not have clustering results. This means the list of per-cluster results may be one element shorter than the original number of clusters

---

## Features

- **Max Jaccard Index ($MJI$):**  
  For each predicted cluster, this metric computes the highest Jaccard index with any true cluster. The Jaccard index is defined as:  
  $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$  
  A score of 1 indicates perfect overlap between the predicted and a true cluster, while lower scores indicate less overlap.

- **Concentration Score ($C$):**  
  This metric measures how unevenly a cluster's total mass is distributed across true clusters. Specifically, it rewards distributions that concentrate the majority of their value in one or a few indices and penalizes distributions that approach uniformity. A concentration score of 0 indicates that the mass is uniformly distributed (low purity), whereas a score of 1 indicates that all the mass is concentrated in a single cluster (high purity). The score is computed by:
  1. **Normalization:**  
     Normalize the input vector (e.g., a list of Jaccard indices) so that its values sum to 1.
  2. **Squaring:**  
     Square each of the normalized values to exaggerate the relative differences.
  3. **Comparison to Uniformity:**  
     Compare the sum of the squared values to that of a uniform distribution over the vector's length.
  4. **Virtual Length (optional):**  
     Increasing the "virtual length" (i.e., adding implicit zeros) boosts the concentration score because the same amount of value is now packed into a relatively smaller number of indexes.
  
- **Jaccard-Concentration Index:**  
  For each predicted cluster, the JCI is computed as the geometric mean of the max Jaccard index and the concentration score:  
  $JCI = \sqrt{MJI \times C}$  
  This formulation ensures that both a high overlap (high MJI) and a focused distribution (high concentration) are necessary for a high overall score.

- **Macro-Averaging:**  
  The library aggregates the scores from each predicted cluster by computing a macro-average weighted by cluster size. This means that larger clusters contribute more to the final score, reducing the impact of small, noisy clusters.

---

## Installation

You can install the library using pip: 

```bash
pip install jaccard-concentration-index
```

---

## Usage Example

Here's a simple example of how to use the library:

```python
import numpy as np
from jaccard_concentration_index import jaccard_concentration_index, concentration

# True and predicted cluster labels
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 0, 1, 1, 2, 2])

# Compute the macro-averaged Jaccard-Concentration Index
global_score = jaccard_concentration_index(y_true, y_pred)
print("Global Jaccard-Concentration Index:", global_score)

# For detailed per-cluster results, use return_all=True
results = jaccard_concentration_index(y_true, y_pred, return_all=True, ordered_labels=["A", "B", "C"])
print("Detailed Results:")
for i, result in enumerate(results['cluster_results']):
    print(f"Cluster {i}:", result)

# Example using the concentration function directly
# Suppose we have a vector of Jaccard indices for a predicted cluster:
jaccard_values = np.array([0.2, 0.7, 0.1])
conc_score = concentration(jaccard_values)
print("Concentration Score:", conc_score)

# Using virtual_length to virtually add more zeros
conc_score_virtual = concentration(jaccard_values, virtual_length=6)
print("Concentration Score with virtual_length=6:", conc_score_virtual)
```

---

## Mathematical Details and Intuition

### Max Jaccard Index

For each predicted cluster $P$, the Max Jaccard Index is computed against every true cluster $T_i$:  
$MJI(P) = \max_i \left( \frac{|P \cap T_i|}{|P \cup T_i|} \right)$  
This score is high only when both the predicted cluster and a true cluster are of similar size and share the majority of the same points. By using this metric, the method avoids inflated scores for predicted clusters that, despite high purity, are only a tiny subset of a true cluster.

### Concentration Score

The concentration metric assesses how the mass of a predicted cluster is spread among the true clusters. Its algorithm is as follows:

1. **Uniform Contribution:**  
   Compute the uniform contribution for $n$ elements:  
   $UC = \frac{1}{n}$  
   This represents the baseline contribution of each element in a perfectly uniform distribution.
2. **Sum of Values:**  
   Compute the sum of the vector $v$:  
   $s = \sum_{i=1}^{n} v_i$
3. **Normalization:**  
   Normalize $v$:  
   $v = \left(\frac{v_1}{s}, \frac{v_2}{s}, \dots, \frac{v_n}{s}\right)$
4. **Squaring:**  
   Square the normalized values:  
   $v = \left(v_1^2, v_2^2, \dots, v_n^2\right)$
5. **Recompute Sum:**  
   Recompute the sum:  
   $s = \sum_{i=1}^{n} v_i$  
   Note that as the normalized values become larger (i.e., more mass is pooled in a few indices), the sum of squares $s$ grows larger. Conversely, $s$ shrinks as the distribution becomes more uniform.
6. **Compute Concentration:**  
   The concentration score is then computed based on the difference between $s$ and $UC$:
    - **Default Mode (`single_index=False`):**  
      $C = \sqrt{\frac{\sqrt{s} - \sqrt{UC}}{1 - \sqrt{UC}}}$  
      The square roots help to scale the final score to an intuitive range.
    - **Single Index Mode (`single_index=True`):**  
      First, set $m = \max_i v_i$  
      Then compute:  
      $C = \left(\frac{\frac{m}{s} - UC}{1 - UC}\right)^2$  
      This mode focuses on the disparity between the maximum value and the rest of the distribution. If the non-maximum values are small, then $\frac{m}{s}$ will be closer to 1, making the score approach 1. Squaring the result scales the score into a range comparable to the default mode.
  
   In both cases, the concentration score increases as the distribution becomes more focused. Both methods maximally reward distributions where all the mass is in one index and minimally reward uniform distributions.

- **Virtual Length:**  
  If you specify a virtual length larger than the actual length of $v$, the function treats $v$ as if it were padded with zeros to reach that length. This makes the nonzero values appear more concentrated relative to the full (virtual) length and can inflate the concentration score for experimental purposes.

- **Examples:**
  - **Uniform Case:**  
    For $v = (1, 1, 1, 1, 1)$ (with no virtual extension), the distribution is perfectly uniform, yielding a low concentration score.
  - **Concentrated Case:**  
    For $v = (0, 0, 1, 0, 0)$, the mass is entirely in one index, so the concentration score is 1.
  - **Virtual Length Effect:**  
    For $v = (1, 1, 1)$, the concentration is low when considered over 3 elements. However, if you set `virtual_length=6`, the same vector is treated as if it were $(1, 1, 1, 0, 0, 0)$; here, the mass is concentrated in 3 out of 6 positions, which boosts the concentration value.

### Jaccard-Concentration Index

The final per-cluster score is given by:  
$JCI(P) = \sqrt{MJI(P) \times C(P)}$  
This formulation ensures that if either the maximum overlap is low or the mass is spread across several true clusters, the overall score remains low. Only clusters that have both a high overlap (high $MJI$) and a focused distribution (high $C$) will achieve a high JCI.

### Macro-Averaging

Once individual cluster scores are computed, a macro-average can be calculated for each of the metrics (JCI, MJI, and $C$). The macro-average JCI serves as the global score for the clustering. Each cluster's weight in the macro-average is proportional to its size relative to the total number of points in the dataset. Mathematically, the macro-average of a cluster metric $M$ is computed as:  
$\text{MacroAvg}(M) = \sum_{i=1}^{n} M(P_i) \times \frac{|P_i|}{N}$  
where $P_i$ is a predicted cluster and $N$ is the total number of points in the dataset(excluding points in a noise cluster if present).

---

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or improvements, please file an issue or submit a pull request on [GitHub](https://github.com/RandyWAidoo/Jaccard-Concentration-Index).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

With the **Jaccard-Concentration Index**, you can obtain a more nuanced evaluation of your clustering algorithms-assessing not only whether predicted clusters match true clusters but also how focused the predicted clusters are.