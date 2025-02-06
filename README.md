# Jaccard-Concentration Index  

**Jaccard-Concentration Index** is a Python library for evaluating clustering(or general classification) quality using a novel metric that combines the well-known Jaccard index with a custom concentration score. It provides a more nuanced view of cluster purity by not only considering the best match between a predicted and true cluster but also measuring how concentrated the predicted cluster’s mass is across the true clusters.  

## Features  

- **Max Jaccard Index ($MJI$):**  
  For each predicted cluster, this metric computes the highest Jaccard index with any true cluster. The Jaccard index is defined as:  
  $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$  
  A score of 1 indicates perfect overlap between the predicted and a true cluster, while lower scores indicate less overlap.  

- **Concentration Score($C$):**  
  This metric measures how unevenly the cluster’s total mass is distributed across true clusters. Specifically, it rewards distributions that generally 'chunk' the majority of their value in one or a few indices and punishes distributions that are close to uniformity. A concentration score of 0 indicates that the mass is uniformly distributed (low purity), whereas a score of 1 means that all the mass is concentrated in a single cluster (high purity). The score is computed by:  
  1. Normalizing the input vector (e.g., a list of Jaccard indices) so that its values sum to 1.  
  2. Squaring the normalized values to exaggerate relative differences between them.  
  3. Comparing the sum of the squared values to that of a uniform distribution over the vector’s length.  
  4. Optionally, increasing the “virtual length” (i.e., adding implicit zeros) will boost the concentration score because it will cause the available mass to appear to be distributed over relatively fewer indices.  

- **Jaccard-Concentration Index (JCI):**  
  For each predicted cluster, the JCI is computed as the geometric mean of the max Jaccard index and the concentration score:  
  $JCI = \sqrt{MJI \times C}$  
  This formulation ensures that both a high overlap and a focused distribution are necessary for a high overall score.  

- **Macro-Averaging:**  
  The library aggregates scores from each predicted cluster by computing a macro-average weighted by cluster size. This means that larger clusters contribute more to the final score, reducing the impact of small, noisy clusters.  

## Installation  

You can install the library using pip:  

```bash
pip install jaccard-concentration-index
```  

## Mathematical Details and Intuition  

### Max Jaccard Index  

For each predicted cluster $P$, the Max Jaccard Index is computed against every true cluster $T_i$:  
$MJI(P) = \max_i \left( \frac{|P \cap T_i|}{|P \cup T_i|} \right)$  
This score will only be high when both the predicted cluster and a true cluster are of a similar size and share a majority of the same points. By using this metric, we avoid naively inflated scores for predicted clusters with high purity that are actually a tiny subset of a true cluster.

### Concentration Score  

The concentration metric assesses how the mass of a predicted cluster is spread among the true clusters. Its algorithm is as follows:  

1. Computes the uniform contribution $\( UC = \frac{1}{n} \)$ for $n$ elements. 
2. Computes the sum of $v$: $s = \sum_{i=1}^{n} v_i$.
3. Normalizes $v$: $v = (\frac{v_1}{s}, ..., \frac{v_n}{s})$.
2. Squares $v$: $v = (v_1^2, ..., v_n^2)$.
3. Recomputes the sum of $v$: $s = \sum_{i=1}^{n} v_i$. Note that as normalized values grow larger(greater pooling of total value in a few indexes), their sum of squares, $s$, grows larger. Conversely, $s$ shrinks as the normalized values in $v$ shrink(bringing it closer to uniformity).
3. Computes concentration \( $C$ \) as some modification to the difference between $s$ and $UC$:
    - If in the default mode(`single_index=False`): 
      - $C = \sqrt{\frac{\sqrt{s} - \sqrt{UC}}{1 - \sqrt{UC}}}$. 
      - The use of square roots here inflates the final score to be slightly more reasonable and intuitive.
    - If `single_index=True`: 
      - Set $m = \max_i v_i$.
      - $C = \( \frac{\frac{m}{s} - UC}{1 - UC} \)^2$. 
      - Here, we use $s$ to renormalize $m$ and compute the difference between that and $UC$. This works because of the fact that as the normalized values in $v$ shrink, $s$ shrinks. This fact applies to the subset of $v$ without its maximum value as well. Thus, when $m$ is renormalized, if the original non-maximum values were small, $m$ will be closer to $s$. As the non-maximum values continue to shrink, $s$ shrinks towards $m$, $\frac{m}{s}$ grows towards 1, $\frac{m}{s} - UC$ grows towards $1 - UC$, and $\frac{\frac{m}{s} - UC}{1 - UC}$ grows towards 1. Finally, squaring the result deflates scores below 1 to make them more reasonable and puts them in a similar value range as the default mode scores. 


- **Virtual Length:**  
  If you specify a virtual length larger than the actual length of $v$, the function assumes the length of $v$ is the virtual length. This is equivalent to padding $v$ out with zeroes, which makes the nonzero values appear more concentrated relative to the full (virtual) length.  

- **Example:**  
  - **Uniform Case:**  
    For $v = (1, 1, 1, 1, 1)$ (with no virtual extension), the distribution is perfectly uniform, yielding a concentration near 0.  
  - **Concentrated Case:**  
    For $v = (0, 0, 1, 0, 0)$, the mass is entirely in one index, and the concentration will be 1.  
  - **Virtual Length Effect:**  
    For $v = (1, 1, 1)$, the concentration is 0 when considered over 3 elements. However, if you set `virtual_length=6`, the same vector is treated as if it were $(1, 1, 1, 0, 0, 0)$, and the mass is now concentrated in 3 out of 6 positions, boosting the concentration value.  

### Jaccard-Concentration Index (JCI)  

The final per-cluster score is given by:  
$JCI(P) = \sqrt{MJI(P) \times C(P)}$  
This ensures that if either the maximum overlap is low or the mass is spread out across several true clusters, the overall score remains low. Only clusters that have both a high match (high MJI) and a focused overlap (high concentration) will have a high JCI.  

## Usage Example  

Here’s a simple example of how to use the library:  

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

## Contributing  

Contributions are welcome! If you have suggestions, bug reports, or improvements, please file an issue or submit a pull request on [GitHub](https://github.com/RandyWAidoo/Jaccard-Concentration-Index.git).  

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

---  

With the **Jaccard-Concentration Index**, you can get a more nuanced evaluation of your clustering algorithms—assessing not only whether predicted clusters match true clusters but also how focused predicted clusters are. 