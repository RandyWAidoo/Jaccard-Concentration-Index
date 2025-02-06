from sklearn.utils.validation import check_array, check_consistent_length
import typing as tp
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

def concentration(
    values: np.ndarray[np.float64], single_index: bool = False, 
    size_invariance: bool = True, virtual_length: int = 0 
)->float:
    """Measure how concentrated the total value of a list of values is within one or a few indexes.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        The list of values.

    single_index : bool, optional, default=False
        If `True`, higher scores are given when the maximum value is significantly larger than all other values.
        For example, in this mode, `concentration([.1,.8,.1]) > concentration([0,.8,.2])`.
        If `False`, higher scores are given when value is concentrated in a minimal number of indices.
        In this case, `concentration([.1,.8,.1]) < concentration([0,.8,.2])`.

    size_invariance : bool, optional, default=True
        If `True`, the function will always return a score in the range [0, 1].
        If `False`, the function will always return a score in the range [`1/len(values)`, 1].

    virtual_length : int, optional, default=0
        Allows the computation to proceed as if `values` were of length 
        `virtual_length` without altering memory usage. 
        `virtual_length` must be greater than or equal to `len(values)`.

    Returns
    -------
    concentration : float
        A float in the specified range, where higher scores indicate more concentrated distributions.

    Intuition
    ---------
    - Concentration is computed based on the distribution of the normalized values of `values`.
    - Best case : All value is concentrated in one index, e.g., `values = [0, ..., 1, ..., 0]` (a sharp peak).
    - Worst case : Value is perfectly distributed across all indices, e.g., `values = [1/len(values), ..., 1/len(values)]` (flat, uniform).
    - Vector length affects the score:
        For example, `concentration([0.4, 0.6]) < concentration([0.4, 0.6, 0, 0])` 
        because smaller vectors have higher uniformity values(.5 when `len(values)==2` vs .25 when `len(values)==4`),
        making smaller vectors closer to being uniform than larger vectors with similar values.

    Examples
    --------
    >>> from jaccard_concentration_index import concentration
    >>> concentration([0, 1, 1, 0])
    0.6435942529055827
    """

    # Edge case
    if len(values) < 1:
        return 0.0

    # Get the length we will use for the array
    n: int = len(values)
    if virtual_length:
        if virtual_length < n:
            raise ValueError("virtual_length must be greater than or equal to the length of the values list")
        n = virtual_length
    
    # Edge case using the virtual_length-incorporated length
    if n < 2:
        return 1.0

    # Compute concentration
    values = np.abs(values, dtype=np.float64)
    s: np.float64 = np.sum(values, dtype=np.float64)
    if s <= 0:
        return 0.0
    
    score: float = 0.0
    
    #  Compute the worst-case(perfectly uniform) contribution(normalized value) for a list of size n.
    #   This will be used to make the minimum score 0 regardless of n; this is size invariance
    uniform_contribution: float = 1/n
    
    values /= s #Normalize v to put it in a fixed range of values that always add to 1
    values *= values #Squaring v makes the sum shrink when values are smaller and grow when they are larger
    if single_index:
        #Take the square of the difference between the max contribution(normalized value) 
        # of the indexes and the uniform contribution. Squaring it deflates the value to be more reasonable.
        # A smaller sum(meaning smaller original values) results in a relatively higher max, 
        # resulting in a higher score for when all values but the max are large
        score = float(np.max(values)/np.sum(values) - uniform_contribution)
        score /= (1 - uniform_contribution) #Before the final squaring, normalize the score by (1 - uniform contribution) to give it a max value of 1
        score *= score
    else:
        #Take the sqrt of the difference between sqrt(sum(v)) 
        # and sqrt(uniform contribution). 
        # The sqrts are used to inflate values to be more reasonable.
        # A larger sum results in a higher score, rewarding distributions that
        # distribute the total value in 1 or a few large chunks. 
        sqrt_uniform: float = uniform_contribution**0.5
        score = float(np.sqrt(np.sum(values)) - sqrt_uniform)
        score /= (1 - sqrt_uniform) #Before the final sqrt, normalize the score by (1 - sqrt(uniform contribution)) to give it a max value of 1
        score = float(np.sqrt(score))

    #  The score is now in the range [0,1].
    #   If size invariance is undesirable, shift the value into the range [uniform contribution, 1]
    if not size_invariance:
        range_min: float = uniform_contribution
        range_size: float = 1 - range_min
        score *= range_size
        score += range_min
    
    return score

def jaccard_concentration_index(
    y_true: np.ndarray[np.int_], y_pred: np.ndarray[np.int_], 
    return_all: bool = False, ordered_labels: tp.Sequence[tp.Any] = []
)->tp.Union[
    float,
    dict[
        tp.Literal['score', 'macroavg_max_jaccard_index', 'macroavg_concentration', 'cluster_results'],
        tp.Union[
            float,
            list[dict[
                tp.Literal[
                    'score', 
                    'max_jaccard_index', 'concentration',
                    'closest_label_index', 'closest_label'
                ],
                tp.Union[float, int, tp.Any]
            ]]
        ]
    ]
]:
    """Compute the Jaccard-Concentration Index for clustering evaluation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True cluster labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    return_all : bool, optional, default=False
        Whether to return all global metrics along with all metrics for every cluster or simply the global score.
    ordered_labels : sequence, optional, default=[]
        Ordered labels to assign to the clusters.

    Returns
    -------
    One of :
    - score : float
        - The global Jaccard-Concentration Index. This is returned when `return_all==False`.
    - detailed_results : dict
        - A dictionary with 'score', 'macroavg_max_jaccard_index', 'macroavg_concentration', and per-cluster results.
        This is returned when `return_all==True`.
    """

    # Validate inputs
    y_true, y_pred = check_array(y_true, ensure_2d=False), check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_true, y_pred)

    #Setup the contigency table and row and column sums
    contingency_table: np.ndarray[np.int_] = contingency_matrix(y_true, y_pred, dtype=np.int_)
    row_sums: np.ndarray[np.float64] = np.sum(contingency_table, axis=1, dtype=np.float64) 
    column_sums: np.ndarray[np.float64] = np.sum(contingency_table, axis=0, dtype=np.float64)

    #If we are returning all and ordered labels were provided, ensure they are the right length
    if return_all and len(ordered_labels):
        if len(ordered_labels) != contingency_table.shape[0]:
            raise ValueError("Length of ordered_labels must equal the number of true clusters.")

    #For each predicted cluster, calculate the max jaccard index between it and every true cluster.
    # Additionally compute the (non-single-index)concentration of its mass across the true clusters 
    # and use the 2 metrics to compute the final score the cluster.
    i: int
    j: int
    pred_cluster_results: list[dict[
        tp.Literal[
            'score', 
            'max_jaccard_index', 'concentration',
            'closest_label_index', 'closest_label'
        ],
        tp.Union[float, int, tp.Any]
    ]] = []
    for j in range(contingency_table.shape[1]):
        #Setup result variables
        max_jaccard_index: np.float64 = np.float64(-1.0)
        jaccard_idxs_with_true_clusters: np.ndarray[np.float64] = np.array([
            0.0 for _ in range(contingency_table.shape[0])
        ], dtype=np.float64)
        closest_label_idx: int = -1
        closest_label: tp.Any = None
        
        #Get the best jaccard index and closest label
        for i in range(contingency_table.shape[0]):
            #Get the jaccard index with this true cluster
            intersection: np.int_ = contingency_table[i][j]
            union: np.int_ = row_sums[i] + column_sums[j] - intersection
            jaccard_idx: np.float64 = (intersection/union).astype(np.float64)
            jaccard_idxs_with_true_clusters[i] = jaccard_idx
            
            #Set the max jaccard index and the closest label info
            if jaccard_idx > max_jaccard_index:
                max_jaccard_index = jaccard_idx
                if return_all:
                    closest_label_idx = i
                    if len(ordered_labels):
                        closest_label = ordered_labels[closest_label_idx]
        
        #Calculate the score, aka Jaccard-Concentration Index for this cluster
        c: float = concentration(jaccard_idxs_with_true_clusters)
        jci: float = float(np.sqrt(max_jaccard_index*c))

        #Save the cluster's results for later
        pred_cluster_results.append({
            "score": jci,
            "max_jaccard_index": float(max_jaccard_index), 
            "concentration": c, 
            "closest_label_index": closest_label_idx,
            "closest_label": closest_label
        })
    
    #Calculate the macroaverage JCI as the macroavg of all clusters' various scores, 
    # with each cluster's weighting determined by its proportion in the dataset
    macroavg_jci = sum([
        float(pred_cluster_results[j]['score']*column_sums[j]/len(y_true))
        for j in range(contingency_table.shape[1])
    ])

    #Return results
    if return_all:
        #Calculate macroaverages of MJI and Concentration in the same manner as the JCI
        macroavg_mji = sum([
            float(pred_cluster_results[j]['max_jaccard_index']*column_sums[j]/len(y_true))
            for j in range(contingency_table.shape[1])
        ])
        macroavg_c = sum([
            float(pred_cluster_results[j]['concentration']*column_sums[j]/len(y_true))
            for j in range(contingency_table.shape[1])
        ])
        
        #Return them along with results for each cluster
        return {
            'score': macroavg_jci, 
            'macroavg_max_jaccard_index': macroavg_mji, 
            'macroavg_concentration': macroavg_c, 
            'cluster_results': pred_cluster_results
        }
    
    return macroavg_jci