import unittest
import numpy as np
from jaccard_concentration_index import concentration, jaccard_concentration_index

# === Begin tests ===

class TestConcentration(unittest.TestCase):
    def test_empty_vector(self):
        # If v is empty, should return 0.
        v = np.array([], dtype=np.float64)
        self.assertEqual(concentration(v), 0.0)

    def test_single_element(self):
        # For a single element vector, the concentration is defined as 1.
        v = np.array([42.0], dtype=np.float64)
        self.assertEqual(concentration(v), 1.0)

    def test_all_zero_vector(self):
        # All zeros should yield a sum <= 0, hence 0 concentration.
        v = np.array([0, 0, 0], dtype=np.float64)
        self.assertEqual(concentration(v), 0.0)

    def test_uniform_distribution(self):
        # A perfectly uniform vector should yield a concentration near 0.
        v = np.array([1, 1, 1, 1, 1], dtype=np.float64)
        score = concentration(v)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_perfectly_concentrated(self):
        # A vector with all mass in one index should yield concentration near 1.
        v = np.array([0, 0, 1, 0, 0], dtype=np.float64)
        score = concentration(v)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_non_uniform_distribution(self):
        # Test with a distribution that is not uniform but not perfectly concentrated.
        v = np.array([0.1, 0.1, 0.6, 0.1, 0.1], dtype=np.float64)
        score = concentration(v)
        # We expect a score > 0 but less than 1.
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_single_index_mode(self):
        # Compare the effect of single_index mode.
        v1 = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        v2 = np.array([0, 0.8, 0.2], dtype=np.float64)
        score1 = concentration(v1, single_index=True)
        score2 = concentration(v2, single_index=True)
        # In single_index mode, v1 should have a higher score because the dominant index
        # is even more pronounced relative to the rest.
        self.assertGreater(score1, score2)

    def test_non_single_index_mode(self):
        # Compare the effect of non-single_index mode.
        v1 = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        v2 = np.array([0, 0.8, 0.2], dtype=np.float64)
        score1 = concentration(v1)
        score2 = concentration(v2)
        # In non-single_index mode, v2 should have a higher score because 
        # value is concentrated in fewer indexes than v1
        self.assertGreater(score2, score1)

    def test_size_invariance_flag(self):
        # When size_invariance is False, the minimum score shifts from 0 to 1/n.
        v = np.array([1, 1, 1, 1], dtype=np.float64)
        score_default = concentration(v, size_invariance=True)
        score_no_inv = concentration(v, size_invariance=False)
        # In a uniform distribution, the default score should be 0.
        self.assertAlmostEqual(score_default, 0.0, places=5)
        # With size invariance off, the score should be the uniform contribution: 1/n.
        self.assertAlmostEqual(score_no_inv, 1/4, places=5)

    def test_bad_virtual_length(self):
        # Should throw an error if virtual_length < len(values)
        v = np.array([1, 1, 1], dtype=np.float64)
        with self.assertRaises(ValueError):
            concentration(v, virtual_length=len(v) - 1)

    def test_virtual_length(self):
        # Increasing the virtual_length (i.e. virtually adding extra zeros)
        # should increase concentration because the same mass is concentrated in a smaller fraction of the total.
        # For a perfectly uniform vector of actual length 3, concentration is 0.
        v = np.array([1, 1, 1], dtype=np.float64)
        score_normal = concentration(v)  # here virtual_length defaults to len(v)=3
        # With virtual_length greater than the actual length, extra zeros are assumed.
        score_virtual = concentration(v, virtual_length=6)
        self.assertGreater(score_virtual, score_normal,
                           msg="Increasing virtual_length should increase concentration (by making the mass appear more concentrated)")


class TestJaccardConcentrationIndex(unittest.TestCase):
    def test_perfect_clustering(self):
        # If the predicted clustering exactly matches the true clustering,
        # each predicted cluster has a perfect Jaccard (1.0) with one true cluster.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        score = jaccard_concentration_index(y_true, y_pred)
        # In a perfect clustering, the macro-average JCI should be 1.
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_non_integer_clustering(self):
        # Test results for non-integer labels
        y_true = np.array(['a', 'a', 'b', 'b'])
        y_pred = np.array(['a', 'a', 'b', 'b'])
        score = jaccard_concentration_index(y_true, y_pred)
        # In a perfect clustering, the macro-average JCI should be 1.
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_complete_misclassification(self):
        # A scenario where predicted clusters are perfectly spread out across true clusters.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        score = jaccard_concentration_index(y_true, y_pred)
        # In this case, the concentration is 0, meaning the jci should be 0
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_single_true_cluster(self):
        # A scenario where there is only 1 true cluster
        y_true = np.array([0, 0, 0, 0])
        # Predicted clusters will split the true cluster in 2, leading a max jaccard idx of .5
        y_pred = np.array([0, 1, 0, 1])
        score = jaccard_concentration_index(y_true, y_pred)
        # In this case, concentration is 1 but mji is 0.5, so jci should be sqrt(0.5).
        # We also generally should not have encountered any errors on the way to this line.
        # One true cluster is allowed here, unlike with some evaluation metrics
        self.assertAlmostEqual(score, 0.5**0.5, places=5)

    def test_single_pred_cluster(self):
        # A scenario where there is only 1 predicted cluster and 2 true clusters.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        score = jaccard_concentration_index(y_true, y_pred)
        # In this case, the max jaccard index of the predicted cluster should be .5,
        # but the concentration should be 0, yielding a score of 0
        self.assertAlmostEqual(score, 0, places=5)

    def test_noisy_pred_clusters(self):
        # A scenario where there is a predicted noise cluster
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, -1, -1])
        score = jaccard_concentration_index(y_true, y_pred, noise_label=-1)
        # In this case, the max jaccard index of each non-noise cluster should be .25,
        # but the concentration should be 1, yielding a score(geometric mean) of .5 for each non-noise cluster,
        # meaning the avg should also be .5
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_all_noise_pred_clusters(self):
        # A scenario where there is all points are predicted as noise
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([-1, -1, -1, -1])
        with self.assertRaises(ValueError):
            jaccard_concentration_index(y_true, y_pred, noise_label=-1)

    def test_bad_noise_label(self):
        # A scenario where there is a predicted noise cluster and a wrong noise label as a parameter to jci
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, -1, -1])
        score = jaccard_concentration_index(y_true, y_pred, noise_label=-2)
        # In this case, the max jaccard index of the last pred cluster 
        # should be .5 while the first 2 will be .25. Then concentration of all of them should be 1.
        # So the scores should be .5,.5, and sqrt(.5), while sizes proportions are .25, .25, .5,
        # ultimately making the avg out to be .5*.25+.5*.25+sqrt(.5)*.5, which is ~0.60355
        self.assertAlmostEqual(score, 0.60355, places=5)

    def test_return_all_structure_with_ordered_labels(self):
        # Test that when return_all is True, the dictionary returned has the proper structure.
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        ordered_labels = ["A", "B", "C"]
        results = jaccard_concentration_index(y_true, y_pred, return_all=True, ordered_labels=ordered_labels)
        self.assertIsInstance(results, dict)
        for key in ['score', 'macroavg_max_jaccard_index', 'macroavg_concentration', 'cluster_results']:
            self.assertIn(key, results)
        # Also check that cluster_results is a list with one dict per predicted cluster.
        self.assertEqual(len(results['cluster_results']), len(np.unique(y_pred)))
        # Then check the structure of all cluster_results
        for key in [
            'score', 'max_jaccard_index', 'concentration', 
            'closest_label_index', 'closest_label', 'size_proportion'
        ]:
            for i, result in enumerate(results["cluster_results"]):
                self.assertIn(key, result)
                if key == "closest_label":
                    self.assertEqual(result[key], ordered_labels[i])
                elif key == "size_proportion":
                    self.assertAlmostEqual(result[key], 1/3, places=5)

    def test_return_all_with_wrong_ordered_labels(self):
        # Ensure an error occurrs if the length of the ordered labels is wrong.
        # This will be tested with 3 true classes and 4 ordered labels
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        ordered_labels = [1,2,3,4]
        with self.assertRaises(ValueError):
            jaccard_concentration_index(y_true, y_pred, return_all=True, ordered_labels=ordered_labels)

    def test_return_all_structure_no_ordered_labels(self):
        # Test that when return_all is True, the dictionary returned has the proper structure.
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        results = jaccard_concentration_index(y_true, y_pred, return_all=True)
        self.assertIsInstance(results, dict)
        for key in ['score', 'macroavg_max_jaccard_index', 'macroavg_concentration', 'cluster_results']:
            self.assertIn(key, results)
        # Also check that cluster_results is a list with one dict per predicted cluster.
        self.assertEqual(len(results['cluster_results']), len(np.unique(y_pred)))
        # Then check the structure of all cluster_results
        for key in [
            'score', 'max_jaccard_index', 'concentration', 
            'closest_label_index', 'closest_label', 'size_proportion'
        ]:
            for result in results["cluster_results"]:
                self.assertIn(key, result)

    def test_inconsistent_length(self):
        # Test that if y_true and y_pred have different lengths, an error is raised.
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        with self.assertRaises(ValueError):
            jaccard_concentration_index(y_true, y_pred)


if __name__ == '__main__':
    unittest.main()
