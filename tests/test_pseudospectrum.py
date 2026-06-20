import unittest

import numpy as np

from pseudospectrum import (
    add_mpos,
    clifford_generators,
    clifford_pseudovalue,
    clifford_pseudovalue_dense,
    clifford_pseudovalue_sparse,
    clifford_pseudovalue_tensor,
    dense_localizer,
    dense_matrix_to_mpo,
    epsilon_mask,
    grid_values,
    localizer_mpo,
    mpo_to_dense,
    product_mpos,
    quadratic_mpo,
    quadratic_pseudovalue,
    quadratic_pseudovalue_dense,
    quadratic_pseudovalue_sparse,
    validate_clifford,
)
from pseudospectrum.examples import (
    hemisphere_polynomial,
    projection_pair,
    projection_pair_clifford_fixed_z,
    projection_pair_quadratic_fixed_z,
    shift_region_function,
    shift_triple_dense,
    universal_order_two_clifford_fixed_z,
    universal_order_two_pair,
    universal_order_two_quadratic_fixed_z,
)


class CliffordGeneratorTests(unittest.TestCase):
    def test_generators_validate_for_small_dimensions(self):
        for dimension in range(1, 6):
            gammas = clifford_generators(dimension)
            self.assertTrue(validate_clifford(gammas))
            self.assertEqual(len(gammas), dimension)

    def test_complex_dtype_is_required(self):
        with self.assertRaises(ValueError):
            clifford_generators(2, dtype=float)


class DensePseudospectrumTests(unittest.TestCase):
    def test_two_variable_localizer_matches_single_operator_pseudospectrum(self):
        a1 = np.array([[0.0, 1.0], [1.0, 0.0]])
        a2 = np.array([[1.0, 0.0], [0.0, -2.0]])
        point = (0.2, -0.4)
        combined = a1 + 1j * a2
        target = np.linalg.svd(
            combined - (point[0] + 1j * point[1]) * np.eye(2),
            compute_uv=False,
        )[-1]
        self.assertAlmostEqual(clifford_pseudovalue_dense([a1, a2], point), target)

    def test_universal_order_two_closed_forms_match_dense(self):
        z = 0.35
        point = (0.25, -0.7)
        operators = universal_order_two_pair(z)
        self.assertAlmostEqual(
            clifford_pseudovalue_dense(operators, point),
            universal_order_two_clifford_fixed_z(*point, z),
        )
        self.assertAlmostEqual(
            quadratic_pseudovalue_dense(operators, point),
            universal_order_two_quadratic_fixed_z(*point, z),
        )

    def test_projection_closed_forms_match_dense(self):
        z = 0.4
        point = (0.3, 0.8)
        operators = projection_pair(z)
        self.assertAlmostEqual(
            clifford_pseudovalue_dense(operators, point),
            projection_pair_clifford_fixed_z(*point, z),
        )
        self.assertAlmostEqual(
            quadratic_pseudovalue_dense(operators, point),
            projection_pair_quadratic_fixed_z(*point, z),
        )

    def test_shift_triple_has_expected_boundary_terms(self):
        a1, a2, a3 = shift_triple_dense(4, b=2.0)
        self.assertAlmostEqual(a1[0, 1], 0.5)
        self.assertAlmostEqual(a1[1, 0], 0.5)
        self.assertAlmostEqual(a2[0, 1], -0.5j)
        self.assertAlmostEqual(a2[1, 0], 0.5j)
        self.assertAlmostEqual(a3[0, 0], 2.0)

    def test_shift_polynomials_reduce_to_appendix_case(self):
        self.assertAlmostEqual(
            hemisphere_polynomial(0.3, 0.0, 0.4, b=1.0),
            0.3**4 - 0.4**4 + 3 * 0.4**3 - 4 * 0.4**2 + 3 * 0.4 - 1,
        )
        self.assertAlmostEqual(
            shift_region_function(0.3, 0.4, b=1.0),
            0.4 * 0.3**2 - 0.3 * 0.4 + 0.3 + 0.4**3 - 2 * 0.4**2 + 2 * 0.4 - 1,
        )

    def test_non_hermitian_inputs_raise_by_default(self):
        bad = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        with self.assertRaises(ValueError):
            clifford_pseudovalue_dense([bad], (0.0,))
        self.assertAlmostEqual(
            clifford_pseudovalue_dense([bad], (0.0,), assume_hermitian=False),
            0.0,
        )

    def test_backend_dispatch_and_grid_helpers(self):
        ops = universal_order_two_pair(0.1)
        point = (0.2, 0.3)
        self.assertAlmostEqual(
            clifford_pseudovalue(ops, point, backend="dense"),
            clifford_pseudovalue_dense(ops, point),
        )
        self.assertAlmostEqual(
            quadratic_pseudovalue(ops, point, backend="dense"),
            quadratic_pseudovalue_dense(ops, point),
        )

        meshes, values = grid_values(
            lambda p: clifford_pseudovalue_dense(ops, p),
            [np.array([0.0, 0.5]), np.array([-0.25, 0.25])],
        )
        self.assertEqual(values.shape, (2, 2))
        self.assertEqual(len(meshes), 2)
        self.assertEqual(epsilon_mask(values, 0.5).shape, values.shape)
        with self.assertRaises(ValueError):
            epsilon_mask(values, -1.0)
        with self.assertRaises(ValueError):
            grid_values(lambda p: 0.0, [])


class OptionalSparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import scipy.sparse  # noqa: F401
        except Exception:
            raise unittest.SkipTest("SciPy is not installed")

    def test_sparse_backend_matches_dense_on_small_example(self):
        ops = universal_order_two_pair(0.25)
        point = (0.2, -0.4)
        self.assertAlmostEqual(
            clifford_pseudovalue_sparse(ops, point),
            clifford_pseudovalue_dense(ops, point),
        )
        self.assertAlmostEqual(
            quadratic_pseudovalue_sparse(ops, point),
            quadratic_pseudovalue_dense(ops, point),
        )


class TensorNetworkTests(unittest.TestCase):
    def _two_site_local(self, left, right):
        return [
            np.asarray(left, dtype=complex).reshape(1, 1, 2, 2),
            np.asarray(right, dtype=complex).reshape(1, 1, 2, 2),
        ]

    def test_to_mpo_objects_are_accepted(self):
        class LazyOneSite:
            def __init__(self, matrix):
                self.matrix = matrix

            def to_mpo(self):
                return dense_matrix_to_mpo(self.matrix)

        a1, a2 = universal_order_two_pair(0.2)
        point = (0.1, -0.3)
        value, _ = clifford_pseudovalue_tensor([LazyOneSite(a1), LazyOneSite(a2)], point)
        self.assertAlmostEqual(value, clifford_pseudovalue_dense([a1, a2], point))

    def test_one_site_mpo_localizer_matches_dense_localizer(self):
        a1, a2 = universal_order_two_pair(0.2)
        point = (0.1, -0.3)
        dense = dense_localizer([a1, a2], point)
        mpo = localizer_mpo([dense_matrix_to_mpo(a1), dense_matrix_to_mpo(a2)], point)
        np.testing.assert_allclose(mpo_to_dense(mpo), dense, atol=1e-12)

    def test_two_site_open_boundary_sum_and_product_match_dense(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        z = np.array([[1.0, 0.0], [0.0, -1.0]])
        identity = np.eye(2)
        op_x0 = self._two_site_local(x, identity)
        op_z1 = self._two_site_local(identity, z)

        summed = add_mpos([op_x0, op_z1])
        product = product_mpos([op_x0, op_z1])
        np.testing.assert_allclose(
            mpo_to_dense(summed),
            np.kron(x, identity) + np.kron(identity, z),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mpo_to_dense(product),
            np.kron(x, identity) @ np.kron(identity, z),
            atol=1e-12,
        )

    def test_two_site_localizer_and_quadratic_mpo_match_dense(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        z = np.array([[1.0, 0.0], [0.0, -1.0]])
        identity = np.eye(2)
        op_x0 = self._two_site_local(x, identity)
        op_z1 = self._two_site_local(identity, z)
        dense_ops = [np.kron(x, identity), np.kron(identity, z)]
        point = (0.2, -0.1)

        np.testing.assert_allclose(
            mpo_to_dense(localizer_mpo([op_x0, op_z1], point)),
            dense_localizer(dense_ops, point),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mpo_to_dense(quadratic_mpo([op_x0, op_z1], point)),
            sum((op - coord * np.eye(4)) @ (op - coord * np.eye(4))
                for op, coord in zip(dense_ops, point)),
            atol=1e-12,
        )

    def test_tensor_pseudovalue_uses_exact_mpo_for_small_system(self):
        a1, a2 = universal_order_two_pair(-0.5)
        point = (0.25, 0.15)
        value, info = clifford_pseudovalue_tensor(
            [dense_matrix_to_mpo(a1), dense_matrix_to_mpo(a2)],
            point,
        )
        self.assertEqual(info.backend, "dense-mpo")
        self.assertAlmostEqual(value, clifford_pseudovalue_dense([a1, a2], point))


if __name__ == "__main__":
    unittest.main()
