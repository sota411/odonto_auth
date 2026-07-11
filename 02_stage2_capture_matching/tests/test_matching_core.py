from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import matching_core  # noqa: E402
from matching_core import (  # noqa: E402
    TOOTH_NAMES,
    MatchingResult,
    aggregate_embeddings,
    masked_square_crop,
    normalize_embedding,
    normalize_tooth_axis,
    score_pair,
)


class ToothTypesTest(unittest.TestCase):
    def test_tooth_types_have_the_required_fixed_order(self) -> None:
        self.assertEqual(TOOTH_NAMES, ("R1", "R2", "R3", "L1", "L2", "L3"))


class ExtractMaskedSquareTest(unittest.TestCase):
    def test_masks_background_and_adds_square_padding(self) -> None:
        image = np.full((4, 6, 3), (20, 40, 60), dtype=np.uint8)
        original = image.copy()
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:4] = True

        result = masked_square_crop(image, mask, output_size=4, padding_ratio=0.5)

        expected = np.zeros((4, 4, 3), dtype=np.uint8)
        expected[1:3, 1:3] = (20, 40, 60)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(image, original)

    def test_resizes_the_square_to_the_requested_size(self) -> None:
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[1, 1] = (10, 20, 30)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        result = masked_square_crop(image, mask, output_size=5, padding_ratio=0.0)

        self.assertEqual(result.shape, (5, 5, 3))
        self.assertEqual(result.dtype, image.dtype)
        np.testing.assert_array_equal(
            result,
            np.broadcast_to(np.array((10, 20, 30), dtype=np.uint8), (5, 5, 3)),
        )

    def test_rejects_an_empty_mask(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.zeros((2, 2), dtype=bool)

        with self.assertRaisesRegex(RuntimeError, "mask must contain"):
            masked_square_crop(image, mask, output_size=2)

    def test_rejects_image_and_mask_shape_mismatch(self) -> None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        with self.assertRaisesRegex(RuntimeError, "image and mask shapes"):
            masked_square_crop(image, mask, output_size=2)

    def test_rejects_invalid_output_sizes(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        for output_size in (0, -1, 1.5, True):
            with self.subTest(output_size=output_size):
                with self.assertRaises(RuntimeError):
                    masked_square_crop(image, mask, output_size=output_size)

    def test_rejects_invalid_image_or_mask_shapes_and_types(self) -> None:
        valid_image = np.zeros((2, 2, 3), dtype=np.uint8)
        valid_mask = np.ones((2, 2), dtype=bool)
        invalid_inputs = (
            (np.zeros((2, 2), dtype=np.uint8), valid_mask),
            (np.zeros((2, 2, 4), dtype=np.uint8), valid_mask),
            (valid_image, np.ones((2, 2), dtype=np.uint8)),
        )

        for image, mask in invalid_inputs:
            with self.subTest(image_shape=image.shape, mask_dtype=mask.dtype):
                with self.assertRaises(RuntimeError):
                    masked_square_crop(image, mask, output_size=2)

    def test_rejects_invalid_padding_ratios(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        for padding_ratio in (-0.1, np.nan, True):
            with self.subTest(padding_ratio=padding_ratio):
                with self.assertRaises(RuntimeError):
                    masked_square_crop(
                        image,
                        mask,
                        output_size=2,
                        padding_ratio=padding_ratio,
                    )


class NormalizeToothAxisTest(unittest.TestCase):
    def test_rotates_a_horizontal_mask_and_image_onto_the_vertical_axis(self) -> None:
        image = np.zeros((9, 9, 3), dtype=np.uint16)
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 2:7] = True
        image[4, 2:7] = np.array(
            ((10, 1, 2), (20, 3, 4), (30, 5, 6), (40, 7, 8), (50, 9, 10)),
            dtype=np.uint16,
        )

        rotated_image, rotated_mask = normalize_tooth_axis(image, mask)

        rotated_y, rotated_x = np.nonzero(rotated_mask)
        self.assertEqual(rotated_image.shape[:2], rotated_mask.shape)
        self.assertEqual(rotated_mask.shape[0], rotated_mask.shape[1])
        self.assertGreaterEqual(rotated_mask.shape[0], math.ceil(math.hypot(1, 5)))
        self.assertLess(rotated_mask.shape[0], image.shape[0])
        self.assertEqual(rotated_image.dtype, image.dtype)
        self.assertEqual(rotated_mask.dtype, mask.dtype)
        self.assertEqual(np.unique(rotated_x).size, 1)
        self.assertEqual(rotated_y.size, 5)
        np.testing.assert_array_equal(np.any(rotated_image != 0, axis=2), rotated_mask)
        np.testing.assert_array_equal(
            np.sort(rotated_image[rotated_mask], axis=0),
            np.sort(image[mask], axis=0),
        )

    def test_returns_a_local_square_even_when_the_mask_is_already_vertical(self) -> None:
        image = np.zeros((30, 40, 3), dtype=np.float32)
        mask = np.zeros((30, 40), dtype=bool)
        mask[5:15, 1:4] = True
        image[mask] = (1.0, 2.0, 3.0)

        rotated_image, rotated_mask = normalize_tooth_axis(image, mask)

        self.assertEqual(rotated_image.shape[:2], rotated_mask.shape)
        self.assertEqual(rotated_mask.shape[0], rotated_mask.shape[1])
        self.assertGreaterEqual(rotated_mask.shape[0], math.ceil(math.hypot(10, 3)))
        self.assertLess(rotated_mask.shape[0], min(image.shape[:2]))
        self.assertEqual(rotated_image.dtype, image.dtype)
        self.assertEqual(rotated_mask.dtype, mask.dtype)
        self.assertEqual(int(rotated_mask.sum()), int(mask.sum()))
        np.testing.assert_array_equal(rotated_image[0, 0], 0.0)

    def test_does_not_clip_a_diagonal_mask_at_the_image_edge(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        diagonal = np.arange(6)
        mask[diagonal, diagonal] = True
        image[diagonal, diagonal] = (20, 40, 60)

        rotated_image, rotated_mask = normalize_tooth_axis(image, mask)

        rotated_y, rotated_x = np.nonzero(rotated_mask)
        self.assertGreaterEqual(int(rotated_mask.sum()), int(mask.sum()))
        self.assertGreater(int(rotated_y.min()), 0)
        self.assertLess(int(rotated_y.max()), rotated_mask.shape[0] - 1)
        self.assertGreater(int(rotated_x.min()), 0)
        self.assertLess(int(rotated_x.max()), rotated_mask.shape[1] - 1)
        self.assertTrue(np.all(np.any(rotated_image[rotated_mask] != 0, axis=1)))
        corners = rotated_image[[0, 0, -1, -1], [0, -1, 0, -1]]
        np.testing.assert_array_equal(corners, np.zeros((4, 3), dtype=np.uint8))

    def test_limits_a_high_resolution_image_rotation_to_the_local_mask(self) -> None:
        image = np.zeros((3000, 4000, 3), dtype=np.uint8)
        mask = np.zeros((3000, 4000), dtype=bool)
        mask[1498:1502, 1980:2020] = True
        image[mask] = (10, 20, 30)
        original_coordinates = matching_core._rotation_source_coordinates

        def assert_local_coordinates(
            shape: tuple[int, int],
            center_y: float,
            center_x: float,
            rotation_angle: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            self.assertLessEqual(shape[0], 50)
            self.assertLessEqual(shape[1], 50)
            return original_coordinates(shape, center_y, center_x, rotation_angle)

        with patch.object(
            matching_core,
            "_rotation_source_coordinates",
            side_effect=assert_local_coordinates,
        ) as coordinate_mock:
            rotated_image, rotated_mask = normalize_tooth_axis(image, mask)

        coordinate_mock.assert_called_once()
        self.assertEqual(rotated_image.shape[:2], rotated_mask.shape)
        self.assertEqual(rotated_mask.shape[0], rotated_mask.shape[1])
        self.assertLessEqual(rotated_mask.shape[0], 50)
        self.assertEqual(rotated_image.dtype, image.dtype)
        self.assertEqual(rotated_mask.dtype, mask.dtype)

    def test_result_can_be_passed_to_masked_square_crop(self) -> None:
        image = np.zeros((7, 9, 3), dtype=np.uint8)
        mask = np.zeros((7, 9), dtype=bool)
        mask[3, 2:7] = True
        image[mask] = (10, 20, 30)

        rotated_image, rotated_mask = normalize_tooth_axis(image, mask)
        crop = masked_square_crop(
            rotated_image,
            rotated_mask,
            output_size=5,
            padding_ratio=0.0,
        )

        self.assertEqual(crop.shape, (5, 5, 3))
        self.assertEqual(crop.dtype, image.dtype)
        np.testing.assert_array_equal(
            crop[:, 2],
            np.broadcast_to(np.array((10, 20, 30), dtype=np.uint8), (5, 3)),
        )

    def test_rejects_an_empty_mask(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=bool)

        with self.assertRaisesRegex(RuntimeError, "mask must contain"):
            normalize_tooth_axis(image, mask)

    def test_rejects_an_isotropic_mask(self) -> None:
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        with self.assertRaisesRegex(RuntimeError, "principal axis is undefined"):
            normalize_tooth_axis(image, mask)


class L2NormalizeTest(unittest.TestCase):
    def test_normalizes_a_one_dimensional_embedding(self) -> None:
        embedding = np.array((3.0, 4.0))

        result = normalize_embedding(embedding)

        np.testing.assert_allclose(result, (0.6, 0.8))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_normalizes_finite_values_across_float64_range(self) -> None:
        for magnitude in (1e-300, 1e300):
            with self.subTest(magnitude=magnitude):
                result = normalize_embedding(np.array((magnitude, magnitude)))

                np.testing.assert_allclose(result, (1.0 / np.sqrt(2.0),) * 2)

    def test_rejects_invalid_embeddings(self) -> None:
        invalid_embeddings = (
            np.array(()),
            np.zeros(2),
            np.array((1.0, np.nan)),
            np.array((1.0, np.inf)),
            np.ones((1, 2)),
            np.array(1.0),
        )

        for embedding in invalid_embeddings:
            with self.subTest(shape=embedding.shape, embedding=embedding):
                with self.assertRaises(RuntimeError):
                    normalize_embedding(embedding)


class AggregateViewEmbeddingsTest(unittest.TestCase):
    def test_returns_a_normalized_confidence_weighted_average(self) -> None:
        embeddings = np.array(((1.0, 0.0), (0.0, 1.0)))
        confidences = np.array((1.0, 3.0))

        result = aggregate_embeddings(embeddings, confidences)

        np.testing.assert_allclose(result, (1.0 / np.sqrt(10.0), 3.0 / np.sqrt(10.0)))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_rejects_invalid_embedding_matrices(self) -> None:
        invalid_embeddings = (
            np.ones(2),
            np.empty((0, 2)),
            np.empty((2, 0)),
            np.array(((1.0, 0.0), (np.nan, 1.0))),
        )

        for embeddings in invalid_embeddings:
            with self.subTest(shape=embeddings.shape):
                with self.assertRaises(RuntimeError):
                    aggregate_embeddings(embeddings, np.ones(embeddings.shape[0]))

    def test_rejects_invalid_confidences(self) -> None:
        embeddings = np.eye(2)
        invalid_confidences = (
            np.ones(1),
            np.ones((2, 1)),
            np.array((1.0, 0.0)),
            np.array((1.0, -1.0)),
            np.array((1.0, np.nan)),
        )

        for confidences in invalid_confidences:
            with self.subTest(confidences=confidences):
                with self.assertRaises(RuntimeError):
                    aggregate_embeddings(embeddings, confidences)

    def test_rejects_a_zero_aggregate(self) -> None:
        embeddings = np.array(((1.0, 0.0), (-1.0, 0.0)))

        with self.assertRaisesRegex(RuntimeError, "norm must be positive"):
            aggregate_embeddings(embeddings, np.ones(2))

    def test_rejects_a_zero_view_embedding(self) -> None:
        embeddings = np.array(((1.0, 0.0), (0.0, 0.0)))

        with self.assertRaisesRegex(RuntimeError, "view embedding"):
            aggregate_embeddings(embeddings, np.ones(2))


class ComputeMatchingScoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.template_embeddings = np.zeros((6, 3), dtype=float)
        self.query_embeddings = np.zeros((6, 3), dtype=float)
        self.template_presence = np.array((True, True, False, True, False, False))
        self.query_presence = np.array((True, True, False, True, True, False))

        self.template_embeddings[0] = (2.0, 0.0, 0.0)
        self.query_embeddings[0] = (3.0, 0.0, 0.0)
        self.template_embeddings[1] = (1.0, 0.0, 0.0)
        self.query_embeddings[1] = (0.0, 4.0, 0.0)
        self.template_embeddings[3] = (1.0, 0.0, 0.0)
        self.query_embeddings[3] = (-5.0, 0.0, 0.0)
        self.query_embeddings[4] = (0.0, 0.0, 1.0)

    def test_scores_common_teeth_and_returns_their_simple_average(self) -> None:
        result = score_pair(
            template_embeddings=self.template_embeddings,
            query_embeddings=self.query_embeddings,
            template_present=self.template_presence,
            query_present=self.query_presence,
            min_common_teeth=3,
        )

        self.assertIsInstance(result, MatchingResult)
        self.assertEqual(result.common_teeth, ("R1", "R2", "L1"))
        self.assertEqual(list(result.per_tooth_scores), ["R1", "R2", "L1"])
        self.assertAlmostEqual(result.per_tooth_scores["R1"], 1.0)
        self.assertAlmostEqual(result.per_tooth_scores["R2"], 0.5)
        self.assertAlmostEqual(result.per_tooth_scores["L1"], 0.0)
        self.assertAlmostEqual(result.fused_score, 0.5)

    def test_rejects_too_few_common_teeth(self) -> None:
        query_presence = np.array((True, False, False, False, False, False))

        with self.assertRaisesRegex(RuntimeError, "common teeth"):
            score_pair(
                template_embeddings=self.template_embeddings,
                query_embeddings=self.query_embeddings,
                template_present=self.template_presence,
                query_present=query_presence,
                min_common_teeth=2,
            )

    def test_scores_extreme_finite_embeddings_without_overflow(self) -> None:
        template_embeddings = self.template_embeddings.copy()
        query_embeddings = self.query_embeddings.copy()
        template_embeddings[0] = (1e300, 1e300, 0.0)
        query_embeddings[0] = (1e300, 1e300, 0.0)
        presence = np.array((True, False, False, False, False, False))

        result = score_pair(
            template_embeddings=template_embeddings,
            query_embeddings=query_embeddings,
            template_present=presence,
            query_present=presence,
        )

        self.assertAlmostEqual(result.per_tooth_scores["R1"], 1.0)
        self.assertAlmostEqual(result.fused_score, 1.0)

    def test_rejects_invalid_embedding_shapes(self) -> None:
        invalid_pairs = (
            (np.zeros((5, 3)), self.query_embeddings),
            (self.template_embeddings, np.zeros((5, 3))),
            (self.template_embeddings, np.zeros((6, 2))),
            (np.zeros((6, 0)), np.zeros((6, 0))),
        )

        for template_embeddings, query_embeddings in invalid_pairs:
            with self.subTest(
                template_shape=template_embeddings.shape,
                query_shape=query_embeddings.shape,
            ):
                with self.assertRaises(RuntimeError):
                    score_pair(
                        template_embeddings=template_embeddings,
                        query_embeddings=query_embeddings,
                        template_present=self.template_presence,
                        query_present=self.query_presence,
                    )

    def test_rejects_invalid_presence_vectors(self) -> None:
        invalid_presence_vectors = (
            np.ones(5, dtype=bool),
            np.ones((6, 1), dtype=bool),
            np.ones(6, dtype=np.uint8),
        )

        for presence in invalid_presence_vectors:
            with self.subTest(shape=presence.shape, dtype=presence.dtype):
                with self.assertRaises(RuntimeError):
                    score_pair(
                        template_embeddings=self.template_embeddings,
                        query_embeddings=self.query_embeddings,
                        template_present=presence,
                        query_present=self.query_presence,
                    )

    def test_rejects_nonfinite_or_zero_common_embeddings(self) -> None:
        invalid_template_embeddings = self.template_embeddings.copy()
        invalid_template_embeddings[0, 0] = np.nan
        zero_query_embeddings = self.query_embeddings.copy()
        zero_query_embeddings[0] = 0.0

        for template_embeddings, query_embeddings in (
            (invalid_template_embeddings, self.query_embeddings),
            (self.template_embeddings, zero_query_embeddings),
        ):
            with self.subTest():
                with self.assertRaises(RuntimeError):
                    score_pair(
                        template_embeddings=template_embeddings,
                        query_embeddings=query_embeddings,
                        template_present=self.template_presence,
                        query_present=self.query_presence,
                    )

    def test_rejects_invalid_minimum_common_teeth(self) -> None:
        for min_common_teeth in (0, 7, 1.5, True):
            with self.subTest(min_common_teeth=min_common_teeth):
                with self.assertRaises(RuntimeError):
                    score_pair(
                        template_embeddings=self.template_embeddings,
                        query_embeddings=self.query_embeddings,
                        template_present=self.template_presence,
                        query_present=self.query_presence,
                        min_common_teeth=min_common_teeth,
                    )


if __name__ == "__main__":
    unittest.main()
