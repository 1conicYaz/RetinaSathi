import unittest

from scripts.audit_messidor import referable_from_messidor_grade


class MessidorMappingTests(unittest.TestCase):
    def test_binary_external_endpoint_is_explicit(self):
        self.assertEqual([referable_from_messidor_grade(grade) for grade in range(4)], [False, False, True, True])

    def test_unknown_grade_fails_loudly(self):
        with self.assertRaises(ValueError):
            referable_from_messidor_grade(4)


if __name__ == "__main__":
    unittest.main()
