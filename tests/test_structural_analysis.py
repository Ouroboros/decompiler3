#!/usr/bin/env python3
'''Merge point selection of the HLIL structural analyzer.'''

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parent.parent))

from ir.hlil.structural_analysis import StructuralAnalyzer


class TestMergePoint(unittest.TestCase):
    def test_merge_is_the_post_dominator_not_a_block_one_branch_reaches(self):
        # QS410_08_00_END: 0 tests x != 0; 1 tests x != 1 and jumps to the shared block 2 when equal;
        # 3 tests x != 2; every path ends in block 6. Block 2 is reached from both branches of 0
        # but skipped when x == 2, so it is not the merge point.
        successors = {0: [1, 2], 1: [3, 2], 2: [6], 3: [4, 5], 4: [6], 5: [6], 6: []}
        analyzer = StructuralAnalyzer(len(successors), successors)
        self.assertEqual(analyzer.find_merge_point(0, 1, 2), 6)
        self.assertEqual(analyzer.find_merge_point(1, 3, 2), 6)

    def test_branch_leaving_the_loop_falls_back_to_the_nearest_common_block(self):
        # TestScenarioflagSet: inside loop 1..5, block 2 either enters the shared block 4 or tests
        # again at 3, which leaves the loop to 6; the branches only meet at the exit, so the merge
        # is the nearest block both reach.
        successors = {0: [1], 1: [2, 6], 2: [4, 3], 3: [4, 6], 4: [5], 5: [1], 6: []}
        analyzer = StructuralAnalyzer(len(successors), successors)
        self.assertIsNone(analyzer.ipdom[2])
        self.assertEqual(analyzer.find_merge_point(2, 4, 3), 4)


if __name__ == '__main__':
    unittest.main()
