#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from presmoke.orchestrate_report import shard_examples


class OrchestrateShardingTest(unittest.TestCase):
    def test_sharding_keeps_same_source_case_on_one_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = write_plan_report(root, ["case/a", "case/b", "case/c", "case/d"])
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"case": "case/a", "source_case": "case/source"},
                        {"case": "case/b", "source_case": "case/source"},
                    ]
                ),
                encoding="utf-8",
            )

            assignments = dict(
                (example, card)
                for card, example in shard_examples(report, ["0", "1"], manifest)
            )

        self.assertEqual(assignments["case/a"], assignments["case/b"])

    def test_sharding_uses_duration_report_to_balance_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = write_plan_report(root, ["case/a", "case/b", "case/c"])
            timings = root / "ALL_CASE_TIMINGS.tsv"
            timings.write_text(
                "\n".join(
                    [
                        "run\texample\tarch\tmode\tstatus\trc\tduration_s",
                        "full\tcase/a\tdav-2201\tnpu\tPASS\t0\t1",
                        "full\tcase/b\tdav-2201\tnpu\tPASS\t0\t100",
                        "full\tcase/c\tdav-2201\tnpu\tPASS\t0\t1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            assignments = dict(
                (example, card)
                for card, example in shard_examples(
                    report, ["0", "1"], schedule_report=timings
                )
            )

        self.assertNotEqual(assignments["case/a"], assignments["case/b"])
        self.assertEqual(assignments["case/a"], assignments["case/c"])

    def test_sharding_uses_stage_timings_to_balance_pipeline_makespan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = write_plan_report(root, ["case/a", "case/b", "case/c"])
            timings = root / "ALL_CASE_TIMINGS.tsv"
            timings.write_text(
                "\n".join(
                    [
                        "example\tbuild_s\trun_s\tverify_s\tduration_s",
                        "case/a\t100\t1\t1\t102",
                        "case/b\t90\t1\t1\t92",
                        "case/c\t1\t1\t1\t3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            assignments = dict(
                (example, card)
                for card, example in shard_examples(
                    report, ["0", "1"], schedule_report=timings, jobs=2
                )
            )

        self.assertNotEqual(assignments["case/a"], assignments["case/b"])
        self.assertIn(assignments["case/c"], {"0", "1"})

    def test_sharding_interleaves_same_card_source_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = write_plan_report(
                root, ["case/a1", "case/a2", "case/b1", "case/b2"]
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"case": "case/a1", "source_case": "case/a"},
                        {"case": "case/a2", "source_case": "case/a"},
                        {"case": "case/b1", "source_case": "case/b"},
                        {"case": "case/b2", "source_case": "case/b"},
                    ]
                ),
                encoding="utf-8",
            )

            assignments = [
                example for _, example in shard_examples(report, ["0"], manifest)
            ]

        self.assertEqual(assignments, ["case/a1", "case/b1", "case/a2", "case/b2"])

    def test_sharding_can_split_source_groups_for_isolated_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = write_plan_report(root, ["case/a1", "case/a2"])
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"case": "case/a1", "source_case": "case/a"},
                        {"case": "case/a2", "source_case": "case/a"},
                    ]
                ),
                encoding="utf-8",
            )

            assignments = dict(
                (example, card)
                for card, example in shard_examples(
                    report, ["0", "1"], manifest, keep_source_groups=False
                )
            )

        self.assertNotEqual(assignments["case/a1"], assignments["case/a2"])


def write_plan_report(root: Path, examples: list[str]) -> Path:
    report = root / "report.json"
    report.write_text(
        json.dumps(
            {"results": [{"example": example, "duration_s": 0} for example in examples]}
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    unittest.main()
