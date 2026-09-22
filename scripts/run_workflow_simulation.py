#!/usr/bin/env python3
"""Run the RetinaSathi district workflow scenarios without MATLAB/SimEvents."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Parameters:
    name: str = "baseline_rural_clinic"
    seed: int = 26038
    minutes: int = 480
    arrival_rate_per_minute: float = 0.10
    camera_count: int = 1
    camera_minutes: float = 6.0
    ai_device_count: int = 1
    # Conservative end-to-end AI-station service allowance. The measured V3.4
    # warm MATLAB model inference is 0.636 s; this also budgets handling/overhead.
    ai_minutes: float = 1.2
    reviewer_count: int = 1
    review_minutes: float = 8.0
    network_minutes: float = 1.5
    poor_quality_rate: float = 0.12
    referable_rate: float = 0.28
    uncertain_rate: float = 0.10
    priority_review: bool = False
    working_days_per_year: int = 250


def exponential_arrivals(rng: random.Random, rate: float, horizon: float) -> list[float]:
    arrivals: list[float] = []
    time = 0.0
    while True:
        time += rng.expovariate(rate)
        if time > horizon:
            return arrivals
        arrivals.append(time)


def take_server(servers: list[float], ready_at: float, duration: float) -> tuple[float, float]:
    available = heapq.heappop(servers)
    start = max(ready_at, available)
    finish = start + duration
    heapq.heappush(servers, finish)
    return start, finish


def peak_waiting(intervals: list[tuple[float, float]]) -> int:
    events = [(start, 1) for start, finish in intervals if finish > start]
    events += [(finish, -1) for start, finish in intervals if finish > start]
    current = peak = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        current += change
        peak = max(peak, current)
    return peak


def simulate(parameters: Parameters) -> dict[str, float | int | str | bool]:
    rng = random.Random(parameters.seed)
    arrivals = exponential_arrivals(rng, parameters.arrival_rate_per_minute, parameters.minutes)
    cameras = [0.0] * parameters.camera_count
    ai_devices = [0.0] * parameters.ai_device_count
    reviewers = [0.0] * parameters.reviewer_count
    heapq.heapify(cameras); heapq.heapify(ai_devices); heapq.heapify(reviewers)
    review_cases: list[tuple[float, int, float, bool, bool]] = []
    completed_waits: list[float] = []
    priority_waits: list[float] = []
    wait_intervals: list[tuple[float, float]] = []
    retakes = 0
    camera_busy = ai_busy = reviewer_busy = network_delay = 0.0

    for sequence, arrival in enumerate(arrivals):
        capture_start, capture_finish = take_server(cameras, arrival, parameters.camera_minutes)
        wait_intervals.append((arrival, capture_start))
        camera_busy += parameters.camera_minutes
        if rng.random() < parameters.poor_quality_rate:
            retakes += 1
            continue
        ai_ready = capture_finish + parameters.network_minutes
        network_delay += parameters.network_minutes
        ai_start, ai_finish = take_server(ai_devices, ai_ready, parameters.ai_minutes)
        wait_intervals.append((ai_ready, ai_start))
        ai_busy += parameters.ai_minutes
        referable = rng.random() < parameters.referable_rate
        uncertain = rng.random() < parameters.uncertain_rate
        if referable or uncertain:
            review_cases.append((ai_finish, sequence, arrival, referable, uncertain))
        elif ai_finish <= parameters.minutes:
            completed_waits.append(ai_finish - arrival)

    review_cases.sort()
    referrals = 0
    pending: list[tuple[float, int, float, bool, bool]] = []
    while review_cases or pending:
        available = heapq.heappop(reviewers)
        while review_cases and review_cases[0][0] <= available:
            pending.append(review_cases.pop(0))
        if not pending:
            available = max(available, review_cases[0][0])
            while review_cases and review_cases[0][0] <= available:
                pending.append(review_cases.pop(0))
        if parameters.priority_review:
            selected = min(range(len(pending)), key=lambda index: (not pending[index][3], not pending[index][4], pending[index][0]))
        else:
            selected = min(range(len(pending)), key=lambda index: (pending[index][0], pending[index][1]))
        ready, _, arrival, referable, _ = pending.pop(selected)
        review_start = max(available, ready)
        finish = review_start + parameters.review_minutes
        heapq.heappush(reviewers, finish)
        wait_intervals.append((ready, review_start))
        reviewer_busy += parameters.review_minutes
        if finish <= parameters.minutes:
            completed_waits.append(finish - arrival)
            referrals += int(referable)
            if referable:
                priority_waits.append(finish - arrival)

    processed = len(completed_waits)
    unfinished = len(arrivals) - processed - retakes
    maximum_queue = peak_waiting(wait_intervals)
    annual = processed * parameters.working_days_per_year
    return {
        "name": parameters.name,
        "seed": parameters.seed,
        "arrivals": len(arrivals),
        "patients_processed_per_day": processed,
        "annual_throughput": annual,
        "average_waiting_minutes": round(sum(completed_waits) / max(processed, 1), 2),
        "maximum_queue": maximum_queue,
        "camera_utilization": round(min(camera_busy / (parameters.minutes * parameters.camera_count), 1), 4),
        "ai_utilization": round(min(ai_busy / (parameters.minutes * parameters.ai_device_count), 1), 4),
        "reviewer_utilization": round(min(reviewer_busy / (parameters.minutes * parameters.reviewer_count), 1), 4),
        "referral_count": referrals,
        "retake_count": retakes,
        "network_delay_contribution_minutes": round(network_delay, 2),
        "priority_average_waiting_minutes": round(sum(priority_waits) / max(len(priority_waits), 1), 2),
        "unfinished": unfinished,
        "priority_review": parameters.priority_review,
    }


def scenario_parameters() -> list[Parameters]:
    base = Parameters()
    return [
        base,
        replace(base, name="network_constrained", network_minutes=8.0),
        replace(base, name="increased_patient_load", seed=26040, arrival_rate_per_minute=0.22),
        replace(base, name="additional_camera", seed=26040, arrival_rate_per_minute=0.22, camera_count=2),
        replace(base, name="additional_reviewer", seed=26040, arrival_rate_per_minute=0.22, reviewer_count=2),
        replace(base, name="priority_review", seed=26040, arrival_rate_per_minute=0.22, camera_count=2, review_minutes=12.0, referable_rate=0.55, uncertain_rate=0.15, priority_review=True),
        replace(
            base,
            name="district_100k_annual",
            seed=26044,
            minutes=600,
            arrival_rate_per_minute=500 / 600,
            camera_count=6,
            ai_device_count=2,
            reviewer_count=4,
        ),
    ]


def render_markdown(results: list[dict[str, object]]) -> str:
    rows = [
        "| Scenario | Processed/day | Annual | Avg wait (min) | Referable wait | Max queue | Camera | AI | Reviewer | Retakes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        rows.append(
            f"| {str(item['name']).replace('_', ' ')} | {item['patients_processed_per_day']} | "
            f"{item['annual_throughput']} | {item['average_waiting_minutes']} | {item['priority_average_waiting_minutes']} | {item['maximum_queue']} | "
            f"{100*float(item['camera_utilization']):.1f}% | {100*float(item['ai_utilization']):.1f}% | "
            f"{100*float(item['reviewer_utilization']):.1f}% | {item['retake_count']} |"
        )
    district = next(item for item in results if item["name"] == "district_100k_annual")
    priority = next(item for item in results if item["name"] == "priority_review")
    return "\n".join(
        [
            "# Workflow Simulation Results",
            "",
            "These are reproducible engineering estimates from `scripts/run_workflow_simulation.py`, the Python parity implementation of `simulink/simulateWorkflow.m`. They are not clinical outcomes or measured field performance. Fixed random seeds are recorded in the JSON output.",
            "",
            *rows,
            "",
            "## Interpretation",
            "",
            "- The increased-load scenario exposes camera saturation; the same-seed additional-camera comparison tests whether capture capacity improves throughput.",
            "- The network-constrained scenario isolates added transfer delay.",
            f"- In a paired reviewer-bottleneck run, priority routing reduced referable-case average wait by **{priority.get('priority_wait_reduction_minutes', 'not measured')} minutes** without adding a reviewer.",
            f"- The configured district scenario estimates **{district['annual_throughput']:,} completed screenings/year** across 250 working days under its assumptions.",
            "- Maximum queue is calculated from all recorded camera, AI, and reviewer waiting intervals; MATLAB parity remains to be executed once MATLAB is installed.",
            "",
            "## Execution status",
            "",
            "Python parity execution: COMPLETE. MATLAB/Simulink execution and `.slx` generation: TO VALIDATE because MATLAB is not installed on the audited Mac.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("simulink/results"))
    parser.add_argument("--docs-output", type=Path, default=Path("docs/SIMULATION_RESULTS.md"))
    args = parser.parse_args()
    results = [simulate(item) for item in scenario_parameters()]
    priority_config = scenario_parameters()[5]
    priority_control = simulate(replace(priority_config, name="priority_review_control", priority_review=False))
    results[5]["priority_control_referable_waiting_minutes"] = priority_control["priority_average_waiting_minutes"]
    results[5]["priority_wait_reduction_minutes"] = round(
        float(priority_control["priority_average_waiting_minutes"]) - float(results[5]["priority_average_waiting_minutes"]), 2
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_output.parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenarios.json").write_text(json.dumps(results, indent=2) + "\n")
    fieldnames = list(dict.fromkeys(key for result in results for key in result))
    with (args.output_dir / "scenarios.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(results)
    args.docs_output.write_text(render_markdown(results))
    print(json.dumps({"scenarios": len(results), "district_annual": results[-1]["annual_throughput"]}))


if __name__ == "__main__":
    main()
