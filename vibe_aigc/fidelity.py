"""Fidelity Measurement — Creative Unit Tests for vibe-aigc.

Paper Section 6: "The Verification Crisis... no universal unit test for a 'cinematic atmosphere'"
Paper Section 7: "We need 'Creative Unit Tests'"

This module measures how well vibe-aigc achieves user intent:
1. Intent Alignment: Does output match the vibe?
2. Consistency: Same prompt → similar results?
3. Quality Distribution: What's the score spread?
4. Refinement Efficacy: Does feedback improve scores?
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from .vibe_backend import VibeBackend, GenerationRequest, GenerationResult
from .discovery import Capability
from .vlm_feedback import VLMFeedback, FeedbackResult


@dataclass
class FidelityScore:
    """Score for a single generation."""
    prompt: str
    output_url: str
    quality_score: float
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    attempt_number: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "output_url": self.output_url,
            "quality_score": self.quality_score,
            "feedback": self.feedback,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp
        }


@dataclass
class FidelityReport:
    """Complete fidelity report for a prompt."""
    prompt: str
    capability: str
    num_runs: int
    scores: List[FidelityScore]
    
    # Statistics
    mean_score: float = 0.0
    std_dev: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    # Refinement analysis
    first_attempt_mean: float = 0.0
    refined_attempts_mean: float = 0.0
    refinement_improvement: float = 0.0
    
    # Common patterns
    common_strengths: List[str] = field(default_factory=list)
    common_weaknesses: List[str] = field(default_factory=list)
    
    def compute_statistics(self) -> None:
        """Compute statistics from scores."""
        if not self.scores:
            return
        
        quality_scores = [s.quality_score for s in self.scores]
        
        self.mean_score = statistics.mean(quality_scores)
        self.std_dev = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
        self.min_score = min(quality_scores)
        self.max_score = max(quality_scores)
        
        # Refinement analysis
        first_attempts = [s.quality_score for s in self.scores if s.attempt_number == 1]
        refined_attempts = [s.quality_score for s in self.scores if s.attempt_number > 1]
        
        if first_attempts:
            self.first_attempt_mean = statistics.mean(first_attempts)
        if refined_attempts:
            self.refined_attempts_mean = statistics.mean(refined_attempts)
            self.refinement_improvement = self.refined_attempts_mean - self.first_attempt_mean
        
        # Common patterns
        all_strengths = []
        all_weaknesses = []
        for s in self.scores:
            all_strengths.extend(s.strengths)
            all_weaknesses.extend(s.weaknesses)
        
        # Count frequency
        from collections import Counter
        strength_counts = Counter(all_strengths)
        weakness_counts = Counter(all_weaknesses)
        
        self.common_strengths = [s for s, _ in strength_counts.most_common(5)]
        self.common_weaknesses = [w for w, _ in weakness_counts.most_common(5)]
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "FIDELITY REPORT",
            "=" * 60,
            "",
            f"Prompt: {self.prompt[:50]}...",
            f"Capability: {self.capability}",
            f"Runs: {self.num_runs}",
            "",
            "QUALITY SCORES:",
            f"  Mean: {self.mean_score:.2f}/10",
            f"  Std Dev: {self.std_dev:.2f}",
            f"  Range: {self.min_score:.1f} - {self.max_score:.1f}",
            "",
            "REFINEMENT EFFICACY:",
            f"  First attempt mean: {self.first_attempt_mean:.2f}",
            f"  Refined attempts mean: {self.refined_attempts_mean:.2f}",
            f"  Improvement: {self.refinement_improvement:+.2f}",
            "",
            "COMMON STRENGTHS:",
        ]
        for s in self.common_strengths[:3]:
            lines.append(f"  + {s}")
        
        lines.append("")
        lines.append("COMMON WEAKNESSES:")
        for w in self.common_weaknesses[:3]:
            lines.append(f"  - {w}")
        
        lines.append("")
        lines.append("=" * 60)
        
        # Verdict
        if self.mean_score >= 7.0:
            lines.append("VERDICT: HIGH FIDELITY - System achieves intent well")
        elif self.mean_score >= 5.0:
            lines.append("VERDICT: MODERATE FIDELITY - Room for improvement")
        else:
            lines.append("VERDICT: LOW FIDELITY - Significant gap from intent")
        
        if self.refinement_improvement > 0.5:
            lines.append(f"REFINEMENT: EFFECTIVE (+{self.refinement_improvement:.1f} improvement)")
        elif self.refinement_improvement < -0.5:
            lines.append(f"REFINEMENT: COUNTERPRODUCTIVE ({self.refinement_improvement:.1f})")
        else:
            lines.append("REFINEMENT: MARGINAL EFFECT")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "capability": self.capability,
            "num_runs": self.num_runs,
            "scores": [s.to_dict() for s in self.scores],
            "statistics": {
                "mean": self.mean_score,
                "std_dev": self.std_dev,
                "min": self.min_score,
                "max": self.max_score,
            },
            "refinement": {
                "first_attempt_mean": self.first_attempt_mean,
                "refined_mean": self.refined_attempts_mean,
                "improvement": self.refinement_improvement,
            },
            "patterns": {
                "common_strengths": self.common_strengths,
                "common_weaknesses": self.common_weaknesses,
            }
        }


class FidelityBenchmark:
    """Benchmark for measuring vibe-aigc fidelity.
    
    Usage:
        benchmark = FidelityBenchmark(comfyui_url="http://192.168.1.143:8188")
        await benchmark.initialize()
        
        report = await benchmark.run(
            prompt="cyberpunk samurai in neon rain",
            capability=Capability.TEXT_TO_IMAGE,
            num_runs=5
        )
        
        print(report.summary())
    """
    
    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        max_attempts_per_run: int = 2,
        quality_threshold: float = 7.0
    ):
        self.backend = VibeBackend(
            comfyui_url=comfyui_url,
            enable_vlm=True,
            max_attempts=max_attempts_per_run,
            quality_threshold=quality_threshold
        )
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the benchmark."""
        await self.backend.initialize()
        self._initialized = True
    
    async def run(
        self,
        prompt: str,
        capability: Capability = Capability.TEXT_TO_IMAGE,
        num_runs: int = 5,
        **kwargs
    ) -> FidelityReport:
        """Run the fidelity benchmark.
        
        Args:
            prompt: The prompt to test
            capability: What to generate
            num_runs: How many times to run
            **kwargs: Additional generation parameters
        
        Returns:
            FidelityReport with scores and statistics
        """
        if not self._initialized:
            await self.initialize()
        
        print(f"Running fidelity benchmark: {num_runs} runs")
        print(f"Prompt: {prompt[:50]}...")
        print()
        
        scores = []
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...")
            
            request = GenerationRequest(
                prompt=prompt,
                capability=capability,
                **kwargs
            )
            
            result = await self.backend.generate(request)
            
            if result.success:
                score = FidelityScore(
                    prompt=prompt,
                    output_url=result.output_url or "",
                    quality_score=result.quality_score or 5.0,
                    feedback=result.feedback or "",
                    strengths=result.strengths or [],
                    weaknesses=result.weaknesses or [],
                    attempt_number=result.attempts,
                    timestamp=datetime.now().isoformat()
                )
                scores.append(score)
                print(f"  Score: {score.quality_score}/10 (attempt {score.attempt_number})")
                if score.strengths:
                    print(f"    Strengths: {', '.join(score.strengths[:2])}")
                if score.weaknesses:
                    print(f"    Weaknesses: {', '.join(score.weaknesses[:2])}")
            else:
                print(f"  Failed: {result.error}")
        
        # Build report
        report = FidelityReport(
            prompt=prompt,
            capability=capability.value,
            num_runs=num_runs,
            scores=scores
        )
        report.compute_statistics()
        
        return report
    
    async def compare_prompts(
        self,
        prompts: List[str],
        capability: Capability = Capability.TEXT_TO_IMAGE,
        runs_per_prompt: int = 3
    ) -> List[FidelityReport]:
        """Compare fidelity across multiple prompts."""
        reports = []
        
        for prompt in prompts:
            report = await self.run(prompt, capability, runs_per_prompt)
            reports.append(report)
        
        return reports
    
    async def test_refinement_efficacy(
        self,
        prompt: str,
        capability: Capability = Capability.TEXT_TO_IMAGE,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Specifically test if VLM refinement improves quality.
        
        Runs with max_attempts=1 (no refinement) vs max_attempts=3 (with refinement)
        """
        print("Testing refinement efficacy...")
        print()
        
        # Without refinement
        print("Phase 1: Without refinement (max_attempts=1)")
        self.backend.max_attempts = 1
        no_refine_scores = []
        
        for i in range(num_runs):
            result = await self.backend.generate(GenerationRequest(
                prompt=prompt,
                capability=capability
            ))
            if result.success:
                no_refine_scores.append(result.quality_score or 5.0)
                print(f"  Run {i+1}: {result.quality_score}/10")
        
        # With refinement
        print()
        print("Phase 2: With refinement (max_attempts=3)")
        self.backend.max_attempts = 3
        with_refine_scores = []
        
        for i in range(num_runs):
            result = await self.backend.generate(GenerationRequest(
                prompt=prompt,
                capability=capability
            ))
            if result.success:
                with_refine_scores.append(result.quality_score or 5.0)
                print(f"  Run {i+1}: {result.quality_score}/10 (attempts: {result.attempts})")
        
        # Analysis
        no_refine_mean = statistics.mean(no_refine_scores) if no_refine_scores else 0
        with_refine_mean = statistics.mean(with_refine_scores) if with_refine_scores else 0
        improvement = with_refine_mean - no_refine_mean
        
        return {
            "prompt": prompt,
            "without_refinement": {
                "scores": no_refine_scores,
                "mean": no_refine_mean,
            },
            "with_refinement": {
                "scores": with_refine_scores,
                "mean": with_refine_mean,
            },
            "improvement": improvement,
            "refinement_effective": improvement > 0.5
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def measure_fidelity(
    prompt: str,
    comfyui_url: str = "http://127.0.0.1:8188",
    num_runs: int = 5
) -> FidelityReport:
    """Quick fidelity measurement."""
    benchmark = FidelityBenchmark(comfyui_url=comfyui_url)
    await benchmark.initialize()
    return await benchmark.run(prompt, num_runs=num_runs)


async def run_creative_unit_test(
    prompt: str,
    expected_min_score: float = 6.0,
    comfyui_url: str = "http://127.0.0.1:8188",
    num_runs: int = 3
) -> bool:
    """Run a creative unit test — does the system achieve minimum quality?
    
    Returns True if mean score >= expected_min_score
    """
    report = await measure_fidelity(prompt, comfyui_url, num_runs)
    passed = report.mean_score >= expected_min_score
    
    print(f"Creative Unit Test: {'PASSED' if passed else 'FAILED'}")
    print(f"  Expected: >= {expected_min_score}")
    print(f"  Actual: {report.mean_score:.2f}")
    
    return passed
