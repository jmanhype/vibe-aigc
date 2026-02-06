#!/usr/bin/env python3
"""Test the VLM feedback refinement system."""

from vibe_aigc.vlm_feedback import VLMFeedback, FeedbackResult, MediaType


def test_refinement():
    """Test the weakness-to-refinement mapping."""
    vlm = VLMFeedback()
    
    # Test 1: Basic weak prompt with common issues
    print("=" * 60)
    print("TEST 1: Basic weak prompt")
    print("=" * 60)
    
    weak_prompt = "a cat sitting"
    feedback = FeedbackResult(
        quality_score=4.5,
        media_type=MediaType.IMAGE,
        description="Generic cat image with flat lighting",
        strengths=["Subject is recognizable"],
        weaknesses=[
            "Lighting is flat and boring",
            "Composition is centered and static",
            "Colors are muddy and dull",
            "Lacks fine detail"
        ],
        prompt_improvements=["add dramatic shadows", "use golden hour lighting"],
        should_retry=True
    )
    
    refined = vlm.refine_prompt(weak_prompt, feedback)
    summary = vlm.get_refinement_summary(weak_prompt, feedback)
    
    print(f"Original: {weak_prompt}")
    print(f"Refined:  {refined}")
    print(f"\nWeaknesses parsed: {summary['weaknesses_parsed']}")
    print(f"Refinements applied:")
    for r in summary['refinements_applied']:
        print(f"  - '{r['weakness']}'")
        for ref in r['refinements']:
            print(f"      -> {ref['addition']}")
    
    # Test 2: Already detailed prompt (should add less)
    print("\n" + "=" * 60)
    print("TEST 2: Already detailed prompt")
    print("=" * 60)
    
    detailed_prompt = "a majestic lion in dramatic lighting, detailed fur, 8k quality"
    feedback2 = FeedbackResult(
        quality_score=6.0,
        media_type=MediaType.IMAGE,
        description="Good lion but composition issues",
        strengths=["Good lighting", "Nice detail"],
        weaknesses=[
            "Composition is too centered",
            "Depth of field is flat"
        ],
        prompt_improvements=["off-center framing"],
        should_retry=True
    )
    
    refined2 = vlm.refine_prompt(detailed_prompt, feedback2)
    print(f"Original: {detailed_prompt}")
    print(f"Refined:  {refined2}")
    
    # Test 3: Video-specific issues
    print("\n" + "=" * 60)
    print("TEST 3: Video-specific issues")
    print("=" * 60)
    
    video_prompt = "a person walking"
    feedback3 = FeedbackResult(
        quality_score=5.0,
        media_type=MediaType.VIDEO,
        description="Walking animation with issues",
        strengths=["Motion is present"],
        weaknesses=[
            "Motion is jerky and unnatural",
            "Flickering between frames",
            "Subject morphs slightly"
        ],
        prompt_improvements=["smoother transitions"],
        should_retry=True
    )
    
    refined3 = vlm.refine_prompt(video_prompt, feedback3)
    print(f"Original: {video_prompt}")
    print(f"Refined:  {refined3}")
    
    # Test 4: Anatomy issues (common in AI art)
    print("\n" + "=" * 60)
    print("TEST 4: Anatomy issues")
    print("=" * 60)
    
    person_prompt = "a woman holding flowers"
    feedback4 = FeedbackResult(
        quality_score=5.5,
        media_type=MediaType.IMAGE,
        description="Portrait with hand issues",
        strengths=["Good face", "Nice colors"],
        weaknesses=[
            "Hand anatomy is distorted",
            "Fingers look unnatural",
            "Proportions seem off"
        ],
        prompt_improvements=["fix hands"],
        should_retry=True
    )
    
    refined4 = vlm.refine_prompt(person_prompt, feedback4)
    print(f"Original: {person_prompt}")
    print(f"Refined:  {refined4}")
    
    # Test 5: No refinements needed (high score)
    print("\n" + "=" * 60)
    print("TEST 5: High quality - minimal changes")
    print("=" * 60)
    
    good_prompt = "a stunning landscape, golden hour, dramatic clouds"
    feedback5 = FeedbackResult(
        quality_score=8.5,
        media_type=MediaType.IMAGE,
        description="Beautiful landscape",
        strengths=["Excellent composition", "Great lighting", "Rich colors"],
        weaknesses=[],  # No weaknesses!
        prompt_improvements=[],
        should_retry=False
    )
    
    refined5 = vlm.refine_prompt(good_prompt, feedback5)
    print(f"Original: {good_prompt}")
    print(f"Refined:  {refined5}")
    print(f"(Should be unchanged)")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_refinement()
