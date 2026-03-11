"""
ExeVision AI - Main Pipeline Orchestrator
==========================================
Processes exercise videos through the complete analysis pipeline.

Usage:
    python main.py <video_path>
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from config import PipelineConfig, PipelineResult
    from pose_extractor import PoseExtractor
    from view_classifier import ViewClassifier
    from temporal_segmenter import TemporalSegmenter
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure all modules (config.py, pose_extractor.py, view_classifier.py, temporal_segmenter.py) exist.")
    sys.exit(1)

# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

class ExeVisionPipeline:
    """Main pipeline that orchestrates all analysis stages"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.extractor = PoseExtractor(self.config)
        self.view_classifier = ViewClassifier(self.config)
        self.segmenter = TemporalSegmenter(self.config)
    
    def process(self, video_path: str, save_output: bool = True) -> PipelineResult:
        """Process a video through the complete pipeline."""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().isoformat()
        
        print(f"\n{'='*60}")
        print(f"  ExeVision AI Pipeline")
        print(f"  Video: {video_id}")
        print(f"{'='*60}\n")
        
        result = PipelineResult(
            video_path=video_path,
            video_id=video_id,
            timestamp=timestamp,
            extraction=None,
            view=None,
            segmentation=None
        )
        
        # Validate video exists
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return result
        
        # Stage 1: Pose Extraction
        print("📍 Stage 1: Pose Extraction...")
        extraction = self.extractor.extract(video_path)
        result.extraction = extraction
        
        if not extraction.success:
            print(f"   ❌ Failed: {extraction.error}")
            return result
        
        print(f"   ✓ Extracted {extraction.frame_count} frames @ {extraction.fps:.1f} fps")
        
        # Stage 2: View Classification
        print("\n📍 Stage 2: View Classification...")
        view = self.view_classifier.classify(extraction)
        result.view = view
        
        if not view.success:
            print(f"   ❌ Failed: {view.error}")
        else:
            status = "✓" if view.is_valid_view else "⚠"
            print(f"   {status} Detected: {view.predicted_view.upper()} ({view.confidence*100:.1f}%)")
            
            if not view.is_valid_view:
                print(f"   ⚠ Warning: View '{view.predicted_view}' may not be optimal for analysis")
                print(f"     Recommended: {', '.join(self.config.valid_views)}")
        
        # Stage 3: Temporal Segmentation
        print("\n📍 Stage 3: Temporal Segmentation...")
        segmentation = self.segmenter.segment(extraction)
        result.segmentation = segmentation
        
        if not segmentation.success:
            print(f"   ❌ Failed: {segmentation.error}")
        else:
            print(f"   ✓ Detected {segmentation.total_reps} repetition(s)")
            for rep in segmentation.repetitions:
                print(f"     Rep {rep['rep_id']}: frames {rep['start_frame']}-{rep['end_frame']} "
                      f"(depth: {rep['squat_depth']:.3f})")
        
        # Future stages
        print("\n📍 Stage 4-7: [Pending Implementation]")
        
        # Save output
        if save_output:
            os.makedirs(self.config.output_root, exist_ok=True)
            output_path = os.path.join(self.config.output_root, f"{video_id}_analysis.json")
            
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        return result

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    
    video_path = os.path.abspath(sys.argv[1])
    pipeline = ExeVisionPipeline()
    result = pipeline.process(video_path)
    
    sys.exit(0 if (result.extraction and result.extraction.success) else 1)

if __name__ == "__main__":
    main()
