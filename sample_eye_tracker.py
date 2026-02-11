#!/usr/bin/env python3
"""
Sample script demonstrating eye tracking with OpenIris Python.

Opens an MP4 file (or webcam) and overlays eye keypoints in real-time.

Usage:
    python sample_eye_tracker.py [video_path]
    
    If video_path is not provided, will try to use the default webcam.
    Press 'q' to quit, 's' to save current frame, 'p' to pause.

Example:
    python sample_eye_tracker.py eye_video.mp4
"""

import sys
import cv2
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from python_openiris import EyeTracker, EyeTrackerSettings
from python_openiris.tracker import draw_eye_overlay


def main():
    parser = argparse.ArgumentParser(
        description="Eye tracking demo with OpenIris Python"
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to video file. If not provided, uses webcam."
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=50,
        help="Threshold for pupil detection (0-255, default: 50)"
    )
    parser.add_argument(
        "--bright-threshold",
        type=int,
        default=200,
        help="Threshold for corneal reflection detection (0-255, default: 200)"
    )
    parser.add_argument(
        "--iris-radius",
        type=float,
        default=80.0,
        help="Expected iris radius in pixels (default: 80)"
    )
    parser.add_argument(
        "--track-eyelids",
        action="store_true",
        help="Enable eyelid tracking"
    )
    parser.add_argument(
        "--no-cr",
        action="store_true",
        help="Disable corneal reflection tracking"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Create settings
    settings = EyeTrackerSettings(
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        iris_radius=args.iris_radius,
        track_eyelids=args.track_eyelids,
        track_corneal_reflections=not args.no_cr,
        eyelid_tracking_method="fixed" if args.track_eyelids else "none"
    )
    
    # Create tracker
    tracker = EyeTracker(settings)
    
    # Open video source
    if args.video:
        print(f"Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print("Opening webcam (camera 0)...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Writing output to: {args.output}")
    
    # Processing loop
    paused = False
    frame_count = 0
    
    print("\nControls:")
    print("  q - Quit")
    print("  p - Pause/Resume")
    print("  s - Save current frame")
    print("  +/- - Adjust dark threshold")
    print("  [/] - Adjust bright threshold")
    print("")
    
    try:
        for result, frame in tracker.track_video(cap):
            frame_count += 1
            
            # Draw overlay
            overlay_frame = draw_eye_overlay(
                frame.copy(),
                result,
                draw_eyelids=args.track_eyelids
            )
            
            # Add info text
            info_text = f"Frame: {result.frame_number}"
            if result.tracking_success:
                info_text += f" | Pupil: ({result.pupil.center[0]:.1f}, {result.pupil.center[1]:.1f})"
                info_text += f" | CRs: {len(result.corneal_reflections)}"
            else:
                info_text += " | Tracking: FAILED"
            
            cv2.putText(
                overlay_frame, info_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Show threshold values
            threshold_text = f"Dark: {settings.dark_threshold} | Bright: {settings.bright_threshold}"
            cv2.putText(
                overlay_frame, threshold_text,
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            # Write to output if enabled
            if writer:
                writer.write(overlay_frame)
            
            # Display
            cv2.imshow("OpenIris Eye Tracker", overlay_frame)
            
            # Handle key input (1ms delay for responsive playback)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):
                filename = f"frame_{frame_count:06d}.png"
                cv2.imwrite(filename, overlay_frame)
                print(f"Saved: {filename}")
            elif key == ord('+') or key == ord('='):
                settings.dark_threshold = min(255, settings.dark_threshold + 5)
                print(f"Dark threshold: {settings.dark_threshold}")
            elif key == ord('-'):
                settings.dark_threshold = max(0, settings.dark_threshold - 5)
                print(f"Dark threshold: {settings.dark_threshold}")
            elif key == ord(']'):
                settings.bright_threshold = min(255, settings.bright_threshold + 5)
                print(f"Bright threshold: {settings.bright_threshold}")
            elif key == ord('['):
                settings.bright_threshold = max(0, settings.bright_threshold - 5)
                print(f"Bright threshold: {settings.bright_threshold}")
            
            # Handle pause state
            while paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    paused = False
                    print("Resumed")
                elif key == ord('q'):
                    print("Quitting...")
                    paused = False
                    cap.release()
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
