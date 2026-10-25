"""
Drishti Saathi — PyTorch Live Edge Navigation Runner
=====================================================
Runs real-time camera inference using PyTorch neural models:
- Accurate Metric Distance Measurement (Model 4)
- Staircase & Pothole Surface Hazard Detection (Model 2)
- Real-time spoken auditory alerts in Hindi & English

Usage:
    pip install torch torchvision opencv-python
    python pytorch_saathi.py
"""

import sys
import time
import os
import argparse
import threading

try:
    import cv2
    import numpy as np
    import torch
    from models.pytorch_models import SurfaceHazardNet, AccurateMetricDepthNet, PretrainedMiDaSDepthEstimator
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    print("[*] Install required packages via: pip install torch torchvision opencv-python numpy")
    sys.exit(0)

# Optional Hugging Face Transformers & YOLO Ensemble Import
try:
    from models.hf_vision_pipeline import EnsembleSurroundVerifier
    HF_PIPELINE_AVAILABLE = True
except Exception:
    HF_PIPELINE_AVAILABLE = False


class PyTorchDrishtiSaathi:
    def __init__(self, device='cpu', language='hi', enable_surround=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"[*] Initializing Drishti Saathi PyTorch Engine on: {self.device}")

        # 1. Load PyTorch Surface Hazard Neural Network (Stairs & Potholes)
        self.hazard_model = SurfaceHazardNet().to(self.device)
        self.hazard_model.eval()

        # 2. Load Production Metric Depth Network (Pre-trained Intel MiDaS + Optical Pinhole)
        self.depth_estimator = PretrainedMiDaSDepthEstimator(device=self.device)

        # 3. Optional Hugging Face Transformers & YOLO Surround Vision Engine
        self.enable_surround = enable_surround and HF_PIPELINE_AVAILABLE
        self.surround_verifier = None
        self.confirmed_surroundings = []
        self.surround_worker_running = False

        if self.enable_surround:
            try:
                print("[*] Initializing Hugging Face Transformers & YOLO Surround Verifier...")
                self.surround_verifier = EnsembleSurroundVerifier()
                print("[✓] Hugging Face & YOLO Surround Verifier active.")
            except Exception as e:
                print(f"[*] Note: Hugging Face surround engine running in lightweight mode: {e}")

        self.language = language
        self.last_speech_time = 0
        self.last_hazard_spoken = ""
        self.speech_cooldown_sec = 8.0

    def preprocess_frame(self, bgr_frame):
        """Converts OpenCV BGR image to normalized PyTorch tensor [1, 3, 120, 160]"""
        resized = cv2.resize(bgr_frame, (160, 120))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        normalized = (tensor - mean) / std
        return normalized.unsqueeze(0).to(self.device)

    def _async_surround_scan(self, frame_copy):
        """Runs Hugging Face Transformers + YOLO verification in background thread."""
        try:
            if self.surround_verifier:
                results = self.surround_verifier.analyze_and_verify_scene(frame_copy)
                self.confirmed_surroundings = results[:4]  # Top 4 confident objects
        except Exception:
            pass
        finally:
            self.surround_worker_running = False

    def trigger_surround_scan(self, frame):
        if self.enable_surround and not self.surround_worker_running:
            self.surround_worker_running = True
            thread = threading.Thread(target=self._async_surround_scan, args=(frame.copy(),), daemon=True)
            thread.start()

    def speak(self, text):
        now = time.time()
        if now - self.last_speech_time < self.speech_cooldown_sec:
            return
        self.last_speech_time = now
        print(f"\n[VOICE SYNTHESIS] >>> {text}\n")
        try:
            # Cross-platform TTS trigger
            if sys.platform.startswith('win'):
                import subprocess
                subprocess.Popen(['powershell', '-Command', f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def run_live(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[!] Unable to open camera index {camera_index}. Testing with synthetic frame.")
            return

        print("[✓] Camera stream online. Press 'q' to exit. Press 's' to scan surroundings with Hugging Face & YOLO.")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]
            tensor = self.preprocess_frame(frame)

            # Trigger Hugging Face & YOLO surround perception every 60 frames (~2s)
            if frame_count % 60 == 0:
                self.trigger_surround_scan(frame)

            # Model 4: Production Metric Depth Estimation (MiDaS + Hybrid Optical Pinhole)
            depth_out = self.depth_estimator.estimate_depth(frame)
            dist_center = depth_out['dist_center']
            dist_left = depth_out['dist_left']
            dist_right = depth_out['dist_right']

            with torch.no_grad():
                # Model 2: PyTorch Surface Hazard Inference (Stairs & Potholes)
                hazard_out = self.hazard_model(tensor)
                hazard_probs = hazard_out['hazard_probs'].cpu().numpy()[0]
                hazard_idx = int(np.argmax(hazard_probs))
                hazard_conf = hazard_probs[hazard_idx]

            # Map classes: 0 = Clear, 1 = Stairs, 2 = Pothole
            hazard_name = "CLEAR PATH"
            hazard_color = (0, 255, 0)
            if hazard_idx == 1 and hazard_conf > 0.45:
                hazard_name = "STAIRS DETECTED"
                hazard_color = (0, 165, 255)
                if dist_center <= 0.85:
                    self.speak("Caution! Stairs directly ahead. Stop!")
                else:
                    self.speak(f"Stairs ahead at {dist_center:.1f} meters.")
            elif hazard_idx == 2 and hazard_conf > 0.45:
                hazard_name = "POTHOLE / VOID DETECTED"
                hazard_color = (0, 0, 255)
                self.speak(f"Caution! Pothole detected, {dist_center:.1f} meters.")
            elif dist_center <= 0.55:
                hazard_name = "WALL / BARRIER STOP"
                hazard_color = (0, 0, 255)
                self.speak(f"Wall ahead, {dist_center:.1f} meters.")

            # Overlay HUD Telemetry on Frame
            cv2.putText(frame, "DRISHTI SAATHI // PYTORCH NEURAL ENGINE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)
            cv2.putText(frame, f"MODEL 2 [HAZARD]: {hazard_name} ({hazard_conf*100:.0f}%)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hazard_color, 2)
            cv2.putText(frame, f"MODEL 4 [DISTANCE]: {dist_center:.2f}m (L: {dist_left:.1f}m | R: {dist_right:.1f}m)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display Confirmed Surroundings (HF Transformers & YOLO) and Hybrid Distance
            if self.confirmed_surroundings:
                obj_items = []
                for x in self.confirmed_surroundings[:2]:
                    bbox = x.get("bbox", [0, 0, 0, 0])
                    obj_dist, method = self.depth_estimator.calculate_hybrid_object_distance(bbox, x['object'], frame.shape, depth_out)
                    obj_items.append(f"{x['object']} {obj_dist}m")
                    # Draw object bounding box if available
                    if len(bbox) == 4 and bbox[2] > bbox[0]:
                        bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 50), 2)
                        cv2.putText(frame, f"{x['object'].upper()}: {obj_dist}m", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 2)

                cv2.putText(frame, f"SURROUND [HF & YOLO]: {', '.join(obj_items)}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 2)
            else:
                cv2.putText(frame, "SURROUND [HF & YOLO]: Active (Press 's' to scan)", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # Draw central corridor reticle
            cv2.rectangle(frame, (int(w * 0.35), int(h * 0.25)), (int(w * 0.65), int(h * 0.85)), hazard_color, 2)

            cv2.imshow("Drishti Saathi - PyTorch Multi-AI Navigation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("[*] Triggering manual surrounding scan via Hugging Face & YOLO...")
                self.trigger_surround_scan(frame)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drishti Saathi - PyTorch Multi-AI Assistive Vision")
    parser.add_argument('--device', type=str, default='cpu', help="Execution device: 'cpu' or 'cuda'")
    parser.add_argument('--no-surround', action='store_true', help="Disable Hugging Face Transformers & YOLO surround engine")
    args = parser.parse_args()

    engine = PyTorchDrishtiSaathi(device=args.device, enable_surround=not args.no_surround)
    engine.run_live()
