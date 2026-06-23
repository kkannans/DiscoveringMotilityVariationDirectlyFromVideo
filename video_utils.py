import imageio
import numpy as np
import subprocess
import os
import cv2
import shutil

try:
    import torch
except ImportError:
    torch = None

def _write_with_cv2(frames, video_path, fps=2):
    """Fallback: write video using OpenCV VideoWriter (no external ffmpeg needed)."""
    if not frames:
        print("No frames to write")
        return None

    out_dir = os.path.dirname(video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Prepare first frame to determine size
    first = frames[0]
    if first is None:
        print("First frame is None")
        return None
    if torch is not None and isinstance(first, torch.Tensor):
        first = first.detach().cpu().numpy()
    if first.ndim == 3 and first.shape[0] in (3, 4, 5):
        first = np.transpose(first, (1, 2, 0))
    if first.ndim == 3 and first.shape[2] > 3:
        first = first[:, :, :3]
    if first.dtype != np.uint8:
        if first.dtype in [np.float32, np.float64]:
            # Check if values are in [0, 1] range (normalized) or [0, 255] range
            if first.max() <= 1.0:
                # Scale from [0, 1] to [0, 255]
                first = (first * 255).clip(0, 255).astype(np.uint8)
            else:
                # Already in [0, 255] range, just clip and convert
                first = np.clip(first, 0, 255).astype(np.uint8)
        else:
            first = np.clip(first, 0, 255).astype(np.uint8)
    if first.ndim == 2:
        first_bgr = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    elif first.ndim == 3 and first.shape[2] == 3:
        # Assume RGB; convert to BGR
        first_bgr = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
    elif first.ndim == 3 and first.shape[2] == 1:
        first_bgr = cv2.cvtColor(first[..., 0], cv2.COLOR_GRAY2BGR)
    else:
        print(f"Unexpected first frame shape: {getattr(first, 'shape', None)}")
        return None

    height, width = first_bgr.shape[:2]
    # Choose codec set based on extension; prefer widely available ones
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ('.mp4', '.m4v', '.mov'):
        # Many OpenCV builds lack H.264; try mp4v. If it fails, fall back to AVI/MJPG.
        codec_candidates = [(cv2.VideoWriter_fourcc(*'mp4v'), video_path)]
    else:
        codec_candidates = []

    # Always include an AVI/MJPG fallback path
    avi_path = video_path if ext == '.avi' else os.path.splitext(video_path)[0] + '.avi'
    codec_candidates.extend([
        (cv2.VideoWriter_fourcc(*'XVID'), avi_path),
        (cv2.VideoWriter_fourcc(*'MJPG'), avi_path),
    ])

    writer = None
    opened_path = None
    for fourcc, path in codec_candidates:
        writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
        if writer.isOpened():
            opened_path = path
            break
        writer.release()
        writer = None
    if writer is None:
        print("Failed to open VideoWriter with common codecs (mp4v/XVID/MJPG).")
        return None

    try:
        for f in frames:
            if f is None:
                continue
            if torch is not None and isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            if f.ndim == 3 and f.shape[0] in (3, 4, 5):
                f = np.transpose(f, (1, 2, 0))
            if f.ndim == 3 and f.shape[2] > 3:
                f = f[:, :, :3]
            if f.dtype != np.uint8:
                if f.dtype in [np.float32, np.float64]:
                    # Check if values are in [0, 1] range (normalized) or [0, 255] range
                    if f.max() <= 1.0:
                        # Scale from [0, 1] to [0, 255]
                        f = (f * 255).clip(0, 255).astype(np.uint8)
                    else:
                        # Already in [0, 255] range, just clip and convert
                        f = np.clip(f, 0, 255).astype(np.uint8)
                else:
                    f = np.clip(f, 0, 255).astype(np.uint8)
            if f.ndim == 2:
                f_bgr = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.ndim == 3 and f.shape[2] == 3:
                f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            elif f.ndim == 3 and f.shape[2] == 1:
                f_bgr = cv2.cvtColor(f[..., 0], cv2.COLOR_GRAY2BGR)
            else:
                continue
            if (f_bgr.shape[1], f_bgr.shape[0]) != (width, height):
                f_bgr = cv2.resize(f_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(f_bgr)
    finally:
        writer.release()

    print("✅ Video created (OpenCV): {}".format(opened_path))
    return opened_path


class FFmpegStreamingVideoWriter:
    """
    Write video frames one at a time to avoid holding all frames in memory.
    Open ffmpeg process, call write_frame() for each frame, then close().
    """

    def __init__(self, video_path: str, fps: int = 2, width: int = 128, height: int = 128, input_bgr: bool = True):
        self.video_path = video_path
        self.fps = fps
        self.width = width
        self.height = height
        self.input_bgr = input_bgr
        self.proc = None
        self._open()

    def _open(self):
        out_dir = os.path.dirname(self.video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if shutil.which('ffmpeg') is None:
            raise RuntimeError("ffmpeg not found in PATH")
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'image2pipe', '-framerate', str(self.fps),
            '-vcodec', 'png', '-i', '-',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-movflags', '+faststart',
            self.video_path
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def write_frame(self, frame: np.ndarray):
        """Encode one frame (BGR uint8, any size) and pipe to ffmpeg."""
        if self.proc is None or self.proc.poll() is not None:
            return False
        if frame is None:
            return True
        f = frame
        if f.dtype != np.uint8:
            if f.dtype in [np.float32, np.float64]:
                f = (f * 255).clip(0, 255).astype(np.uint8) if f.max() <= 1.0 else np.clip(f, 0, 255).astype(np.uint8)
            else:
                f = np.clip(f, 0, 255).astype(np.uint8)
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.ndim == 3 and f.shape[2] == 3 and not self.input_bgr:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        elif f.ndim == 3 and f.shape[2] == 1:
            f = cv2.cvtColor(f[..., 0], cv2.COLOR_GRAY2BGR)
        if (f.shape[1], f.shape[0]) != (self.width, self.height):
            f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        ok, enc = cv2.imencode('.png', f)
        if not ok:
            return False
        try:
            self.proc.stdin.write(enc.tobytes())
        except (BrokenPipeError, ValueError, OSError):
            return False
        return True

    def close(self):
        if self.proc is None:
            return
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
        self.proc = None


def create_video_ffmpeg(frames, video_path: str, fps: int = 2, input_bgr: bool = False):
    """
    Create video using ffmpeg by piping PNG-encoded frames via stdin.
    Ensures even dimensions and a widely compatible pixel format.
    
    Args:
        frames: List or array of frames (numpy arrays)
        video_path: Output video path
        fps: Frames per second
        input_bgr: If True, frames are already BGR (e.g. from cv2.imread).
                   If False, frames are assumed RGB and converted to BGR for encoding.
    """
    if len(frames) == 0:
        print("No frames to write")
        return None

    # Ensure output dir exists
    out_dir = os.path.dirname(video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg not found in PATH. Please install ffmpeg to create videos.")
        return None

    # Build ffmpeg command with h264 (libx264) codec for maximum compatibility
    # - image2pipe reads a stream of images
    # - we encode each frame to PNG and write to stdin
    # - scale enforces even width/height to avoid green artifacts with yuv420p
    # Using libx264 instead of libx265 for better compatibility across players/systems
    cmd = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-f', 'image2pipe',
        '-framerate', str(fps),
        '-vcodec', 'png',
        '-i', '-',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-movflags', '+faststart',
        video_path
    ]

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for f in frames:
            # Check if ffmpeg process is still running
            if proc.poll() is not None:
                # Process has already exited, get the error
                try:
                    stderr_data = proc.stderr.read()
                    print("FFmpeg error: {}".format(stderr_data.decode('utf-8', errors='ignore')))
                except:
                    print("FFmpeg error (could not read stderr)")
                return None
                
            if f is None:
                continue

            # Convert PyTorch tensor to numpy if needed
            if torch is not None and isinstance(f, torch.Tensor):
                f = f.detach().cpu().numpy()
            if hasattr(f, 'numpy') and not isinstance(f, np.ndarray):
                f = np.asarray(f)
            # (C, H, W) -> (H, W, C); if C > 3 use first 3 channels as RGB
            if f.ndim == 3 and f.shape[0] in (3, 4, 5):
                f = np.transpose(f, (1, 2, 0))
            if f.ndim == 3 and f.shape[2] > 3:
                f = f[:, :, :3]
                
            # Ensure uint8 and 3-channel BGR for PNG encoder
            if f.dtype != np.uint8:
                if f.dtype in [np.float32, np.float64]:
                    # Check if values are in [0, 1] range (normalized) or [0, 255] range
                    if f.max() <= 1.0:
                        # Scale from [0, 1] to [0, 255]
                        f = (f * 255).clip(0, 255).astype(np.uint8)
                    else:
                        # Already in [0, 255] range, just clip and convert
                        f = np.clip(f, 0, 255).astype(np.uint8)
                else:
                    f = np.clip(f, 0, 255).astype(np.uint8)
            if f.ndim == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            elif f.ndim == 3 and f.shape[2] == 3:
                if not input_bgr:
                    # Caller passed RGB; convert to BGR for OpenCV encoding
                    f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            elif f.ndim == 3 and f.shape[2] == 1:
                # Single-channel image with channel dim; treat as grayscale
                f = cv2.cvtColor(f[..., 0], cv2.COLOR_GRAY2BGR)
            else:
                print(f"Skipping unexpected frame shape: {getattr(f, 'shape', None)}")
                continue

            ok, enc = cv2.imencode('.png', f)
            if not ok:
                print('Failed to encode frame to PNG; skipping frame')
                continue
                
            try:
                proc.stdin.write(enc.tobytes())
            except (BrokenPipeError, ValueError, OSError):
                # ffmpeg likely exited early
                print("FFmpeg process ended unexpectedly")
                return None
                
        # Close stdin to signal EOF to ffmpeg
        if proc.stdin:
            proc.stdin.close()
            
        # Wait for ffmpeg to finish
        return_code = proc.wait()
        
        if return_code != 0:
            # Get stderr if there was an error
            try:
                stderr_data = proc.stderr.read()
                print("FFmpeg error: {}".format(stderr_data.decode('utf-8', errors='ignore')))
            except:
                print("FFmpeg error (could not read stderr)")
            return None
            
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg to create videos.")
        return None
    except Exception as e:
        print(f"Error creating video: {e}")
        return None
    finally:
        # Ensure process is cleaned up
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("✅ Video created: {}".format(video_path))
    return video_path

if __name__ == "__main__":
    frames = [np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8) for _ in range(10)]
    fps = 2
    codec = 'libx264'  # More widely supported than libx265
    quality = 18
    output_path = "output_ffmpeg.mp4"
    create_video_ffmpeg(frames, output_path, fps=fps)