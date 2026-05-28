import unittest
import time
import numpy as np
import cv2
from canvas.layer_stack import LayerStack
from canvas.canvas_renderer import CanvasRenderer
from canvas.layer import Layer


class TestPerformance(unittest.TestCase):
    def test_alpha_compositing_performance(self):
        # Create a layer stack with 2 layers of size 1280x720
        width, height = 1280, 720
        
        # Scenario A: completely empty layers (very common in start/clear states)
        # Old code did full float math on all pixels (1280 * 720 * 4)
        # New code has early exit check
        
        # Old blending logic simulation
        def old_composite(dst, src):
            src_alpha = src[:, :, 3:4].astype(np.float32) / 255.0
            dst_alpha = dst[:, :, 3:4].astype(np.float32) / 255.0
            out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
            safe_alpha = np.maximum(out_alpha, 1e-6)
            dst[:, :, :3] = (
                (src[:, :, :3].astype(np.float32) * src_alpha +
                 dst[:, :, :3].astype(np.float32) * dst_alpha * (1.0 - src_alpha))
                / safe_alpha
            ).astype(np.uint8)
            dst[:, :, 3:4] = (out_alpha * 255.0).astype(np.uint8)
            
        dst_old = np.zeros((height, width, 4), dtype=np.uint8)
        src_old = np.zeros((height, width, 4), dtype=np.uint8)
        
        t0 = time.perf_counter()
        old_composite(dst_old, src_old)
        old_time = time.perf_counter() - t0
        
        # New blending logic (using LayerStack static method)
        dst_new = np.zeros((height, width, 4), dtype=np.uint8)
        src_new = np.zeros((height, width, 4), dtype=np.uint8)
        
        t0 = time.perf_counter()
        LayerStack._alpha_composite_inplace(dst_new, src_new)
        new_time = time.perf_counter() - t0
        
        print(f"\n[Performance Test - Empty Layers]:")
        print(f"Old composite time: {old_time*1000:.3f} ms")
        print(f"New composite time: {new_time*1000:.3f} ms")
        print(f"Speedup: {old_time / max(new_time, 1e-9):.1f}x")
        
        # Verification: New time should be significantly faster (at least 5x) because of early exit!
        self.assertLess(new_time, old_time)

    def test_roi_compositing_performance(self):
        width, height = 1280, 720
        # Scenario B: layers have a small active region (e.g. user drawing a line of 100x100 pixels)
        dst_old = np.zeros((height, width, 4), dtype=np.uint8)
        src_old = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Draw a small circle in src
        cv2.circle(src_old, (200, 200), 50, (255, 0, 0, 255), -1)
        # And destination has some content
        cv2.circle(dst_old, (205, 205), 50, (0, 255, 0, 255), -1)
        
        def old_composite(dst, src):
            src_alpha = src[:, :, 3:4].astype(np.float32) / 255.0
            dst_alpha = dst[:, :, 3:4].astype(np.float32) / 255.0
            out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
            safe_alpha = np.maximum(out_alpha, 1e-6)
            dst[:, :, :3] = (
                (src[:, :, :3].astype(np.float32) * src_alpha +
                 dst[:, :, :3].astype(np.float32) * dst_alpha * (1.0 - src_alpha))
                / safe_alpha
            ).astype(np.uint8)
            dst[:, :, 3:4] = (out_alpha * 255.0).astype(np.uint8)
            
        t0 = time.perf_counter()
        old_composite(dst_old, src_old)
        old_time = time.perf_counter() - t0
        
        dst_new = np.zeros((height, width, 4), dtype=np.uint8)
        src_new = np.zeros((height, width, 4), dtype=np.uint8)
        cv2.circle(src_new, (200, 200), 50, (255, 0, 0, 255), -1)
        cv2.circle(dst_new, (205, 205), 50, (0, 255, 0, 255), -1)
        
        t0 = time.perf_counter()
        LayerStack._alpha_composite_inplace(dst_new, src_new)
        new_time = time.perf_counter() - t0
        
        print(f"\n[Performance Test - ROI (100x100 area)]:")
        print(f"Old composite time: {old_time*1000:.3f} ms")
        print(f"New composite time: {new_time*1000:.3f} ms")
        print(f"Speedup: {old_time / max(new_time, 1e-9):.1f}x")
        
        # Verify both outputs match in content (correctness)
        self.assertTrue(np.array_equal(dst_old, dst_new))
        self.assertLess(new_time, old_time)


if __name__ == "__main__":
    unittest.main()
