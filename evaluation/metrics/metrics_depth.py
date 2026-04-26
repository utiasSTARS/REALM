import torch
import numpy as np

def abs_rel_diff(y_input, y_target, eps=1e-6):
    abs_diff = np.abs(y_target - y_input)
    valid = ~np.isnan(abs_diff) & ~np.isnan(y_target) & (y_target > eps)
    if not np.any(valid): return 0.0
    return (abs_diff[valid] / (y_target[valid])).mean()

def squ_rel_diff(y_input, y_target, eps=1e-6):
    abs_diff = np.abs(y_target - y_input)
    valid = ~np.isnan(abs_diff) & ~np.isnan(y_target) & (y_target > eps)
    if not np.any(valid): return 0.0
    return (abs_diff[valid]**2 / (y_target[valid])).mean()

def rms_linear(y_input, y_target):
    abs_diff = np.abs(y_target - y_input)
    valid = ~np.isnan(abs_diff) & ~np.isnan(y_target)
    if not np.any(valid): return 0.0
    return np.sqrt((abs_diff[valid]**2).mean())

def rms_log(y_input, y_target, eps=1e-6):
    # The paper uses RMSE log, which is different from scale_invariant_error
    valid = ~np.isnan(y_input) & ~np.isnan(y_target) & (y_target > eps) & (y_input > eps)
    if not np.any(valid): return 0.0
    log_diff = np.log(y_target[valid]) - np.log(y_input[valid])
    return np.sqrt((log_diff**2).mean())

def threshold_accuracy(y_input, y_target, threshold=1.25, eps=1e-6):
    valid = ~np.isnan(y_input) & ~np.isnan(y_target) & (y_target > eps) & (y_input > eps)
    if not np.any(valid): return 0.0
    
    thresh = np.maximum((y_target[valid] / y_input[valid]), (y_input[valid] / y_target[valid]))
    return (thresh <= threshold).mean()

def mean_error_at_range(y_input, y_target, max_range):
    """Calculates Mean Absolute Error (MAE) for pixels where ground truth <= max_range."""
    abs_diff = np.abs(y_target - y_input)
    # Mask: Valid pixels AND within the specified range (e.g., 0 < depth <= 10m)
    valid = ~np.isnan(abs_diff) & ~np.isnan(y_target) & (y_target > 0) & (y_target <= max_range)
    
    if not np.any(valid): return 0.0
    return abs_diff[valid].mean()

def silog_error(y_input, y_target, eps=1e-6):
    valid = ~np.isnan(y_input) & ~np.isnan(y_target) & (y_target > eps) & (y_input > eps)
    if not np.any(valid): return 0.0
    
    log_diff = np.log(y_target[valid]) - np.log(y_input[valid])
    # SILog formula: mean(d^2) - (mean(d))^2
    return (log_diff**2).mean() - (log_diff.mean())**2

# Update your threshold function to use <=
def threshold_accuracy(y_input, y_target, threshold=1.25, eps=1e-6):
    valid = ~np.isnan(y_input) & ~np.isnan(y_target) & (y_target > eps) & (y_input > eps)
    if not np.any(valid): return 0.0
    
    thresh = np.maximum((y_target[valid] / y_input[valid]), (y_input[valid] / y_target[valid]))
    return (thresh <= threshold).mean()

class MetricsDepth:
    def __init__(self, min_depth=2.0, max_depth=80.0, ranges=[10.0, 20.0, 30.0]):
        self.min_depth = min_depth  
        self.max_depth = max_depth  
        self.ranges = ranges
        self.metrics_accum = {}
        self.count = 0
        self.reset()

    def reset(self):
        self.metrics_accum = {}
        self.count = 0

    def _compute_metrics_for_mask(self, y_hat, y, mask):
        """Helper to compute all metrics for a given valid mask."""
        if not np.any(mask):
            # Return zeros if no valid pixels exist in this mask
            return {k: 0.0 for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3']}
        
        y_hat_valid = y_hat[mask]
        y_valid = y[mask]

        return {
            'abs_rel': abs_rel_diff(y_hat_valid, y_valid),
            'sq_rel': squ_rel_diff(y_hat_valid, y_valid),
            'rmse': rms_linear(y_hat_valid, y_valid),
            'rmse_log': rms_log(y_hat_valid, y_valid),
            'silog': silog_error(y_hat_valid, y_valid),
            'a1': threshold_accuracy(y_hat_valid, y_valid, 1.25),
            'a2': threshold_accuracy(y_hat_valid, y_valid, 1.25 ** 2),
            'a3': threshold_accuracy(y_hat_valid, y_valid, 1.25 ** 3)
        }

    def _add_to_accum(self, prefix, metrics_dict, batch_size):
        """Adds computed metrics to the running total with an optional prefix."""
        for k, v in metrics_dict.items():
            full_key = f"{prefix}{k}" if prefix else k
            if full_key not in self.metrics_accum:
                self.metrics_accum[full_key] = 0.0
            self.metrics_accum[full_key] += v * batch_size

    def update_batch(self, y_hat, y, event_mask=None):
        with torch.no_grad():
            if torch.is_tensor(y_hat): y_hat = y_hat.detach().cpu().numpy()
            if torch.is_tensor(y): y = y.detach().cpu().numpy()
            if torch.is_tensor(event_mask): event_mask = event_mask.detach().cpu().numpy()
            
            if y_hat.ndim == 4 and y_hat.shape[1] == 1: y_hat = y_hat.squeeze(1)
            if y.ndim == 4 and y.shape[1] == 1: y = y.squeeze(1)

            batch_size = y_hat.shape[0]

            # 1. Clip predictions strictly between your min and max
            y_hat = np.clip(y_hat, self.min_depth, self.max_depth) 

            # 2. Mask out any ground truth that is less than 2m or greater than 80m
            base_mask = ~np.isnan(y_hat) & ~np.isnan(y) & (y >= self.min_depth) & (y <= self.max_depth)
            
            if event_mask is not None:
                if event_mask.ndim == 4 and event_mask.shape[1] == 1: event_mask = event_mask.squeeze(1)
                base_mask = base_mask & (event_mask > 0)

            global_metrics = self._compute_metrics_for_mask(y_hat, y, base_mask)
            self._add_to_accum("", global_metrics, batch_size)

            for dist in self.ranges:
                # Target must be <= dist AND still respect the base_mask (which enforces >= 2m)
                range_mask = base_mask & (y <= dist) 
                
                if np.any(range_mask):
                    mae = np.abs(y[range_mask] - y_hat[range_mask]).mean()
                else:
                    mae = 0.0
                
                err_key = f"err_{int(dist)}m"
                if err_key not in self.metrics_accum:
                    self.metrics_accum[err_key] = 0.0
                self.metrics_accum[err_key] += mae * batch_size

            self.count += batch_size

    def get_metrics_summary(self):
        if self.count == 0:
            return {k: 0.0 for k in self.metrics_accum.keys()}
        return {k: v / self.count for k, v in self.metrics_accum.items()}