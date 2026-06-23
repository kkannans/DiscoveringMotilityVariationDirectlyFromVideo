from pathlib import Path
import csv
import datetime
import torch
from collections import defaultdict
from typing import Any, Dict, List, Optional
try:
    import psutil
except ImportError:
    psutil = None
import numpy as np


class MetricsLogger:
    """Local metrics logging system for experiment tracking."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate CSV files for different metrics
        self.train_log = self.log_dir / "train_metrics.csv"
        self.val_log = self.log_dir / "val_metrics.csv"
        self.system_log = self.log_dir / "system_metrics.csv"
        self.profiling_log = self.log_dir / "profiling_summary.csv"
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        # Track metrics in memory for plotting
        self.metrics_history: Dict[str, List[Any]] = defaultdict(list)
    
    def _init_csv_files(self):
        """Initialize CSV files with headers. If a file already exists (e.g. after resume),
        do not overwrite it so that previous run history is preserved."""
        # Training metrics
        if not self.train_log.exists():
            with open(self.train_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'batch', 'loss', 'learning_rate',
                    'epoch_duration_sec', 'batch_duration_sec', 'timestamp'
                ])

        # Validation metrics
        if not self.val_log.exists():
            with open(self.val_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'loss', 'baseline_copy', 'baseline_black',
                    'baseline_linear_interp',
                    'is_best_model', 'timestamp'
                ])

        # System metrics (only create if missing, so resume keeps history)
        if not self.system_log.exists():
            with open(self.system_log, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'timestamp', 'epoch', 'batch', 'cpu_percent',
                    'cpu_memory_used_gb', 'cpu_memory_total_gb'
                ]
                if torch.cuda.is_available():
                    headers.extend([
                        'gpu_memory_allocated_gb', 'gpu_memory_reserved_gb',
                        'gpu_memory_peak_allocated_gb', 'gpu_memory_total_gb'
                    ])
                writer.writerow(headers)
    
    def log_training_batch(self, epoch: int, batch: int, loss: float, 
                          learning_rate: float, batch_duration: float):
        """Log training batch metrics."""
        with open(self.train_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch, loss, learning_rate, 
                '', batch_duration, datetime.datetime.now().isoformat()
            ])
        
        # Store in memory for plotting
        self.metrics_history['train_loss'].append((epoch, batch, loss))
        self.metrics_history['learning_rate'].append((epoch, batch, learning_rate))
    
    def log_training_epoch(self, epoch: int, avg_loss: float, epoch_duration: float, training_baseline_losses: Optional[Dict[str, Any]] = None):
        """Log training epoch summary."""
        # Update the last entry with epoch duration
        # Read, modify last line, write back (simple approach)
        lines = []
        with open(self.train_log, 'r') as f:
            lines = f.readlines()
        
        if lines:
            # Update the last line with epoch duration
            last_line = lines[-1].strip().split(',')
            last_line[4] = str(epoch_duration)  # epoch_duration_sec column
            lines[-1] = ','.join(last_line) + '\n'
            
            with open(self.train_log, 'w') as f:
                f.writelines(lines)

        # Store in memory  
        self.metrics_history['train_loss'].append((epoch, avg_loss))    
        if training_baseline_losses:
            self.metrics_history['train_baseline_copy'].append(
                (epoch, training_baseline_losses.get('copy', 0.0))
            )
            self.metrics_history['train_baseline_black'].append(
                (epoch, training_baseline_losses.get('black', 0.0))
            )
            self.metrics_history['train_baseline_linear_interp'].append(
                (epoch, training_baseline_losses.get('linear_interpolation', 0.0))
            )
    
    def log_validation_epoch(self, epoch: int, loss: float, is_best: bool = False, validation_baseline_losses: Optional[Dict[str, Any]] = None):
        """Log validation epoch metrics."""
        with open(self.val_log, 'a', newline='') as f:
            writer = csv.writer(f)
            if validation_baseline_losses:
                linear_interp = validation_baseline_losses.get('linear_interpolation', '')
                writer.writerow([
                    epoch,
                    loss,
                    validation_baseline_losses.get('copy', ''),
                    validation_baseline_losses.get('black', ''),
                    linear_interp,
                    is_best,
                    datetime.datetime.now().isoformat(),
                ])
            else:
                writer.writerow([
                    epoch,
                    loss,
                    '',
                    '',
                    '',
                    is_best,
                    datetime.datetime.now().isoformat(),
                ])
        
        # Store in memory
        self.metrics_history['val_loss'].append((epoch, loss))
        if validation_baseline_losses:
            self.metrics_history['val_baseline_copy'].append(
                (epoch, validation_baseline_losses.get('copy', 0.0))
            )
            self.metrics_history['val_baseline_black'].append(
                (epoch, validation_baseline_losses.get('black', 0.0))
            )
            self.metrics_history['val_baseline_linear_interp'].append(
                (epoch, validation_baseline_losses.get('linear_interpolation', 0.0))
            )
    
    def log_system_metrics(self, epoch: int, batch: int):
        """Log current system metrics. All memory values are in GB."""
        if psutil is None:
            return
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            cpu_memory_used_gb = memory.used / (1024**3)
            cpu_memory_total_gb = memory.total / (1024**3)
            
            # Prepare row data
            row = [
                datetime.datetime.now().isoformat(), epoch, batch,
                cpu_percent, cpu_memory_used_gb, cpu_memory_total_gb
            ]
            
            # GPU memory metrics (all in GB)
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_memory_allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_memory_reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                gpu_memory_peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                
                row.extend([
                    gpu_memory_allocated_gb,
                    gpu_memory_reserved_gb,
                    gpu_memory_peak_allocated_gb,
                    gpu_memory_total_gb
                ])
            
            with open(self.system_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"Warning: Could not log system metrics: {e}", flush=True)
    
    def create_summary_plots(self):
        """Create comprehensive summary plots of training progress."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.switch_backend('Agg')
            
            # Create a comprehensive dashboard with multiple plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Training and Validation Losses
            if self.metrics_history['train_loss'] and self.metrics_history['val_loss']:
                # Extract epoch-level data (remove batch-level data)
                train_epochs = []
                train_losses = []
                val_epochs = []
                val_losses = []
                
                # Group by epoch for training losses
                epoch_train_losses = {}
                for epoch, batch, loss in self.metrics_history['train_loss']:
                    if epoch not in epoch_train_losses:
                        epoch_train_losses[epoch] = []
                    epoch_train_losses[epoch].append(loss)
                
                # Calculate average loss per epoch
                for epoch in sorted(epoch_train_losses.keys()):
                    train_epochs.append(epoch)
                    train_losses.append(np.mean(epoch_train_losses[epoch]))
                
                # Validation losses (already epoch-level)
                for epoch, loss in self.metrics_history['val_loss']:
                    val_epochs.append(epoch)
                    val_losses.append(loss)
                
                ax1.plot(train_epochs, train_losses, 'b-', alpha=0.8, label='Train Loss', linewidth=2)
                ax1.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Progress')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_yscale('log')
            
            # 2. Learning Rate Schedule
            if self.metrics_history['learning_rate']:
                # Group by epoch for learning rates
                epoch_lrs = {}
                for epoch, batch, lr in self.metrics_history['learning_rate']:
                    if epoch not in epoch_lrs:
                        epoch_lrs[epoch] = []
                    epoch_lrs[epoch].append(lr)
                
                lr_epochs = []
                lr_values = []
                for epoch in sorted(epoch_lrs.keys()):
                    lr_epochs.append(epoch)
                    lr_values.append(np.mean(epoch_lrs[epoch]))
                
                ax2.plot(lr_epochs, lr_values, 'g-', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
            
            # 3. Baseline Losses Comparison
            if (self.metrics_history['val_baseline_copy'] and 
                self.metrics_history['val_loss']):
                
                baseline_epochs = [x[0] for x in self.metrics_history['val_baseline_copy']]
                baseline_copy = [x[1] for x in self.metrics_history['val_baseline_copy']]
                val_epochs = [x[0] for x in self.metrics_history['val_loss']]
                val_losses = [x[1] for x in self.metrics_history['val_loss']]
                
                ax3.plot(baseline_epochs, baseline_copy, 'orange', label='Copy Baseline', linewidth=2)
                ax3.plot(val_epochs, val_losses, 'r-', label='Model Loss', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.set_title('Validation Loss vs Copy Baseline')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_yscale('log')
            
            # 4. Linear Interpolation Baseline (if available)
            if self.metrics_history['val_baseline_linear_interp']:
                linear_epochs = [x[0] for x in self.metrics_history['val_baseline_linear_interp']]
                linear_losses = [x[1] for x in self.metrics_history['val_baseline_linear_interp']]
                ax4.plot(linear_epochs, linear_losses, 'mediumpurple', label='Linear Interp Baseline', linewidth=2)
                if self.metrics_history['val_loss']:
                    val_epochs = [x[0] for x in self.metrics_history['val_loss']]
                    val_losses = [x[1] for x in self.metrics_history['val_loss']]
                    ax4.plot(val_epochs, val_losses, 'r-', label='Model Loss', linewidth=2)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss')
                ax4.set_title('Validation Loss vs Linear Interpolation Baseline')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'metrics_summary_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also create the original simple plot for compatibility
            if self.metrics_history['train_loss'] and self.metrics_history['val_loss']:
                plt.figure(figsize=(10, 6))
                
                # Use epoch-level data
                train_epochs = []
                train_losses = []
                val_epochs = []
                val_losses = []
                
                # Group by epoch for training losses
                epoch_train_losses = {}
                for epoch, batch, loss in self.metrics_history['train_loss']:
                    if epoch not in epoch_train_losses:
                        epoch_train_losses[epoch] = []
                    epoch_train_losses[epoch].append(loss)
                
                for epoch in sorted(epoch_train_losses.keys()):
                    train_epochs.append(epoch)
                    train_losses.append(np.mean(epoch_train_losses[epoch]))
                
                for epoch, loss in self.metrics_history['val_loss']:
                    val_epochs.append(epoch)
                    val_losses.append(loss)
                
                plt.plot(train_epochs, train_losses, 'b-', alpha=0.7, label='Train Loss')
                plt.plot(val_epochs, val_losses, 'r-', label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(self.log_dir / 'training_summary.png', dpi=150, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not create summary plots: {e}")
    
    def get_training_summary(self) -> dict:
        """Get comprehensive training summary from logged metrics."""
        try:
            summary: Dict[str, Any] = {}
            
            # Extract epoch-level metrics
            if self.metrics_history['train_loss']:
                # Group by epoch and calculate averages
                # Handle both formats: (epoch, batch, loss) from batches and (epoch, avg_loss) from epochs
                epoch_train_losses: Dict[int, List[float]] = {}
                for entry in self.metrics_history['train_loss']:
                    if len(entry) == 3:
                        # Format: (epoch, batch, loss) from log_training_batch
                        epoch, batch, loss = entry
                    elif len(entry) == 2:
                        # Format: (epoch, avg_loss) from log_training_epoch
                        epoch, loss = entry
                    else:
                        continue  # Skip malformed entries
                    
                    if epoch not in epoch_train_losses:
                        epoch_train_losses[epoch] = []
                    epoch_train_losses[epoch].append(loss)
                
                train_epochs = sorted(epoch_train_losses.keys())
                avg_train_losses = [np.mean(epoch_train_losses[epoch]) for epoch in train_epochs]
                
                summary['total_epochs'] = len(train_epochs)
                summary['final_train_loss'] = avg_train_losses[-1] if avg_train_losses else None
                summary['min_train_loss'] = min(avg_train_losses) if avg_train_losses else None
                summary['max_train_loss'] = max(avg_train_losses) if avg_train_losses else None
            
            if self.metrics_history['val_loss']:
                val_epochs = [x[0] for x in self.metrics_history['val_loss']]
                val_losses = [x[1] for x in self.metrics_history['val_loss']]
                
                summary['final_val_loss'] = val_losses[-1] if val_losses else None
                summary['min_val_loss'] = min(val_losses) if val_losses else None
                summary['max_val_loss'] = max(val_losses) if val_losses else None
                
                # Find best epoch
                if val_losses:
                    best_idx = np.argmin(val_losses)
                    summary['best_epoch'] = val_epochs[best_idx]
                    summary['best_val_loss'] = val_losses[best_idx]
            
            if self.metrics_history['learning_rate']:
                # Group by epoch for learning rates
                epoch_lrs: Dict[int, List[float]] = {}
                for epoch, batch, lr in self.metrics_history['learning_rate']:
                    if epoch not in epoch_lrs:
                        epoch_lrs[epoch] = []
                    epoch_lrs[epoch].append(lr)
                
                lr_values = [np.mean(epoch_lrs[epoch]) for epoch in sorted(epoch_lrs.keys())]
                summary['learning_rate_range'] = (min(lr_values), max(lr_values)) if lr_values else (0, 0)
                summary['final_learning_rate'] = lr_values[-1] if lr_values else None
            
            # Baseline comparisons
            if self.metrics_history['val_baseline_copy']:
                baseline_copy = [x[1] for x in self.metrics_history['val_baseline_copy']]
                summary['baseline_copy_avg'] = np.mean(baseline_copy) if baseline_copy else None
            
            if self.metrics_history['val_baseline_linear_interp']:
                baseline_linear_interp = [x[1] for x in self.metrics_history['val_baseline_linear_interp']]
                summary['baseline_linear_interp_avg'] = np.mean(baseline_linear_interp) if baseline_linear_interp else None
            
            return summary
            
        except Exception as e:
            print(f"Warning: Could not generate training summary: {e}")
            return {}
    
    def save_profiling_summary(self, profile_dir: Path):
        """Extract key metrics from profiling traces and save summary."""
        if not profile_dir.exists():
            return
        
        summary_data = []
        for trace_file in profile_dir.glob("trace_*.json"):
            try:
                # Simple parsing of trace file for key metrics
                # This is a basic implementation - could be enhanced
                with open(trace_file, 'r') as f:
                    # Skip full parsing, just record file info
                    file_size = trace_file.stat().st_size / (1024*1024)  # MB
                    summary_data.append({
                        'file': trace_file.name,
                        'size_mb': file_size,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"Warning: Could not process {trace_file}: {e}")
        
        # Save profiling summary
        if summary_data:
            with open(self.profiling_log, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['file', 'size_mb', 'timestamp'])
                writer.writeheader()
                writer.writerows(summary_data)
