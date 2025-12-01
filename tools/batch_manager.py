"""
Batch Processing Manager
Enhanced utilities for batch image processing with progress tracking and reporting
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


class BatchProcessor:
    """Manage batch processing of multiple images"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'batch_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_log = None
        self.start_time = None
        self.results = []
    
    def get_image_files(self, folder_path: str) -> List[Path]:
        """Get all supported image files from folder (recursive)"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        images = []
        for ext in self.SUPPORTED_FORMATS:
            images.extend(folder.rglob(f'*{ext}'))
            images.extend(folder.rglob(f'*{ext.upper()}'))
        
        # Remove duplicates
        images = list(set(images))
        return sorted(images)
    
    def start_batch(self, batch_name: str = None):
        """Start a new batch processing session"""
        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.batch_log = {
            'batch_name': batch_name,
            'start_time': datetime.now().isoformat(),
            'images': [],
            'total_count': 0,
            'success_count': 0,
            'failed_count': 0
        }
        self.start_time = time.time()
        self.results = []
    
    def add_result(self, filename: str, success: bool, output_path: str = None, 
                  error_msg: str = None, metrics: Dict = None):
        """Add processing result for an image"""
        result = {
            'filename': filename,
            'success': success,
            'output_path': str(output_path) if output_path else None,
            'error': error_msg,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        self.batch_log['images'].append(result)
        self.batch_log['total_count'] += 1
        
        if success:
            self.batch_log['success_count'] += 1
        else:
            self.batch_log['failed_count'] += 1
    
    def end_batch(self) -> Dict:
        """End batch processing and generate report"""
        if self.batch_log is None:
            return None
        
        elapsed_time = time.time() - self.start_time
        
        self.batch_log['end_time'] = datetime.now().isoformat()
        self.batch_log['total_time_seconds'] = round(elapsed_time, 2)
        self.batch_log['avg_time_per_image'] = round(
            elapsed_time / max(self.batch_log['total_count'], 1), 2
        )
        self.batch_log['success_rate'] = round(
            self.batch_log['success_count'] / max(self.batch_log['total_count'], 1) * 100, 2
        )
        
        # Save batch log
        log_path = self.output_dir / f"{self.batch_log['batch_name']}_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.batch_log, f, indent=2, ensure_ascii=False)
        
        return self.batch_log
    
    def get_batch_summary(self) -> Dict:
        """Get summary statistics of current batch"""
        if self.batch_log is None:
            return None
        
        return {
            'total_images': self.batch_log['total_count'],
            'successful': self.batch_log['success_count'],
            'failed': self.batch_log['failed_count'],
            'success_rate': f"{self.batch_log.get('success_rate', 0)}%",
            'total_time': f"{self.batch_log.get('total_time_seconds', 0)}s",
            'avg_time_per_image': f"{self.batch_log.get('avg_time_per_image', 0)}s"
        }


class ProcessingScheduler:
    """Schedule and optimize batch processing"""
    
    def __init__(self):
        self.queue = []
        self.completed = []
        self.failed = []
    
    def add_to_queue(self, image_path: str, priority: int = 1):
        """Add image to processing queue"""
        item = {
            'path': image_path,
            'priority': priority,
            'status': 'pending',
            'attempts': 0
        }
        self.queue.append(item)
        self.sort_queue()
    
    def sort_queue(self):
        """Sort queue by priority (higher priority first)"""
        self.queue.sort(key=lambda x: (-x['priority'], x['attempts']))
    
    def get_next_item(self):
        """Get next item from queue"""
        if self.queue:
            return self.queue.pop(0)
        return None
    
    def mark_completed(self, item: Dict, result: Dict = None):
        """Mark item as completed"""
        item['status'] = 'completed'
        item['result'] = result
        self.completed.append(item)
    
    def mark_failed(self, item: Dict, error: str):
        """Mark item as failed"""
        item['status'] = 'failed'
        item['error'] = error
        item['attempts'] += 1
        
        if item['attempts'] < 3:  # Retry up to 3 times
            self.add_to_queue(item['path'], item['priority'])
        else:
            self.failed.append(item)
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        total = len(self.completed) + len(self.failed) + len(self.queue)
        return {
            'total_items': total,
            'completed': len(self.completed),
            'failed': len(self.failed),
            'pending': len(self.queue),
            'success_rate': round(len(self.completed) / max(total, 1) * 100, 2)
        }


class ResultsAnalyzer:
    """Analyze batch processing results"""
    
    @staticmethod
    def analyze_batch_results(batch_log: Dict) -> Dict:
        """Analyze batch results and extract insights"""
        images = batch_log.get('images', [])
        
        # Collect metrics
        metrics_list = [img.get('metrics', {}) for img in images if img.get('success')]
        
        analysis = {
            'summary': {
                'total': batch_log['total_count'],
                'successful': batch_log['success_count'],
                'failed': batch_log['failed_count'],
                'success_rate': batch_log.get('success_rate', 0)
            },
            'performance': {
                'total_time': batch_log.get('total_time_seconds', 0),
                'avg_time_per_image': batch_log.get('avg_time_per_image', 0)
            }
        }
        
        if metrics_list:
            # Average metrics
            avg_metrics = {}
            for key in metrics_list[0].keys():
                values = [m.get(key, 0) for m in metrics_list if isinstance(m.get(key), (int, float))]
                if values:
                    avg_metrics[key] = round(sum(values) / len(values), 4)
            
            analysis['average_metrics'] = avg_metrics
        
        return analysis
    
    @staticmethod
    def generate_html_report(batch_log: Dict, output_path: str):
        """Generate HTML report from batch results"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Batch Processing Report - {batch_log['batch_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failed {{ color: red; }}
                .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
            </style>
        </head>
        <body>
            <h1>Batch Processing Report</h1>
            <p>Batch: <strong>{batch_log['batch_name']}</strong></p>
            <p>Date: <strong>{batch_log['start_time']}</strong></p>
            
            <div class="stats">
                <div class="stat-card">
                    <div>Total Images</div>
                    <div class="stat-value">{batch_log['total_count']}</div>
                </div>
                <div class="stat-card">
                    <div>Successful</div>
                    <div class="stat-value success">{batch_log['success_count']}</div>
                </div>
                <div class="stat-card">
                    <div>Failed</div>
                    <div class="stat-value failed">{batch_log['failed_count']}</div>
                </div>
                <div class="stat-card">
                    <div>Success Rate</div>
                    <div class="stat-value">{batch_log.get('success_rate', 0)}%</div>
                </div>
            </div>
            
            <div class="summary">
                <h3>Processing Time</h3>
                <p>Total Time: <strong>{batch_log.get('total_time_seconds', 0)}s</strong></p>
                <p>Average per Image: <strong>{batch_log.get('avg_time_per_image', 0)}s</strong></p>
            </div>
            
            <h3>Processed Images</h3>
            <table>
                <tr>
                    <th>Filename</th>
                    <th>Status</th>
                    <th>Output</th>
                    <th>Error</th>
                </tr>
        """
        
        for img in batch_log.get('images', []):
            status_class = 'success' if img['success'] else 'failed'
            status_text = '✓ Success' if img['success'] else '✗ Failed'
            output_path = img.get('output_path', '-')
            error = img.get('error', '-')
            
            html_content += f"""
                <tr>
                    <td>{img['filename']}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{output_path}</td>
                    <td>{error}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


if __name__ == '__main__':
    print("Batch Processing Manager")
    print("=" * 50)
    
    # Example usage
    # processor = BatchProcessor(output_dir='./batch_results')
    # processor.start_batch('test_batch')
    # processor.add_result('image1.jpg', True, './output/image1.jpg', metrics={'ssim': 0.95})
    # processor.add_result('image2.jpg', False, error_msg='Processing failed')
    # summary = processor.end_batch()
    # print(f"Summary: {summary}")
