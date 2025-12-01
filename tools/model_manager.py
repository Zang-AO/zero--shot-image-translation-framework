"""
Model Manager
Utilities for managing, comparing, and switching between different models
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple


class ModelManager:
    """Manage multiple model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.current_model = None
        self.load_models_info()
    
    def load_models_info(self):
        """Load information about available models"""
        models_info_file = self.checkpoint_dir / 'models_info.json'
        
        if models_info_file.exists():
            with open(models_info_file, 'r', encoding='utf-8') as f:
                self.models = json.load(f)
        else:
            self.scan_checkpoints()
    
    def scan_checkpoints(self):
        """Scan checkpoint directory and discover models"""
        self.models = {}
        
        for checkpoint_file in self.checkpoint_dir.glob('*.pth'):
            model_name = checkpoint_file.stem
            
            self.models[model_name] = {
                'name': model_name,
                'path': str(checkpoint_file),
                'size_mb': round(checkpoint_file.stat().st_size / (1024 * 1024), 2),
                'created': checkpoint_file.stat().st_ctime,
                'description': f'Model: {model_name}',
                'tags': self.infer_tags(model_name)
            }
        
        self.save_models_info()
    
    def infer_tags(self, model_name: str) -> List[str]:
        """Infer model tags from filename"""
        tags = []
        
        if 'best' in model_name.lower():
            tags.append('best')
        if 'final' in model_name.lower():
            tags.append('final')
        if 'mae' in model_name.lower():
            tags.append('mae-optimized')
        if 'ssim' in model_name.lower():
            tags.append('ssim-optimized')
        if 'gen' in model_name.lower():
            tags.append('generator')
        
        return tags
    
    def save_models_info(self):
        """Save models info to JSON file"""
        models_info_file = self.checkpoint_dir / 'models_info.json'
        
        with open(models_info_file, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=2, ensure_ascii=False)
    
    def get_model(self, model_name: str) -> Dict:
        """Get model information"""
        return self.models.get(model_name, None)
    
    def list_models(self, tag: str = None) -> List[Dict]:
        """List all available models, optionally filtered by tag"""
        models_list = list(self.models.values())
        
        if tag:
            models_list = [m for m in models_list if tag in m.get('tags', [])]
        
        return sorted(models_list, key=lambda x: x['size_mb'])
    
    def set_default_model(self, model_name: str) -> bool:
        """Set default model"""
        if model_name in self.models:
            self.current_model = model_name
            return True
        return False
    
    def add_model_description(self, model_name: str, description: str):
        """Add or update model description"""
        if model_name in self.models:
            self.models[model_name]['description'] = description
            self.save_models_info()
    
    def tag_model(self, model_name: str, tag: str):
        """Add tag to model"""
        if model_name in self.models:
            if tag not in self.models[model_name]['tags']:
                self.models[model_name]['tags'].append(tag)
            self.save_models_info()
    
    def compare_models(self, model1: str, model2: str) -> Dict:
        """Compare two models"""
        m1 = self.models.get(model1)
        m2 = self.models.get(model2)
        
        if not m1 or not m2:
            return None
        
        return {
            'model1': model1,
            'model2': model2,
            'size_diff_mb': round(m1['size_mb'] - m2['size_mb'], 2),
            'model1_tags': m1['tags'],
            'model2_tags': m2['tags'],
            'model1_created': m1['created'],
            'model2_created': m2['created']
        }


class ConfigManager:
    """Manage configuration files"""
    
    def __init__(self, config_dir: str = '.'):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.load_configs()
    
    def load_configs(self):
        """Load all available configurations"""
        for config_file in self.config_dir.glob('*.yaml'):
            config_name = config_file.stem
            
            with open(config_file, 'r', encoding='utf-8') as f:
                try:
                    config = yaml.safe_load(f)
                    self.configs[config_name] = {
                        'name': config_name,
                        'path': str(config_file),
                        'data': config,
                        'created': config_file.stat().st_ctime
                    }
                except:
                    pass
    
    def get_config(self, config_name: str) -> Dict:
        """Get configuration data"""
        if config_name in self.configs:
            return self.configs[config_name]['data']
        return None
    
    def list_configs(self) -> List[str]:
        """List all available configurations"""
        return list(self.configs.keys())
    
    def create_config_variant(self, base_config: str, variant_name: str, 
                            modifications: Dict) -> bool:
        """Create a variant of existing configuration"""
        base = self.get_config(base_config)
        if not base:
            return False
        
        variant = base.copy()
        variant.update(modifications)
        
        config_path = self.config_dir / f'{variant_name}.yaml'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(variant, f, default_flow_style=False)
        
        self.load_configs()
        return True


class PerformanceProfiler:
    """Profile model performance across different configurations"""
    
    def __init__(self):
        self.profiles = []
    
    def add_profile(self, model_name: str, config_name: str, metrics: Dict):
        """Add performance profile"""
        profile = {
            'model': model_name,
            'config': config_name,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.profiles.append(profile)
    
    def get_best_model(self, metric_name: str = 'ssim') -> Dict:
        """Get best performing model for given metric"""
        if not self.profiles:
            return None
        
        best = max(self.profiles, 
                  key=lambda x: x['metrics'].get(metric_name, 0))
        return best
    
    def get_fastest_model(self) -> Dict:
        """Get fastest model"""
        if not self.profiles:
            return None
        
        fastest = min(self.profiles, 
                     key=lambda x: x['metrics'].get('inference_time', float('inf')))
        return fastest
    
    def get_profiles_summary(self) -> Dict:
        """Get summary of all profiles"""
        if not self.profiles:
            return None
        
        models = {}
        for profile in self.profiles:
            model = profile['model']
            if model not in models:
                models[model] = {
                    'config': profile['config'],
                    'metrics': profile['metrics']
                }
        
        return models
    
    def save_profiles(self, output_path: str):
        """Save profiles to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)
    
    def load_profiles(self, input_path: str):
        """Load profiles from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            self.profiles = json.load(f)


from datetime import datetime


if __name__ == '__main__':
    print("Model Manager")
    print("=" * 50)
    
    # Example usage
    # manager = ModelManager('./checkpoints')
    # models = manager.list_models()
    # print(f"Available models: {len(models)}")
    # 
    # for model in models:
    #     print(f"  - {model['name']} ({model['size_mb']} MB)")
