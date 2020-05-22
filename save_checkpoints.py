from tensorflow.keras.callbacks import Callback
from IPython import embed
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
from datetime import datetime

class SaveCheckpoints(Callback):
    def __init__(self, save_optimizer = True, frequency = 1, time_unit = 'epoch', save_best = True, monitor_metric = 'val_loss', monitor_criteria = 'auto'):
            
        self.save_optimizer = save_optimizer
        self.frequency = frequency
        self.time_unit = time_unit
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.monitor_criteria = monitor_criteria
        self.model_path = Path('checkpoints')
        #self.metadata_path = str(Path(self.checkpoints_path,metadata_filename).absolute())
        
        if self.monitor_criteria == 'auto':
            if 'acc' in self.monitor_metric or self.monitor_metric.startswith('fmeasure'):
                self.monitor_criteria = 'max'
            else:
                self.monitor_criteria = 'min'
                
        if self.monitor_criteria == 'max':
            self.current_best = -np.Inf
        else:
            self.current_best = np.Inf
        
        self.step = 0
        self.epoch = 0
        
        self.checkpoints_history = []
    
    def on_epoch_end(self, batch, logs):
        if self.time_unit == 'epoch':
            self.current_metric = logs.get(self.monitor_metric, None)
            if self.save_best:
                if self.current_metric:
                    if self.monitor_criteria == 'max' and self.current_metric>self.current_best:
                        self.current_best = self.current_metric
                        self.save(mode = 'epoch')
                    elif self.monitor_criteria == 'min' and self.current_metric<self.current_best:
                        self.current_best = self.current_metric
                        self.save(mode = 'epoch')                    
            else:
                if self.epoch%self.frequency == 0:
                    self.save(mode='epoch')
        self.epoch += 1
        
    def save(self, mode):
        
        ckpt_path = Path(self.model_path,'checkpoints')
        if not ckpt_path.exists():
            ckpt_path.mkdir(parents=True)
            
        weights = {}
        
        for layer in self.model.layers:
            if layer.name not in self.model.input_names:
                weights[layer.name] = layer.get_weights()
            
        if mode == 'batch':
            current_step = self.step
        elif mode == 'epoch':
            current_step = self.epoch
        else:
            raise Exception("Unknown mode")
        
        checkpoint_history = {'mode': mode, 'step': current_step}
        
        if self.monitor_metric and self.current_metric:
            str_metric = "{}: {}".format(self.monitor_metric,self.current_metric)
            checkpoint_history['metric'] = self.monitor_metric
            checkpoint_history['metric_val'] = self.current_metric
        else:
            str_metric = ""
        filename = "{}-{}-{}".format(mode,current_step,str_metric)
        
        weights_dir = Path(self.model_path,'checkpoints',filename+'.weights')
        pickle.dump(weights, open(str(weights_dir.absolute()),"wb"))
        checkpoint_history['weights_path'] = str(weights_dir.absolute())
        
        if self.save_optimizer:
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            if symbolic_weights:
                opt_weights = tf.keras.backend.batch_get_value(symbolic_weights)
            
            opt_dir = Path(self.model_path,'checkpoints',filename+'.opt')
            pickle.dump(opt_weights, open(str(opt_dir.absolute()),"wb"))
            checkpoint_history['opt_weights_path'] = str(opt_dir.absolute())
            
        self.checkpoints_history.append(checkpoint_history)
        metadata_dir = Path(self.model_path,'checkpoints','metadata')
        pickle.dump(self.checkpoints_history,open(str(metadata_dir.absolute()),"wb"))
        
    def on_batch_end(self, batch, logs):       
        if self.time_unit == 'batch' and self.step%self.frequency == 0:
            self.current_metric = logs.get(self.monitor_metric, None)
            self.save(mode = 'batch')
        self.step += 1