"""Training engine for MoE models."""

import os
import time
import torch
import numpy as np
import wandb
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics


class MoE_Engine(BaseEngine):
    """Training engine with MoE-specific load balancing and routing statistics."""
    
    def __init__(self, track_expert_stats=True, **args):
        super(MoE_Engine, self).__init__(**args)
        self.track_expert_stats = track_expert_stats
        
        if not hasattr(self.model, 'num_experts'):
            raise ValueError("Model must be a MoE model with 'num_experts' attribute")
        
        self._logger.info(f'MoE Engine initialized with {self.model.num_experts} experts')
        self._logger.info(f'Expert names: {self.model.expert_names}')

        self.model.visualize_routing()
    
    def train_batch(self):
        """Train for one epoch with MoE-specific loss."""
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        train_balance_loss = []
        
        self._dataloader['train_loader'].shuffle()
        
        for X, label in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()
            
            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            
            pred, _, routing_weights = self.model(X, label, self._iter_cnt, return_analysis=True)
            pred, label = self._inverse_transform([pred, label])
            
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            
            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()
            
            # Add load balance + entropy loss
            if self.model.load_balance_weight > 0:
                balance_loss = self.model.compute_load_balance_loss(
                    routing_weights,
                    add_importance_loss=True,
                    add_entropy_loss=True,       # minimise per-node routing entropy
                    entropy_weight=self.model.routing_entropy_weight,
                )
                total_loss = loss + self.model.load_balance_weight * balance_loss
                train_balance_loss.append(balance_loss.item())
            else:
                total_loss = loss

            total_loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
            self._iter_cnt += 1
        
        if train_balance_loss:
            avg_balance_loss = np.mean(train_balance_loss)
            if self._iter_cnt % 100 == 0:
                self._logger.info(f'  Load balance loss: {avg_balance_loss:.6f}')
        
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
    
    def evaluate(self, mode):
        """Evaluate on val or test set."""
        self.model.eval()
        
        eval_loss = []
        eval_mape = []
        eval_rmse = []
        
        with torch.no_grad():
            for X, label in self._dataloader[f'{mode}_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                
                pred = self.model(X, label, self._iter_cnt, return_analysis=False)
                pred, label = self._inverse_transform([pred, label])
                
                mask_value = torch.tensor(0)
                if label.min() < 1:
                    mask_value = label.min()
                
                loss = self._loss_fn(pred, label, mask_value)
                mape = masked_mape(pred, label, mask_value).item()
                rmse = masked_rmse(pred, label, mask_value).item()
                
                eval_loss.append(loss.item())
                eval_mape.append(mape)
                eval_rmse.append(rmse)
        
        return np.mean(eval_loss), np.mean(eval_mape), np.mean(eval_rmse)
    
    def train(self):
        """Main training loop with early stopping."""
        self._logger.info('Start training MoE-STLoRA!')
        
        wait = 0
        min_loss = np.inf
        
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape,
                                           mvalid_loss, mvalid_rmse, mvalid_mape,
                                           (t2 - t1), (v2 - v1), cur_lr))
            
            if self._use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": mtrain_loss,
                    "train_rmse": mtrain_rmse,
                    "train_mape": mtrain_mape,
                    "val_loss": mvalid_loss,
                    "val_rmse": mvalid_rmse,
                    "val_mape": mvalid_mape,
                    "learning_rate": cur_lr
                }
                
                if self.track_expert_stats and (epoch + 1) % 5 == 0:
                    stats = self.model.get_expert_statistics()
                    for expert_name, weight in stats['expert_avg_weights'].items():
                        log_dict[f'expert_weight/{expert_name}'] = weight
                    for expert_name, count in stats['expert_max_nodes'].items():
                        log_dict[f'expert_nodes/{expert_name}'] = count
                    log_dict['routing_entropy'] = stats['routing_entropy']
                
                wandb.log(log_dict, step=epoch)

            if mvalid_loss < min_loss:
                self._logger.info(f'Validation loss decreased from {min_loss:.4f} to {mvalid_loss:.4f}. Saving model...')
                wait = 0
                min_loss = mvalid_loss
                self.save_model(self._save_path)
            else:
                wait += 1
                if wait >= self._patience:
                    self._logger.info(f'Early stopping at epoch {epoch + 1}')
                    break
        
        self._logger.info('\nFinal Expert Routing Statistics:')
        self.model.visualize_routing()
        
        self._logger.info('best valid_loss:{:.6f}'.format(min_loss))
        self.test()
    
    def test(self, model=None, save_path=None):
        """Evaluate on test set with per-horizon metrics."""
        if model is None:
            model = self.model
            self.load_model(self._save_path)
        
        model.eval()
        
        y_pred = []
        y_true = []
        
        self._logger.info('Start testing...')
        
        with torch.no_grad():
            for X, label in self._dataloader['test_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                
                pred = model(X, label, return_analysis=False)
                pred, label = self._inverse_transform([pred, label])
                y_pred.append(pred.squeeze(-1).cpu())
                y_true.append(label.squeeze(-1).cpu())
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        mask_value = torch.tensor(0)
        if y_true.min() < 1:
            mask_value = y_true.min()
        
        self._logger.info(f'Check mask value: {mask_value}')
        
        test_mae = []
        test_mape = []
        test_rmse = []
        
        for i in range(self.model.horizon):
            pred_t = y_pred[:, i, :]
            true_t = y_true[:, i, :]
            
            res = compute_all_metrics(pred_t, true_t, mask_value)
            log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
            test_mae.append(res[0])
            test_mape.append(res[1])
            test_rmse.append(res[2])
        
        log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
        
        if self._use_wandb:
            for i in range(self.model.horizon):
                wandb.log({
                    f"test_mae_horizon_{i+1}": test_mae[i],
                    f"test_rmse_horizon_{i+1}": test_rmse[i],
                    f"test_mape_horizon_{i+1}": test_mape[i]
                }, step=self._max_epochs)
            
            wandb.log({
                "test_mae_avg": np.mean(test_mae),
                "test_rmse_avg": np.mean(test_rmse),
                "test_mape_avg": np.mean(test_mape)
            }, step=self._max_epochs)
        
        test_results = {
            'mae': test_mae,
            'mape': test_mape,
            'rmse': test_rmse,
            'mae_avg': np.mean(test_mae),
            'mape_avg': np.mean(test_mape),
            'rmse_avg': np.mean(test_rmse)
        }
        
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.savez(os.path.join(save_path, f'predictions_s{self._seed}.npz'),
                    pred=y_pred.cpu().numpy(), true=y_true.cpu().numpy())
            self._logger.info(f'Predictions saved to {save_path}')
        
        return test_results
    
    def analyze_expert_routing(self, save_path=None):
        """Save detailed routing analysis."""
        self._logger.info('\nAnalyzing Expert Routing Patterns...')
        
        stats = self.model.get_expert_statistics(return_per_node=True)
        
        self.model.visualize_routing()
        
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            np.savez(os.path.join(save_path, f'routing_analysis_s{self._seed}.npz'),
                    per_node_routing=stats['per_node_routing'],
                    node_embeddings=stats['node_embeddings'],
                    preferred_expert=stats['preferred_expert'],
                    expert_names=self.model.expert_names)
            
            self._logger.info(f'Routing analysis saved to {save_path}')
        
        return stats
