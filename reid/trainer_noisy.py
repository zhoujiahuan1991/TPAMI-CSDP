'''by xkl'''
from __future__ import print_function, absolute_import
import time
import os
from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *
from sklearn.mixture import GaussianMixture
from reid.utils.make_loss import make_loss, loss_fn_kd
import copy
from reid.loss.noisy_loss import LabelRefineLoss, CoRefineLoss, Balance_ClassCrossEntropyLoss
from reid.loss.triplet import SoftTripletLoss
from sklearn.cluster import DBSCAN
from reid.metric_learning.distance import cosine_similarity
from reid.utils.faiss_rerank import compute_jaccard_distance
from scipy.interpolate import interp1d
from lreid_dataset.datasets.get_data_loaders_noisy import get_data_purify
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.interpolate import interp1d
from matplotlib import rcParams

def plot_PR(precision, recall, i, name):
    config = {
    "font.size": 15,
    "mathtext.fontset":'stix',
    }
    plt.rcParams.update({'font.size': 15})
    rcParams.update(config)

    plt.rcParams["font.family"] = "Times New Roman"
    
    auc_score = auc(recall, precision)
    plt.title(f'PR Curve (AUC = {auc_score:.2f})')


    colors=['r','b','darkgreen','m']
    recall_points = np.linspace(0, 1, 100)
    f = interp1d(recall, precision, fill_value="extrapolate", kind="linear")
    precision_interp = f(recall_points)
    plt.plot(recall_points,precision_interp,color=colors[i], label=name)
   
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('PR Curve: AUC:{}'.format())
    plt.grid(True)
    # plt.savefig(name)
def get_PR_pre(Clean_FLAG, prob):      
    if not isinstance(Clean_FLAG, torch.Tensor):
        Clean_FLAG==torch.tensor(Clean_FLAG)
    if not isinstance(prob, torch.Tensor):
        prob=torch.tensor(prob)
    indexes=torch.argsort(prob, descending=True)
    Clean_FLAG=Clean_FLAG[indexes]
    res=torch.cumsum(Clean_FLAG.float(),0)

    precision=res/torch.arange(1,len(res)+1)
    recall=res/len(res)
    return precision.numpy(), recall.numpy()

def select_self_pace(p_score, thre, epoch, stride, base_rate=0.5):
    max_keep=(p_score>thre).float().sum()
    ratios=(torch.range(1,1+stride)/stride)*(1-base_rate)+base_rate
    select_num=max_keep*ratios[epoch%stride]
    while thre<1:
        if select_num<(p_score>thre).float().sum():
            thre=thre+0.01
        else:
            break
    return p_score>thre
class Trainer(object):
    def __init__(self,args, model_list, model_old_list, num_classes,origin_data, writer=None, add_num=0):
        super(Trainer, self).__init__()        
        self.args = args
        self.model_list = model_list
        self.model_old_list = model_old_list
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.num_classes=num_classes

  
        self.criterion_tp=SoftTripletLoss(margin=0.0).cuda()      
        # self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
        self.LabelRefineLoss=LabelRefineLoss(aggregate=None)
        self.CoRefineLoss=CoRefineLoss(aggregate=None)
        self.Balance_ClassCrossEntropyLoss=Balance_ClassCrossEntropyLoss(num_classes)
        self.origin_labels=[x[1] for x in origin_data]
        self.origin_labels= torch.LongTensor(self.origin_labels)+add_num
        self.refine_labels=self.origin_labels.clone()

        scores_one_hot = torch.zeros(len(origin_data),num_classes).scatter_(1,self.origin_labels.unsqueeze(1),1).cuda()
        self.scores_one_hot=[scores_one_hot.clone(),scores_one_hot.clone()]
        self.gt_one_hot=scores_one_hot.clone()  # annotated ID
        self.pre_one_hot=scores_one_hot.clone() # predicted ID score

        self.cluster = DBSCAN(eps=0.5, min_samples=3, metric='precomputed', n_jobs=-1)
        self.psedo_dist=[]
        # self.psedo_dist_old_model=[]
        self.pseudo_labels_old=[]

        # self.losses=[torch.zeros(len(origin_data)).cuda(), torch.zeros(len(origin_data)).cuda()]

        self.clean_labels=[x[-1] for x in origin_data]  
        self.clean_labels= torch.LongTensor(self.clean_labels).cuda()


    def eval_old_dist(self):
        self.psedo_dist_old=[]        
        for m_id in range(self.args.n_model):
            pseudo_one_hot_old = self.pseudo_one_hot_old[m_id]
            n_ID=pseudo_one_hot_old.size(-1)  # all ID number
            psedo_dist_old=torch.zeros_like(pseudo_one_hot_old)      # recored the distinace from the label of each instance                 

            # Labels=self.origin_labels.cpu() # priginal noisy labels
            Labels=self.refine_labels.cpu() # rectified labels
            '''obtain the offset from the refined labels'''
            for id in set(Labels.tolist()):
                img_ids=torch.where(id==Labels)   
                psedo_dist_old[img_ids]=pseudo_one_hot_old[img_ids]-pseudo_one_hot_old[img_ids].mean(dim=0, keepdim=True)            
            '''transform into squared distance'''
            psedo_dist_old=(psedo_dist_old**2).sum(dim=-1) # upper limit of psedo_dist is 2            

            psedo_dist_old[self.pseudo_labels_old[m_id]==n_ID-1]=2   # set outlier distance as the upper limit                                
            '''store the distnce'''
            self.psedo_dist_old.append(psedo_dist_old)

            '''obtain the threshold according to EQ. 3'''
            Thre=2-2*self.args.T_o

            print("*********************")
            print("data keep ratio by old model:{},"                
                .format(
                        (psedo_dist_old<Thre).float().sum()/psedo_dist_old.size(0)
                        ))    

    def obtain_cluster(self, init_loader, add_num, model_list,dataset_name=None,res_list=None,epoch=0):
        if dataset_name:
            self.dataset_name=dataset_name       
        self.oldmodel_filter={}
        self.pseudo_labels=[]
        self.pseudo_one_hot=[]
        self.psedo_dist=[]
        for m_id, model in enumerate(model_list):
            all_loss=[]
            if res_list is not None:
                prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=res_list[m_id]
            else:
                prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(model,all_loss, init_loader, add_num, num_classes=self.num_classes)
                self.Clean_FLAG=Clean_FLAG
               
            rerank_dist = compute_jaccard_distance(All_features, k1=30, k2=6)
            # select & cluster images as training set of this epochs
            pseudo_labels = self.cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            print("*********cluster number:",num_cluster)
            pseudo_labels= torch.LongTensor(pseudo_labels)
            print("ratio of out lier",(pseudo_labels==-1).float().sum()/len(Clean_FLAG))
            print("out lier prediction accuracy",Clean_FLAG[pseudo_labels==-1].float().sum()/((pseudo_labels==-1).float().sum()+1e-5)) 
            pseudo_labels[torch.where(-1==pseudo_labels)]=num_cluster            # the outliers are noted as -1 by DBSCAN

            
            pseudo_one_hot = torch.zeros(len(pseudo_labels),num_cluster+1).scatter_(1,pseudo_labels.unsqueeze(1),1)

            
            Labels=self.refine_labels.cpu()
            # num_id=(Labels-add_num).max()+1

            psedo_dist=torch.zeros(len(pseudo_labels),500)

            pre_scores=self.pre_one_hot.cpu()[:,-500:]    # 预测分数
           
            for id in range(500):
                id_score=pre_scores[:, id]
                img_ids=id_score>0.5
                id_center=(pseudo_one_hot[img_ids]*id_score[img_ids].unsqueeze(1)).sum(dim=0)/(id_score[img_ids].sum()+1e-6)
                # img_ids=torch.where(id==Labels)
                id_diff=(pseudo_one_hot-id_center.unsqueeze(0))
                id_sim=2-(id_diff**2).sum(dim=-1)
                # psedo_dist[:,id]=2-id_score*id_sim
                psedo_dist[:,id]=1-(id_sim/2*self.args.cluster_weight+id_score*(1-self.args.cluster_weight))
                psedo_dist[:,id]=psedo_dist[:,id]*2
                
                
                # psedo_dist[img_ids]=pseudo_one_hot[img_ids]-pseudo_one_hot[img_ids].mean(dim=0, keepdim=True)            
            
            # psedo_dist=(psedo_dist**2).sum(dim=-1) # upper limit of psedo_dist is 2
            
            psedo_dist[pseudo_labels==num_cluster]=2   # set outlier distance as the upper limit
            psedo_dist=torch.tensor([(psedo_dist[i,x-add_num] if x>=add_num else 2) for i, x in enumerate(Labels)])
            self.pseudo_labels.append(pseudo_labels)
            self.pseudo_one_hot.append(pseudo_one_hot)
            self.psedo_dist.append(psedo_dist)

        # mean dists
        psedo_dist=(self.psedo_dist[0]+self.psedo_dist[1])/2
        self.psedo_dist.append(psedo_dist)
        # larger dists
        self.psedo_dist.append(torch.max(self.psedo_dist[0],self.psedo_dist[1]))
        # visualization  
        if epoch%self.args.plot_freq==0:   
            for nn, psedo_dist, name in zip([0,1,2,3],self.psedo_dist,['1','2','avg','max']):
                # num_cluster=len(pseudo_labels)
                # print("ratio of out lier",len(torch.where(pseudo_labels==num_cluster))/len(Clean_FLAG))
                recall=[]
                precise=[]
                for i in range(1,20):
                    thre=i/20
                    clean=psedo_dist<thre     
                    # clean=clean*(pseudo_labels<num_cluster)  

                    # print(clean.shape,Clean_FLAG.shape)     
                    
                    recall.append(Clean_FLAG[clean].sum()/Clean_FLAG.sum())
                    precise.append(Clean_FLAG[clean].sum()/(clean.sum()+1e-6))
                    if nn>0  or i%2:
                        continue
                    print("*********************",thre)
                    print("clean ratio:{},"
                        "selected data precise:{},"
                        "clean data recall:{},"
                        .format(Clean_FLAG.sum()/len(Clean_FLAG), 
                                Clean_FLAG[clean].sum()/(clean.sum()+1e-6),
                                Clean_FLAG[clean].sum()/Clean_FLAG.sum()
                                ))    
                    
                precise, recall=get_PR_pre(Clean_FLAG=Clean_FLAG,prob=(1-psedo_dist/2).cpu().tolist())
                plot_PR(precise,recall,nn, name)
            clean=(1-self.psedo_dist[1]/2)>self.args.T_c    # obtain positive (predicted clean) Flag
            TP=Clean_FLAG[clean].sum()  # obtain true-positive number
            
            TN=(1-Clean_FLAG)[~clean].sum()    # obtain true-negative number
            print("selection accuracy:", (TP+TN)/(len(clean)+1e-5))
            wrong_num=len(Clean_FLAG)-Clean_FLAG.sum()  # obtain wrongly predicted number

            print("*******wrong label number: {}, filter number: {}, filter ratio:{}*******".format(
                wrong_num, TN, TN/(wrong_num+1e-5)
            ))
            
            os.makedirs(self.args.logs_dir+'/'+self.dataset_name,exist_ok=True)
            save_name=self.args.logs_dir+'/{}/PR-curves-{}.png'.format(self.dataset_name,epoch)
            print("saving PR curve to ", save_name)
            plt.legend()
            plt.savefig(save_name)
            plt.clf()
        
        if 0==epoch:
            self.pseudo_labels_old=copy.deepcopy(self.pseudo_labels)
            self.pseudo_one_hot_old=copy.deepcopy(self.pseudo_one_hot)
        self.eval_old_dist()    
 
           
    def decode_pre(self, model,eval_loader, add_num=0):
        
        all_loss=[]
        prob1,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits=eval_train(model,all_loss,eval_loader, add_num=add_num, num_classes=self.num_classes)  

        print("*********************")
        print("noisy ratio:{},"  
                .format(Clean_FLAG.sum()/len(Clean_FLAG)                       
                        ))
        pre_ids=torch.softmax(All_logits,dim=1)[:,add_num:].argmax(dim=-1)  # obtain the predicted ID
        T_pre=pre_ids==(Clean_IDS)   # obtain the correctly predicted ID
        print(
                "predicted ID precise:{},"
                "noisy ID recall:{},"
                .format(
                        T_pre.float().sum()/len(Clean_FLAG),
                        T_pre[~Clean_FLAG.bool()].float().sum()/(~Clean_FLAG.bool()).float().sum()
                        ))
        print("*********************")

        # for thre in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95,0.98, 0.99]:
        #     GMM_flag=prob1>thre
        #     if not isinstance(GMM_flag, torch.Tensor):
        #         GMM_flag=torch.tensor(GMM_flag)

        #     print(
        #             "GMM Threshold: {}, GMM keep ratio:{},"
        #             "GMM keep precise:{},"
        #             .format(thre,
        #                     GMM_flag.float().sum()/len(Clean_FLAG),
        #                     GMM_flag[Clean_FLAG.bool()].float().sum()/len(Clean_FLAG)
        #                     ))
        #     print("*********************")


        return prob1,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits

    def train(self, epoch, data_loader_train,  optimizer_list, training_phase,
              train_iters=200, add_num=0, weight_r=None ,eval_loader=None ,dataset=None          
              ):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tr = [AverageMeter(),AverageMeter()]
        losses_relation = [AverageMeter(),AverageMeter()]
        # warm up 
        if True:      
            p_score=(2-self.psedo_dist[1])/2    # Eq. 3
            # Keep=p_score.cpu()>self.args.T_c    
            Keep=select_self_pace(p_score.cpu(), self.args.T_c, epoch, self.args.cluster_stride,base_rate=self.args.base_rate)       
            Pseudo=(self.refine_labels-add_num).clamp(min=0) # obtain the refined label in currect dataset
           
            ''' Eq. 4 '''
            if epoch>=self.args.warm_up_epochs:   
                data_loader_train,eval_loader=get_data_purify(dataset, height=256, width=128, batch_size=self.args.batch_size,
                             workers=self.args.workers, num_instances=self.args.num_instances, Keep=Keep, Pseudo=Pseudo)
            if epoch%self.args.cluster_stride==0:
                print("*********************")
                Pseudo=Pseudo.to(self.clean_labels.device)
                Is_purified_true=(self.clean_labels==Pseudo).float()
                Keep=Keep.to(self.clean_labels.device)
                print(
                        "purified data precise:{},"
                        "Keeped ratio:{},"
                        "Keeped data precise:{},"
                        .format(Is_purified_true.sum()/len(Pseudo), 
                                Keep.sum()/(len(Keep)+1e-6), 
                                Is_purified_true[Keep].sum()/(Keep.sum()+1e-6),                           
                                ))
        
        if epoch%self.args.cluster_stride==0 and True:
            res_list=[]     
            self.pre_one_hot=torch.zeros_like(self.pre_one_hot)
            '''obtain prediction of previous model'''
            for m_id in range(self.args.n_model):             
                res=self.decode_pre(self.model_list[m_id],eval_loader, add_num)
                res_list.append(res)  
                self.pre_one_hot+=torch.softmax(res[-1].cuda(), -1)
            self.pre_one_hot=self.pre_one_hot/self.args.n_model

            weight=self.args.w_l    
            Scores=self.pre_one_hot*(1-weight)+self.gt_one_hot*weight  # Eq. 8 

            self.refine_labels=torch.argmax(Scores, -1) # Eq. 8 
         

            label_flag=(self.origin_labels-add_num)==self.clean_labels.cpu()  
            wrong_num=len(label_flag)-label_flag.float().sum()  
            rect_flag=(self.refine_labels-add_num).cpu()==self.clean_labels.cpu()   
            num_correct_rect=rect_flag[~label_flag].float().sum()
            print("******wrong label number: {}, rectify correct number: {}, rectify ratio:{}*****".format(
                wrong_num, num_correct_rect,num_correct_rect/(wrong_num+1e-5)))



            if training_phase>0 and epoch>0:
                self.obtain_cluster(eval_loader, add_num, self.model_list, res_list=res_list,epoch=epoch)
                self.psedo_dist[0]=self.psedo_dist[0].cuda()
                self.psedo_dist[1]=self.psedo_dist[1].cuda()

             

        self.model_list[0].train()    
        self.model_list[1].train()    
        
        for m_id in range(self.args.n_model):
            # freeze the bn layer totally
            for m in self.model_list[m_id].module.base.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if m.weight.requires_grad == False and m.bias.requires_grad == False:
                        m.eval()    
            
            
        end = time.time()  
        for i in range(train_iters):    
            try:            
                train_inputs = data_loader_train.next()
            except:
                continue
            data_time.update(time.time() - end)


            s_inputs, targets, cids, image_id,clean_pid=self._parse_data(train_inputs)         

            targets += add_num
            s_features_1, bn_feat_1, cls_outputs_1, feat_final_layer_1 = self.model_list[0](s_inputs)
            s_features_2, bn_feat_2, cls_outputs_2, feat_final_layer_2 = self.model_list[1](s_inputs)
           
              

            if epoch <20:
                loss_ce1 = self.LabelRefineLoss(cls_outputs_1,  targets, weight_r[0])  # ID loss
                loss_ce2 = self.LabelRefineLoss(cls_outputs_2,  targets, weight_r[1])  # ID loss
                # self.losses[0][image_id]=1/(loss_ce1.detach()+1)
                # self.losses[1][image_id]=1/(loss_ce2.detach()+1)
                
                loss_ce1=loss_ce1.mean()
                loss_ce2=loss_ce2.mean()
                
                loss_tp_1 = self.criterion_tp(s_features_1, s_features_1, targets)
                loss_tp_2 = self.criterion_tp(s_features_2, s_features_2, targets)
                loss_1 = loss_ce1+loss_tp_1
                loss_2=loss_ce2+loss_tp_2
                losses_ce[0].update(loss_ce1.item(), s_inputs.size(0))
                losses_tr[0].update(loss_tp_1.item(), s_inputs.size(0))
                losses_ce[1].update(loss_ce2.item(), s_inputs.size(0))
                losses_tr[1].update(loss_tp_2.item(), s_inputs.size(0))
            else:
                loss_ce1 = self.CoRefineLoss(cls_outputs_1, cls_outputs_2.detach(), targets, 1)
                loss_ce2 = self.CoRefineLoss(cls_outputs_2, cls_outputs_1.detach(), targets, 1)
                
                loss_ce1=loss_ce1.mean()
                loss_ce2=loss_ce2.mean()                                
                
                loss_tp_1 = self.criterion_tp(s_features_1, s_features_1, targets)
                loss_tp_2 = self.criterion_tp(s_features_2, s_features_2, targets)
                loss_1 = loss_ce1+loss_tp_1
                loss_2=loss_ce2+loss_tp_2
                losses_ce[0].update(loss_ce1.item(), s_inputs.size(0))
                losses_tr[0].update(loss_tp_1.item(), s_inputs.size(0))
                losses_ce[1].update(loss_ce2.item(), s_inputs.size(0))
                losses_tr[1].update(loss_tp_2.item(), s_inputs.size(0))

                # loss_sym_1 =self.Balance_ClassCrossEntropyLoss(cls_outputs_1, targets)
                # loss_sym_2 =self.Balance_ClassCrossEntropyLoss(cls_outputs_2, targets)
                # loss_1=loss_1+loss_sym_1*self.args.sym_weight
                # loss_2=loss_2+loss_sym_2*self.args.sym_weight
      

            if len(self.model_old_list):
                Thre=2-self.args.T_o
                Keep1=(self.psedo_dist_old[0][image_id]<Thre).to(s_inputs.device)
                Keep2=(self.psedo_dist_old[1][image_id]<Thre).to(s_inputs.device) 
                                           

                af_loss_1=self.anti_forgetting(self.model_old_list[0],s_inputs,cls_outputs_1,s_features_1, targets,feat_final_layer_1,Keep1)
                af_loss_2=self.anti_forgetting(self.model_old_list[1],s_inputs,cls_outputs_2,s_features_2, targets,feat_final_layer_2,Keep2)
                # print("antiforgetting loss",af_loss_1)

                loss_1+=af_loss_1
                loss_2+=af_loss_2

                losses_relation[0].update(af_loss_1.item(), s_inputs.size(0))
                losses_relation[1].update(af_loss_2.item(), s_inputs.size(0))

                                                
         

            optimizer_list[0].zero_grad()
            loss_1.backward()
            optimizer_list[0].step()      


            optimizer_list[1].zero_grad()
            loss_2.backward()
            optimizer_list[1].step()       

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce[0].val,
                        global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr[0].val,
                        global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                        global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Loss_ce1 {:.3f} ({:.3f}) Loss_ce2 {:.3f} ({:.3f})\t'
                    'Loss_tp1 {:.3f} ({:.3f}) Loss_tp2 {:.3f} ({:.3f})\t'
                    'Loss_relation1 {:.3f} ({:.3f}) Loss_relation2 {:.3f} ({:.3f})\t'
                    .format(epoch, i + 1, train_iters,
                            batch_time.val, batch_time.avg,
                            losses_ce[0].val, losses_ce[0].avg,losses_ce[1].val, losses_ce[1].avg,
                            losses_tr[0].val, losses_tr[0].avg,losses_tr[1].val, losses_tr[1].avg,
                            losses_relation[0].val, losses_relation[0].avg,losses_relation[1].val, losses_relation[1].avg,
                ))     
        weight_r = [1. / (1. + losses_ce[0].avg), 1. / (1. + losses_ce[1].avg)]   
        return  weight_r

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        "img, image_id, pid, camid, clean_pid"
        imgs, image_id, pids, cids, clean_pid = inputs
        
        inputs = imgs
        targets = pids.cuda()
        # image_id=image_id.
        return inputs, targets, cids, image_id,clean_pid
    def cal_KL(self,Affinity_matrix_new, Affinity_matrix_old,targets, rectify=True):
        if rectify:
            Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
            Gts = Gts.float().to(targets.device)
            '''obtain TP,FP,TN,FN'''
            attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
            attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

            '''# prediction is correct on old model'''
            Old_Keep = attri_old['TN'] + attri_old['TP']
            Target_1 = Affinity_matrix_old * Old_Keep
            '''# prediction is false on old model but correct on mew model'''
            New_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
            Target_2 = Affinity_matrix_new * New_keep
            '''# both missed correct person'''
            Hard_pos = attri_new['FN'] * attri_old['FN']
            Thres_P = torch.maximum(attri_new['Thres_P'], attri_old['Thres_P'])
            Target_3 = Hard_pos * Thres_P

            '''# both false wrong person'''
            Hard_neg = attri_new['FP'] * attri_old['FP']
            Thres_N = torch.minimum(attri_new['Thres_N'], attri_old['Thres_N'])
            Target_4 = Hard_neg * Thres_N

            Target__ = Target_1 + Target_2 + Target_3 + Target_4
            Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization
        else:
            Target=Affinity_matrix_old

        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)

        return divergence.sum(-1)

    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))

        FN=Gts-TP

        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris
    def anti_forgetting(self, old_model,s_inputs,cls_outputs,s_features, targets,feat_final_layer, Keep=None):
        divergence=0
        loss=0
        old_model.eval()
        with torch.no_grad():            
            s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
        if isinstance(s_features_old, tuple):
            s_features_old=s_features_old[0]
        
        if Keep is not None and Keep.sum()>0:
            s_features=s_features[Keep]
            targets=targets[Keep]
            s_features_old=s_features_old[Keep]

        Affinity_matrix_new = self.get_normal_affinity(s_features)
        Affinity_matrix_old = self.get_normal_affinity(s_features_old)
        divergence = self.cal_KL(Affinity_matrix_new, Affinity_matrix_old, targets,rectify=False)
        
        divergence=divergence.mean()
            
        loss = loss + divergence * self.AF_weight
        return loss


def eval_train(model,all_loss, eval_loader, add_num=0, num_classes=500):  
    CE = nn.CrossEntropyLoss(reduction='none').cuda()  
    model.eval()
    losses = torch.zeros(50000)    
    Clean_IDS=torch.zeros(50000)
    Noisy_IDS=torch.zeros(50000)
    Clean_FLAG=torch.zeros(50000)
    All_features=torch.zeros(50000,2048)
    All_logits=torch.zeros(50000,num_classes)
    Count=0
    "img, image_id, pid, camid, clean_pid"
    with torch.no_grad():
        for i, (imgs, image_id, pids, cids, clean_pid) in enumerate(eval_loader):
        # for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            index=image_id
            inputs=imgs
            targets=pids+add_num
            inputs, targets = inputs.cuda(), targets.cuda() 
            # _, _, outputs = model(inputs) 
            s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = model(inputs, get_all_feat=True)
            Count+=len(imgs)
            loss = CE(cls_outputs_old, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]    
                Clean_IDS[index[b]]=clean_pid[b]     
                Noisy_IDS[index[b]]=pids[b] 
                Clean_FLAG[index[b]]=clean_pid[b] ==  pids[b] 
                All_features[index[b]]=s_features_old[b].detach().cpu().clone()
                All_logits[index[b]]=cls_outputs_old[b].detach().cpu().clone()
    losses=losses[:Count]
    Clean_IDS=Clean_IDS[:Count]
    Noisy_IDS=Noisy_IDS[:Count]
    Clean_FLAG=Clean_FLAG[:Count]
    All_features=All_features[:Count]
    All_logits=All_logits[:Count]
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    input_loss = losses.reshape(-1,1)
    
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss, Clean_IDS, Noisy_IDS, Clean_FLAG, All_features, All_logits

