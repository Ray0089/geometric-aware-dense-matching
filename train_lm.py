from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import time
import tqdm
import argparse
import numpy as np
import torch
import torch.optim as optim
import logging
import torch.backends.cudnn as cudnn
import config.lmo_cfg as cfg
import models.pytorch_utils as pt_utils
from models.geoMatch import GeoMatch
from torch.optim.lr_scheduler import CyclicLR
from evaluator import Evaluator
import ref
import shutil
from datasets.lm.linemod_pbr import LMDataset as dataset_desc
from utils.logging import get_logger
#from apex.parallel import convert_syncbn_model
logger = get_logger(__name__)
logger.setLevel(logging.INFO)
DEBUG = False

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-weight_decay", type=float, default=0,
    help="L2 regularization coeff [default: 0.0]",
)
parser.add_argument(
    "-lr", type=float, default=1e-2,
    help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay", type=float, default=0.5,
    help="Learning rate decay gamma [default: 0.5]",
)
parser.add_argument(
    "-decay_step", type=float, default=2e5,
    help="Learning rate decay step [default: 20]",
)
parser.add_argument(
    "-bn_momentum", type=float, default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
)
parser.add_argument(
    "-bn_decay", type=float, default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]",
)
parser.add_argument(
    "-checkpoint", type=str, default=None,
    help="Checkpoint to start from"
)
parser.add_argument(
    "-state", type=str, default='eval',
    help="train or evaluate the model"
)
parser.add_argument(
    "-dataset_name", type=str, default='lmo',
    help="train or evaluate the model"
)
parser.add_argument(
    "-cls_id", type=int, default=5,
    help="train or evaluate the model"
)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--gpu_id', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=8, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument('--deterministic', action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


lr_clip = 1e-5
bnm_clip = 1e-2

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id+args.local_rank)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }
def load_checkpoint(model = None, 
                    optimizer = None, 
                    device =  torch.device('cuda:{}'.format(args.local_rank)),
                    filename = "checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=device)
        epoch = checkpoint["epoch"]
        print(epoch)
        it = checkpoint.get("it", 0.0)
        #best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            ck_st = checkpoint['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            model.load_state_dict(ck_st)
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        #amp.load_state_dict(checkpoint["amp"])
        print("==> Done")
        return it, epoch
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def save_checkpoint(
        state, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    common_file = filename[:-11] + '.pth.tar'
    shutil.copyfile(filename, common_file)

    #teval = TorchEval()

def model_fn_dec(model, data):      
    cu_dt = {}
    for key in data.keys():
        # if key == 'det':
        #     cu_dt[key] = data[key]
        # if type(data[key]) == list:
        #     cu_dt[key] = data[key]
        if data[key].dtype in [np.float32, np.uint8]:
            cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
        elif data[key].dtype in [np.int32, np.uint32]:
            cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
        elif data[key].dtype in [torch.uint8, torch.float32]:
            cu_dt[key] = data[key].float().cuda()
        elif data[key].dtype in [torch.int32, torch.int16]:
            cu_dt[key] = data[key].long().cuda()

    end_points = model(cu_dt)
    
    return end_points
        
class Trainer(object):

    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )

        self.checkpoint_name, self.best_name = checkpoint_name, best_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    def train(
        self,
        start_epoch,
        n_epochs,
        train_loader,
        train_sampler
    ):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        #print("Totally train %d iters per gpu." % tot_iter)
        iters = 0
        
        with tqdm.tqdm(range(start_epoch,n_epochs), desc="epochs") as tbar:
            for epoch in tbar:
                if epoch > n_epochs:
                    break
                if args.local_rank == 0:
                    print(f'start epoch:{epoch}')
                np.random.seed()
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                sum_loss = 0.0
                sum_match_loss = 0.0
                sum_seg_loss = 0.0
                start_time = time.time()
               
                for it,batch in enumerate(train_loader):
                    self.model.train()
                
                    out_dict = self.model_fn(self.model, batch)
                    loss = out_dict['loss']
                    sum_loss += loss.item()
                    sum_match_loss += out_dict['match_loss'].item()
                    sum_seg_loss += out_dict['seg_loss'].item()
          
                    if (it+1) % 100 == 0:
                        print("avg_loss:{:.4f} seg: {:.4f} match: {:.4f}".format(sum_loss / 100,sum_seg_loss / 100,sum_match_loss/100))
                        sum_loss = 0.0
                        sum_match_loss = 0.0
                        sum_seg_loss = 0.0

                        end = time.time()
                        print(f'time cost:{end - start_time } s')
                        start_time = time.time()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
               
                    self.lr_scheduler.step()
                    self.bnm_scheduler.step()

                if (epoch+1) %10 ==0:
                    if args.local_rank == 0:
                        save_checkpoint(
                                    checkpoint_state(self.model, self.optimizer, 0, epoch),
                                    filename=self.checkpoint_name + '_' + f'{epoch:02d}')
                    
def cal_result_multimodel(data,model_dict):
    bs,_ = data['labels'].shape
    inst = {}
    result = {}
    for i in range(bs):
        for key in data.keys():
            if type(data[key]) == list:
                continue
            inst[key] = data[key][i].unsqueeze(0)
        
        out = model_dict[inst['cls_id'].item()](inst)
        for key in out.keys():
            if key in result:
                result[key] = torch.cat([result[key],out[key]])
            else:
                result[key] = out[key]
    return result


def test(ref_data):

    
    val_ds = dataset_desc(cfg.DATASETS,'test')
    val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.DATALOADER['VAL_BATCH_SIZE'], shuffle=False,  #cfg.DATALOADER['VAL_BATCH_SIZE']
            drop_last=False, num_workers=4)
    peval = Evaluator(cfg,args.dataset_name,False,'output',val_ds)
                
    model_dict = {}
    device = torch.device('cuda:{}'.format(args.local_rank))
    print('local_rank:', args.local_rank)
    cu_dt = {}

    for id in cfg.DATASETS['OBJS'].keys():
        model = GeoMatch(cfg.MODEL,id)
        model.to(device)
        model_dict[id] = model
        filename = os.path.join(cfg.MODEL['checkpoints'],cfg.DATASETS['OBJS'][id],'geomatch.pth.tar')
        if os.path.exists(filename):
            checkpoint_status = load_checkpoint(
                model_dict[id], filename = filename[:-8]
            )
        model_dict[id].eval()
    with torch.no_grad():
        #print(val_loader.shape)
        for i, data in enumerate(tqdm.tqdm(val_loader)):

            start_compute_time = time.perf_counter()

            for key in data.keys():
                if key == 'det':
                    cu_dt[key] = data[key]
                if type(data[key]) == list:
                    cu_dt[key] = data[key]
                elif data[key].dtype in [np.float32, np.uint8]:
                    cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
                elif data[key].dtype in [np.int32, np.uint32]:
                    cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
                elif data[key].dtype in [torch.uint8, torch.float32]:
                    cu_dt[key] = data[key].float().cuda()
                elif data[key].dtype in [torch.int32, torch.int16]:
                    cu_dt[key] = data[key].long().cuda()
              
            end_points = cal_result_multimodel(cu_dt,model_dict)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            bs,_,_ = cu_dt['cld_rgb_nrm'].shape

            outputs = [{} for _ in range(bs)]
            for _i in range(len(outputs)):
                outputs[_i]["time"] = cur_compute_time

            peval.process(cu_dt,end_points,outputs)
        peval.evaluate()

def train(args,ref_data):
    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
    torch.cuda.set_device(args.local_rank)
 
    if not DEBUG:
        torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://',
            )
        #torch.manual_seed(0)
    
    if args.local_rank == 0:
        logger.info(f'Training object {args.cls_id}')
    

    cfg.DATASETS["SELECTED_OBJ_ID"] = args.cls_id
    train_ds = dataset_desc(cfg.DATASETS ,'train')

    if not DEBUG:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.DATALOADER['TRAIN_BATCH_SIZE'], shuffle=False,
            drop_last=True, num_workers=12, sampler=train_sampler,worker_init_fn = worker_init_fn
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.DATALOADER['TRAIN_BATCH_SIZE'], shuffle=False,
            drop_last=True, num_workers=2
        )

    model = GeoMatch(cfg.MODEL,args.cls_id)
    if not DEBUG:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = optim.Adam(
        model.parameters(),lr=0.0001, weight_decay=args.weight_decay
    )
    device = torch.device('cuda:{}'.format(args.local_rank))
    print('local_rank:', args.local_rank)
    model.to(device)
    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    start_epoch = 0

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_dir = os.path.join(args.checkpoint,cfg.DATASETS['OBJS'][args.cls_id],'geomatch')
        checkpoint_status = load_checkpoint(
            model, optimizer, device,filename=checkpoint_dir,
        )
        if checkpoint_status is not None:
            it, start_epoch = checkpoint_status
        # if args.eval_net:
        #     assert checkpoint_status is not None, "Failed loadding model."


    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
    )
    clr_div = 6
    lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-6, max_lr=1e-3,
            cycle_momentum=False,
            step_size_up=cfg.SOLVER["TOTAL_EPOCHS"] * len(train_ds) // cfg.DATALOADER['TRAIN_BATCH_SIZE'] // clr_div // args.gpus,
            step_size_down= cfg.SOLVER["TOTAL_EPOCHS"] * len(train_ds) // cfg.DATALOADER['TRAIN_BATCH_SIZE'] // clr_div // args.gpus,
            mode='triangular'
        )
  
    bnm_lmbd = lambda it: max(
        args.bn_momentum * args.bn_decay ** (int(it * cfg.DATALOADER['TRAIN_BATCH_SIZE'] / args.decay_step)),
        bnm_clip,
    )


    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    checkpoint_fd = ref_data.log_model_dir + '/' + cfg.DATASETS['OBJS'][args.cls_id]
    ensure_fd(checkpoint_fd)
    trainer = Trainer(
        model,
        model_fn_dec,
        optimizer,
        checkpoint_name=os.path.join(checkpoint_fd, "geomatch"),
        best_name=os.path.join(checkpoint_fd, "geomatch_best"),
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
    )

    trainer.train(start_epoch, cfg.SOLVER["TOTAL_EPOCHS"], train_loader,train_sampler)

if __name__ == "__main__":
    args.world_size = args.gpus * args.nodes
    ref_data = ref.__dict__[args.dataset_name]
    if args.state == 'train':
        train(args,ref_data)
    else:
        test(ref_data)
