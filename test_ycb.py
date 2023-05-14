
from models.geoMatch import GeoMatch
import torch
import config.ycbv_cfg as cfg
from datasets.ycbv.ycbv import YCBVDataset
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()
def main():
    # from common import ConfigRandLA,ConfigRandLA3D
    # rndla_cfg = ConfigRandLA
    # rnd3d = ConfigRandLA3D

    # config = Config(ds_name='linemod', cls_type='ape')
    cls_id = 16
    cfg.DATASETS["SELECTED_OBJ_ID"] = cls_id
    
    model = GeoMatch(cfg.MODEL,cls_id)
    
     # config.mini_batch_size = 1
    global DEBUG
    DEBUG = True
    ds = {}
    #ds['train'] = YCBVDataset(cfg.DATASETS,'train')
    ds['train'] = YCBVDataset(cfg.DATASETS,'train')
    # ds['val'] = Dataset('validation')
    # ds['test'] = Dataset('test', DEBUG=False)
    # idx = dict(
    #     train=0,
    #     val=0,
    #     test=0
    # )
    train_loader = torch.utils.data.DataLoader(
            ds['train'], batch_size=1, shuffle=True,
            drop_last=True, num_workers=1, pin_memory=True
        )
    optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0001, weight_decay=0.95
        )
   ## inputs = ds['train'].__getitem__(0)
    #writer.add_graph(model,(inputs,))

    model = model.to('cuda')

    for data in train_loader:
       
        model.train()
        #model.eval()
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        
        #writer.add_graph(model,(cu_dt,))
        #writer.close()



        optimizer.zero_grad()
        end_points = model(cu_dt)
        loss = end_points['loss'].item()
        print('seg_loss:{:.4f} match_loss:{:.4f}'.format(end_points['seg_loss'].item(),end_points['match_loss'].item()))
        loss.backward()
        optimizer.step()
        
    
        # K = datum['K']
        # cam_scale = datum['cam_scale']
        # rgb = datum['rgb'].transpose(1, 2, 0)[...,::-1].copy()# [...,::-1].copy()
        # for i in range(22):
        #     pcld = datum['cld_rgb_nrm'][:3, :].transpose(1, 0).copy()
        #     p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
        #     rgb = bs_utils.draw_p2ds(rgb, p2ds)
        # cv2.imwrite('sss.png',rgb)
         
          
  

if __name__ == "__main__":
    main()
