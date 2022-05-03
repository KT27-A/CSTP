import torch
from torch import nn
from .pace import r21d, r21d_byol, c3d
from .pace.c3d_byol import C3DBYOL
from .coclr import s3dg
from .BE.r3d_byol import R3DBYOL
from .BE.i3d_byol import I3DBYOL
import numpy as np


def neq_load_customized(model, pretrained_dict, verbose=True):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
        for k, v in pretrained_dict.items():
            if k in model_dict:
                tmp[k] = v
            else:
                # print(k)
                pass
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def generate_model(opts):
    # build model
    if opts.model_name == 'r21d':
        from .pace.r21d import get_fine_tuning_parameters
        model = r21d.R2Plus1DNet(num_classes=opts.n_classes)
    elif opts.model_name == 'r21d_byol':
        from .pace.r21d_byol import get_fine_tuning_parameters
        if opts.task in ["r_byol", "loss_com"]:
            model = r21d_byol.R21DBYOL(pretrain=True)
        elif opts.task in ["ft_fc", "ft_all", "scratch", "test"]:
            model = r21d_byol.R21DBYOL(pretrain=False, num_classes=opts.n_classes, cls_bn=True)
    elif opts.model_name == "s3d_classify":
        from .coclr.s3dg import get_fine_tuning_parameters
        assert opts.task in ["ft_fc", "ft_all", "scratch", "test"]
        model = s3dg.S3DClassify(pretrain=False, gating=False, slow=False, num_classes=opts.n_classes)
    elif opts.model_name == 's3d_byol':
        from .coclr.s3dg import get_fine_tuning_parameters
        if opts.task in ['r_byol', "loss_com"]:
            model = s3dg.S3DGBYOL(pretrain=True, gating=True, slow=False, num_classes=opts.n_classes)
        elif opts.task in ['ft_fc', 'ft_all', 'scratch', 'test', 'test_colorjit']:
            model = s3dg.S3DGBYOL(pretrain=False, gating=True, slow=False, num_classes=opts.n_classes)
    elif opts.model_name == "r3d_byol":
        from .BE.r3d_byol import get_fine_tuning_parameters
        if opts.task in ["r_byol", "loss_com"]:
            model = R3DBYOL(pretrain=True, opts=opts)
        elif opts.task in ["ft_fc", "ft_all", "scratch", "test"]:
            model = R3DBYOL(pretrain=False, cls_bn=True, opts=opts)
    elif opts.model_name == "i3d_byol":
        from .BE.i3d_byol import get_fine_tuning_parameters
        if opts.task in ["r_byol", "loss_com"]:
            model = I3DBYOL(pretrain=True, opts=opts)
        elif opts.task in ["ft_fc", "ft_all", "scratch", "test"]:
            model = I3DBYOL(pretrain=False, opts=opts)
    elif opts.model_name == "c3d_byol":
        from .pace.c3d_byol import get_fine_tuning_parameters
        if opts.task in ["r_byol", "loss_com"]:
            model = C3DBYOL(pretrain=True)
        elif opts.task in ["ft_fc", "ft_all", "scratch", "test"]:
            model = C3DBYOL(pretrain=False, cls_bn=True, num_classes=opts.n_classes)
    else:
        raise ValueError("Please check the input backbone!")

    # set model
    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        model.cuda(opts.local_rank)
        # TODO test whether to whole world
        if opts.sync_bn:
            if opts.task == "ft_fc":
                process_group = torch.distributed.new_group([opts.local_rank])
                sync_bn_model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
                model = torch.nn.parallel.DistributedDataParallel(sync_bn_model,
                                                                  device_ids=[opts.local_rank],
                                                                  output_device=opts.local_rank,
                                                                  find_unused_parameters=True)
            else:
                process_group = torch.distributed.new_group([opts.local_rank])
                sync_bn_model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
                model = torch.nn.parallel.DistributedDataParallel(sync_bn_model,
                                                                  device_ids=[opts.local_rank],
                                                                  output_device=opts.local_rank,
                                                                  find_unused_parameters=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[opts.local_rank], output_device=opts.local_rank, find_unused_parameters=False)
    else:
        model = model.to(opts.device)
        model = nn.DataParallel(model)

    if opts.task in ["scratch", "r_ctr", 'r_cls', 'r_byol', "loss_com"]:
        return model, model.parameters()
    elif "test" in opts.task:
        print('Test model {}!'.format(opts.test_md_path))
        test_md = torch.load(opts.test_md_path, map_location=opts.device)
        assert opts.arch == test_md['arch']
        model.load_state_dict(test_md['state_dict'])
        return model
    elif opts.task == 'resume':
        print('Resume model {}!'.format(opts.resume_md_path))
        resume_md = torch.load(opts.resume_md_path)
        assert opts.arch == resume_md['arch']
        model.load_state_dict(resume_md['state_dict'])
        return model, model.parameters()
    elif opts.task in ['ft_fc', 'ft_all']:
        if opts.task == 'ft_fc':
            print('Fine-tune FC layer!')
            opts.ft_begin_index = 5
        elif opts.task == 'ft_all':
            print('Fine-tune all layers')
            opts.ft_begin_index = 0

        # checkpoint = torch.load(opts.pretrained_path, map_location=torch.device('cpu'))
        checkpoint = torch.load(opts.pretrained_path, map_location=torch.device('cpu'))
        assert (opts.arch in checkpoint['arch'] or checkpoint['arch'] in opts.arch)
        print('adjust input weights according to new network')
        model = neq_load_customized(model, checkpoint['state_dict'], verbose=True)
        # model.module.classify = nn.Linear(model.module.classify.in_features, opts.n_finetune_classes)
        # # initialize classifier
        # model.module.classify.weight = model.module._glorot_uniform(model.module.classify.weight)
        # if opts.cuda:
        #     model.module.classify = model.module.classify.cuda()

        print("loaded pretrained checkpoint '{}' (epoch {})".format(opts.pretrained_path, checkpoint['epoch']))
        parameters = get_fine_tuning_parameters(model, opts.ft_begin_index)

    return model, parameters
