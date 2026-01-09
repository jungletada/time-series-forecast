from data_provider.data_loader import \
    Dataset_ETT_hour, Dataset_ETT_minute, \
    Dataset_Custom, Dataset_M4, Dataset_PEMS, \
    PSMSegLoader, MSLSegLoader, \
    SMAPSegLoader, SMDSegLoader, \
    SWATSegLoader, UEAloader
from data_provider.data_dep_loader import \
    Dataset_ETT_Decomposed, Dataset_Custom_Decomposed, \
    Dataset_PEMS_Decomposed, Dataset_M4_Decomposed
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'ETTh1_dep': Dataset_ETT_Decomposed,
    'ETTh2_dep': Dataset_ETT_Decomposed,
    'ETTm1_dep': Dataset_ETT_Decomposed,
    'ETTm2_dep': Dataset_ETT_Decomposed,
    'Exchange_dep': Dataset_Custom_Decomposed,
    'Illness_dep': Dataset_Custom_Decomposed,
    'Weather_dep': Dataset_Custom_Decomposed,
    'Traffic_dep': Dataset_Custom_Decomposed,
    'Electricity_dep': Dataset_Custom_Decomposed,
    'PEMS_dep': Dataset_PEMS_Decomposed,
    'M4_dep': Dataset_M4_Decomposed,
}

def data_provider(args, flag):
    DataSet = data_dict[args.data_type]
    time_enc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq # freq for time features encoding

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = DataSet(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, f"length of data set: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    elif args.task_name == 'classification':
        drop_last = False
        data_set = DataSet(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    
    else:
        if args.data_type == 'm4':
            drop_last = False
        data_set = DataSet(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            time_enc=time_enc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, f"length of data set: {len(data_set)}")
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        return data_set, data_loader
