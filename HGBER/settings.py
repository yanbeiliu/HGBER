def set(): 
    # parameter = {
    #     'GPU': 3,

    #     'data_dir': './data/',
    #     'data_set': 'ACM',
    #     'model': 'VGAE',
    #     'need_cluster': True,

    #     'prtraindgi_EPOCH': 90,
    #     'prtrainH_EPOCH': 10,
    #     'withkl_EPOCH':10,
    #     'inter_H': 10,
    #     'inter_H2': 1,
    #     'inter_cluster': 1,

    #     'lr_dgi': 0.001,
    #     'lr_dg': 0.001,
    #     'lr_H': 0.003,
    #     'lr_cluster': 0.001,

    #     'lamb_dg': 1,
    #     'lamb_cluster': 1,
    #     'view0':1,
    #     'view1':0.8,
        
    #     'H_dim': 64,
    #     'latent_dim': 20
    # }
    parameter = {
        'GPU': 3,
        'data_dir': './data/',
        'data_set': 'DBLP',
        'model': 'GAE',
        'need_cluster': True,

        'prtraindgi_EPOCH': 120,
        'prtrainH_EPOCH': 10,
        'withkl_EPOCH':10,
        'inter_H': 10,
        'inter_H2': 1,
        'inter_cluster': 1,

        'lr_dgi': 0.001,
        'lr_dg': 0.001,
        'lr_H': 0.003,
        'lr_cluster': 0.001,

        'lamb_dg': 1,
        'lamb_cluster': 1,
        'view0':0.7,
        'view1':1,
        'view2':0.7,
        
        'H_dim': 64,
        'latent_dim': 20
    }
    # parameter = {
    #     'GPU': 5,

    #     'data_dir': './data/',
    #     'data_set': 'IMDB',
    #     'model': 'VGAE',
    #     'need_cluster': True,

    #     'prtraindgi_EPOCH': 30,
    #     'prtrainH_EPOCH': 10,
    #     'withkl_EPOCH':70,
    #     'inter_H': 10,
    #     'inter_H2': 50,
    #     'inter_cluster': 1,

    #     'lr_dgi': 0.003,
    #     'lr_dg': 0.001,
    #     'lr_H': 0.003,
    #     'lr_cluster': 0.001,

    #     'lamb_dg': 1,
    #     'lamb_cluster': 1,
    #     'view0':0.8,
    #     'view1':1,
         
    #     'H_dim': 64,
    #     'latent_dim': 20
    # }
    # parameter = {
    #     'GPU': 1,

    #     'data_dir': './data/',
    #     'data_set': 'Yelp',
    #     'model': 'GAE',
    #     'need_cluster': True,

    #     'prtraindgi_EPOCH': 40,
    #     'prtrainH_EPOCH': 10,
    #     'withkl_EPOCH':70,
    #     'inter_H': 10,
    #     'inter_H2': 50,
    #     'inter_cluster': 1,

    #     'lr_dgi': 0.001,
    #     'lr_dg': 0.001,
    #     'lr_H': 0.003,
    #     'lr_cluster': 0.001,

    #     'lamb_dg': 1,
    #     'lamb_cluster': 1,
    #     'view0':0.7,
    #     'view1':0.8,
    #     'view2':1,
    #     'view3':0.8,
         
    #     'H_dim': 20,
    #     'latent_dim': 20
    # }
    if parameter['data_set'] == 'ACM':
        parameter['num_class'] = 3
        parameter['num_view'] = 2
    if parameter['data_set'] == 'DBLP':
        parameter['num_class'] = 4
        parameter['num_view'] = 3
    if parameter['data_set'] == 'IMDB':
        parameter['num_class'] = 3
        parameter['num_view'] = 2
    if parameter['data_set'] == 'Yelp':
        parameter['num_class'] = 3
        parameter['num_view'] = 4
    return parameter
    
    
    
    
    
    
