# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):
        self.init_path()
        # self.check_path()


    def init_path(self):

        self.DATA_ROOT = './data'
        self.VIT_DATA_ROOT = '../../../shenxiang/sda/ccq/vit_feat'
        # self.DATA_ROOT = '/data/datasets'
        # self.DATA_ROOT = '/data1/datasets'
        # self.DATA_ROOT = '/home/features'
        self.BERT_PATH = './data/vqa/bert_tensor'

        self.DATA_PATH = {
            'vqa': self.DATA_ROOT + '/vqa',
            'gqa': self.DATA_ROOT + '/gqa',
            'clevr': self.DATA_ROOT + '/clevr',
            'vqa_grid': self.DATA_ROOT + '/vqa_grid',
            'vqa_vit': self.VIT_DATA_ROOT + '/vqa_vit'
        }


        self.FEATS_PATH = {

            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/feats' + '/train2014',
                'val': self.DATA_PATH['vqa'] + '/feats' + '/val2014',
                'test': self.DATA_PATH['vqa'] + '/feats' + '/test2015',
            },
            
            'vqa_vit': {
                'train': self.DATA_PATH['vqa_vit'] + '/feats' + '/train2014',
                'val': self.DATA_PATH['vqa_vit'] + '/feats' + '/val2014',
                'test': self.DATA_PATH['vqa_vit'] + '/feats' + '/test2015',
            },            
            
            'gqa': {
                'default-frcn': self.DATA_PATH['gqa'] + '/feats' + '/gqa-frcn',
                'default-grid': self.DATA_PATH['gqa'] + '/feats' + '/gqa-grid',
            },
            'clevr': {
                'train': self.DATA_PATH['clevr'] + '/feats' + '/train',
                'val': self.DATA_PATH['clevr'] + '/feats' + '/val',
                'test': self.DATA_PATH['clevr'] + '/feats' + '/test',
            },
            'vqa_grid': {
                'train': self.DATA_PATH['vqa_grid'] + '/feats' + '/train2014',
                'val': self.DATA_PATH['vqa_grid'] + '/feats' + '/val2014',
                'test': self.DATA_PATH['vqa_grid'] + '/feats' + '/test2015',
            },
        }
        
        
        self.SPATIALS_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/graph' + '/train2014',
                'val': self.DATA_PATH['vqa'] + '/graph' + '/val2014',
                'test': self.DATA_PATH['vqa'] + '/graph' + '/test2015',
            }
        }



        self.RAW_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_train2014_questions.json',
                'train-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_train2014_annotations.json',
                'val': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_val2014_questions.json',
                'val-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_val2014_annotations.json',
                'vg': self.DATA_PATH['vqa'] + '/raw' + '/VG_questions.json',
                'vg-anno': self.DATA_PATH['vqa'] + '/raw' + '/VG_annotations.json',
                'test': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_test2015_questions.json',
            },
            'gqa': {
                'train': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/train_balanced_questions.json',
                'val': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/val_balanced_questions.json',
                'testdev': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/testdev_balanced_questions.json',
                'test': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/submission_all_questions.json',
                'val_all': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/val_all_questions.json',
                'testdev_all': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/testdev_all_questions.json',
                'train_choices': self.DATA_PATH['gqa'] + '/raw' + '/eval/train_choices',
                'val_choices': self.DATA_PATH['gqa'] + '/raw' + '/eval/val_choices.json',
            },
            'clevr': {
                'train': self.DATA_PATH['clevr'] + '/raw' + '/questions/CLEVR_train_questions.json',
                'val': self.DATA_PATH['clevr'] + '/raw' + '/questions/CLEVR_val_questions.json',
                'test': self.DATA_PATH['clevr'] + '/raw' + '/questions/CLEVR_test_questions.json',
            },
            'vqa_grid': {
                'train': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_train2014_questions.json',
                'train-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_train2014_annotations.json',
                'val': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_val2014_questions.json',
                'val-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_val2014_annotations.json',
                'vg': self.DATA_PATH['vqa'] + '/raw' + '/VG_questions.json',
                'vg-anno': self.DATA_PATH['vqa'] + '/raw' + '/VG_annotations.json',
                'test': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_test2015_questions.json',
            }
        }


        self.SPLITS = {
            'vqa': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'gqa': {
                'train': '',
                'val': 'testdev',
                'test': 'test',
            },
            'clevr': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'vqa_grid': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            
            'vqa_vit': {
                'train': '',
                'val': 'val',
                'test': 'test',
            }

        }


        self.RESULT_PATH = './results/result_test'
        self.PRED_PATH = './results/pred'
        self.CACHE_PATH = './results/cache'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self, dataset=None):
        print('Checking dataset ........')


        if dataset:
            for item in self.FEATS_PATH[dataset]:
                if not os.path.exists(self.FEATS_PATH[dataset][item]):
                    print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)
                    
            for item in self.SPATIALS_PATH['vqa']:
                if not os.path.exists(self.SPATIALS_PATH['vqa'][item]):
                    print(self.SPATIALS_PATH['vqa'][item], 'NOT EXIST')
                    exit(-1)

            for item in self.RAW_PATH[dataset]:
                if not os.path.exists(self.RAW_PATH[dataset][item]):
                    print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

        else:
            for dataset in self.FEATS_PATH:
                for item in self.FEATS_PATH[dataset]:
                    if not os.path.exists(self.FEATS_PATH[dataset][item]):
                        print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)
                        
            for dataset in self.SPATIALS_PATH:
                for item in self.SPATIALS_PATH[dataset]:
                    if not os.path.exists(self.SPATIALS_PATH[dataset][item]):
                        print(self.SPATIALS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

            for dataset in self.RAW_PATH:
                for item in self.RAW_PATH[dataset]:
                    if not os.path.exists(self.RAW_PATH[dataset][item]):
                        print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

        print('Finished!')
        print('')

