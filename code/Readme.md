# 代码说明

+ 数据分析：data_analysis.ipynb

+ 获取test的json：get_testjson.py

+ Retinex与加速测试代码：Retinex.py

  已集成在mmdet的transform里，此处仅为测试用

+ WBF: Weighted-Boxes-Fusion/ensemble.ipynb

+ 泊松融合：poisson-image-editing和poisson_blending.ipynb

+ 画标注框到训练集图上：draw_bbox.ipynb

+ 画预测框到测试集图上：draw_pred_bbox.ipynb

+ 实例平衡增强：instance_balanced_augmentation.ipynb

+ 动态模糊可视化样本图代码：Motion_Blurring.ipynb

+ 动态模糊，Mixup和Retinex加入mmdetection后的代码：mmdet/datasets/pipelines/transforms.py，使用时请自行在__init__.py下加入它们的名称。

+ 配置文件按照需求修改以下内容：

  ```python
  train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mixup', prob=0.5, lambd=0.8, mixup=True,
         json_path='data/seacoco/train_waterweeds.json',
         img_path='data/seacoco/train/'),
    dict(type='MotionBlur', p=0.3),
    dict(type='Resize', img_scale=[(4096, 600), (4096, 1000)],
         multiscale_mode='range', keep_ratio=True),
    dict(type='Retinex', model='MSR', sigma=[30, 150, 300],
         restore_factor=2.0, color_gain=6.0, gain=128.0, offset=128.0),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Albu',
         transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_labels'],
                          min_visibility=0.0,
                          filter_lost_elements=True),
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         },
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
  ]
  ```

+ 标签平滑：mmdet/losses/cross_entropy_loss.py加入了标签平滑的方法。在配置文件需要修改（rpn_head下的CEloss的平滑指数为0.0，bbox_head下的CEloss的平滑指数>0.0即可）如下：

  ```python
  oss_cls=dict( type='CrossEntropyLoss',
               use_sigmoid=False,
               loss_weight=1.0，
               smoothing=0.001)
  ```

