dataset_info = dict(
    dataset_name='signcorner',
    paper_info=dict(
        author='Navinfo',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        year='2019',
    ),
    keypoint_info={
        0:
        dict(name='L_Up', id=0, color=[0, 255, 0], type='upper', swap='R_Up'),
        1:
        dict(name='R_Up', id=1, color=[255, 128, 0], type='upper', swap='L_Up'),
        2:
        dict(name='L_Down', id=2, color=[0, 255, 0], type='upper', swap='R_Down'),
        3:
        dict(name='R_Down', id=3, color=[255, 128, 0], type='upper', swap='L_Down')
    },
    skeleton_info={
        0: dict(link=('L_Up', 'R_Up'), id=0, color=[51, 153, 255]),
        1: dict(link=('R_Up', 'R_Down'), id=1, color=[0, 255, 0]),
        2: dict(link=('R_Down', 'L_Down'), id=2, color=[255, 128, 0]),
        3: dict(link=('L_Down', 'L_Up'), id=3, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1., 1., 1.
    ],

    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.025, 0.025, 0.026, 0.035
    ])
