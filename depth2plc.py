import numpy as np

# 加载深度数据
img = np.genfromtxt('/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/depth/depth_data/0000000362.csv', delimiter=',').astype(np.float32)

# 参数
CAM_WID, CAM_HGT = 1408, 376
CAM_FX, CAM_FY = 552.554261, 552.554261
CAM_CX, CAM_CY = 682.049453, 238.769549

# 转换
x, y = np.meshgrid(range(CAM_WID), range(CAM_HGT))
x = x.astype(np.float32) - CAM_CX
y = y.astype(np.float32) - CAM_CY

img_z = img.copy()
if False:  # 如果需要矫正视线到Z的转换的话使能
    f = (CAM_FX + CAM_FY) / 2.0
    img_z *= f / np.sqrt(x ** 2 + y ** 2 + f ** 2)

pc_x = img_z * x / CAM_FX  # X=Z*(u-cx)/fx
pc_y = img_z * y / CAM_FY  # Y=Z*(v-cy)/fy

pc = np.array([pc_x.ravel(), pc_y.ravel(), img_z.ravel()]).T

# 结果保存
np.savetxt('pc.csv', pc, fmt='%.18e', delimiter=',', newline='\n')

# 从CSV文件加载点云并显示
pc = np.genfromtxt('pc.csv', delimiter=',').astype(np.float32)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure(1).gca(projection='3d')
ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], 'b.', markersize=0.5)
plt.title('point cloud')
plt.show()
