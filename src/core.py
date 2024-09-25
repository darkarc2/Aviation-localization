import cv2
import numpy as np
import os
import time
from Image_Viewer import ImageViewer



viewer = ImageViewer(uav)
    
    load_thread = threading.Thread(target=load_images, args=(image_dir, viewer, uav))
    load_thread.start()

    while True:
        # 显示图像
        viewer.run()
        cv2.imshow('Image Viewer', viewer.view)
        # 退出条件
        if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
            break

    cv2.destroyAllWindows()
    # 等待加载线程结束
    load_thread.join()
