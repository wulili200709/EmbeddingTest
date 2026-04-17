EmbeddingTest/test_nkio_board.py
它现在会做这些事：
自动找默认 nkio_config.ini
打开 NP-6133-16I16O 板卡
打印当前 16 路 DI 和 DO
默认测试：
DI0 作为脚踏输入观察点
DO0/DO1/DO2 依次点亮红绿蓝
DO3/DO4 依次测试相机1/相机2光源
支持监控脚踏上升沿变化
退出前会自动把本次测试过的 DO 清零
现在和它配套的底层封装文件是：
EmbeddingTest/devices/nkio_errors.py
EmbeddingTest/devices/nkio_raw.py
EmbeddingTest/devices/nkio_board.py
你可以先用默认点位跑，如果现场点位不一样，也可以改参数。
建议你现场先这样试：
py EmbeddingTest/test_nkio_board.py
如果只想先测灯，不想监控脚踏：
py EmbeddingTest/test_nkio_board.py --skip-watch-di
如果只想看脚踏，不想动输出：
py EmbeddingTest/test_nkio_board.py --skip-output-test
如果点位不一样，也可以这样：
py EmbeddingTest/test_nkio_board.py --foot-di 1 --red-do 4 --green-do 5 --blue-do 6 --light1-do 7 --light2-do 8
现在最建议的下一步不是继续扩代码，而是先实际跑这个脚本，确认三件事：
nkio_config.ini 能正常初始化
脚踏 DI 能读到变化
红绿蓝和两路光源的 DO 实际点位与极性是否正确


io_controller.py 就是：
把“板卡点位”翻译成“业务动作”的那一层。

