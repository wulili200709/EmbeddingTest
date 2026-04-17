开发清单
A. 检测项配置层
目标：把“ROI”升级成“可运行的检测项”。

新增 inspection_items.py
定义 InspectionItem 数据结构
定义检测项持久化文件，例如 inspection_items.json
支持字段：
item_id
display_name
camera_id
roi_label
algorithm_type
enabled
补加载/保存接口
约定第一版只允许 camera_id in {"cam1", "cam2"}
B. 模板与检测项联动
目标：line2dup 参考图画几个 ROI，就生成几个检测项。

从 line2dup recipe 或参考 ROI 中提取 ROI 标签
自动生成默认检测项
默认规则先定：
全部分配给 cam1
或按后续配置页人工修改
增加“重新同步检测项”能力
保证检测项删除/重命名时和 ROI 标签关系一致
C. 调试页面完善
目标：明确“调试页面负责手动测试”和“运行页面负责现场运行”。

把 ToolPage 明确作为调试页主体
在调试页里梳理/拆分能力区：
相机调试
模板与 ROI
算法测试
DI/DO 调试
保留并强化：
手动加载单张本地图测试
手动加载多张/目录批量测试
相机单帧采图测试
区分两类测试入口：
来自相机
来自本地文件
D. 运行页面升级
目标：运行页不再只是基础状态页，而是符合需求的产线运行页。

runtime_mode_pyside6.py 支持 0/1/2 相机布局
0 相机时：
显示未连接
禁止触发
1 相机时：
显示 1 个画面
2 相机时：
显示 2 个画面
新增显示区域：
当前产品
每个检测项状态
相机级结果
总结果
锁定状态
检测项状态灯支持：
未检测灰色
OK 绿色
NG 红色
E. 运行结果数据模型
目标：把零散状态整理成统一结果对象。

新增 inspection_models.py
定义：
InspectionTask
InspectionItemResult
CameraInspectionResult
FinalInspectionResult
字段建议包含：
task_id
trigger_timestamp
camera_images
item_results
camera_results
final_result
elapsed_ms
error_message
is_system_error
所有运行日志、CSV、界面显示统一基于这些对象
F. 结果汇总逻辑
目标：严格按文档实现结果聚合。

新增 result_aggregator.py
实现：
item 级结果
camera 级结果
final 级结果
汇总规则：
camera_result = AND(该相机下全部 item_result)
final_result = AND(全部 camera_result)
支持单相机模式：
只有 cam1 时，最终结果直接取 cam1_result
区分：
业务 NG
系统异常 NG
G. 运行执行器
目标：把“采图”和“检测”从 UI 完全下沉。

新增 inspection_executor.py
负责：
根据相机图片执行检测
返回 item/camera 级结果
第一版先支持：
cam1
cam2
后续为线程池并行执行做准备
H. 双相机采图时序落地
目标：严格符合文档中的“顺序采图、并行检测”。

cam1 开灯
等待稳定时间
cam1 触发拍照
cam1 关灯
cam1 图片提交算法线程
cam2 开灯
等待稳定时间
cam2 触发拍照
cam2 关灯
cam2 图片提交算法线程
等待全部算法结束
汇总结果
刷新运行页
输出三色灯与锁定状态
建议这部分逐步放进：

RuntimeController
或后续再拆 capture_scheduler.py
I. IO 配置层
目标：不要把点位和极性写死在业务代码里。

新增 io_config.py
新增配置文件，例如：
io_mapping.yaml
或 io_mapping.json
配置字段至少包含：
foot_switch
tower_red
tower_green
tower_blue
light_cam1
light_cam2
每路 active_high
默认配置按当前确认值：
三色灯：active_high = false
光源：保留待确认，可配置
J. IO 控制层
目标：真正让 DI/DO 和业务解耦。

新增 io_manager.py
新增 tower_light_controller.py
新增 light_controller.py
新增 di_monitor.py
IoManager 统一负责：
读 DI
写 DO
极性转换
三色灯控制全部走 TowerLightController
光源控制全部走 LightController
禁止 UI 直接操作板卡
K. DI 上升沿触发
目标：从手工按钮触发过渡到真实脚踏触发。

di_monitor.py 监听脚踏输入
只响应 0 -> 1 上升沿
加去抖时间
当前状态不允许时忽略触发
将有效触发转给 RuntimeController.trigger()
记录无效触发日志
L. 三色灯与锁定逻辑
目标：让运行状态、结果输出、密码放行闭环。

明确状态映射：
WaitingTrigger：蓝灯亮
Capturing/Inspecting/Aggregating：全灭
CompletedOk：绿灯亮
CompletedNg：红灯亮
三色灯按低电平有效输出
NG 后进入锁定
密码正确后只放行一次
“放行一次”必须在真正开始有效检测后才消耗
增加放行日志记录
M. 记录与报表
目标：让运行记录和调试测试结果都可追溯。

统一运行记录格式
记录：
产品名
相机结果
检测项结果
总结果
耗时
是否系统异常
CSV 每日滚动保存
调试页批量测试结果可导出
放行日志独立记录
N. 双窗口最终落地
目标：彻底完成运行界面与调试界面分离。

新增 RunMainWindow
新增 DebugMainWindow
调试页挂 ToolPage
运行页挂 RuntimeModePage
共用：
ProductSession
AlgorithmController
RuntimeController
检测项配置层
后续再考虑登录/权限入口
最推荐的实际开工顺序
如果你要我按“最稳、返工最少”的顺序排，我建议这样执行：

inspection_items.py
模板 ROI -> 检测项自动生成
运行页显示检测项状态 + 1/2 相机布局
inspection_models.py
result_aggregator.py
inspection_executor.py
双相机顺序采图 + 并行检测
io_config.py
io_manager.py / tower_light_controller.py / light_controller.py
di_monitor.py
锁定/放行日志
RunMainWindow / DebugMainWindow


EmbeddingTest/
├── ui/
│   ├── debug/
│   │   ├── tool_page_pyside6.py
│   │   ├── roi_canvas_pyside6.py
│   │   ├── line2dup_template_page_pyside6.py
│   │   └── embedding_analysis_dialog.py
│   ├── runtime/
│   │   ├── runtime_mode_pyside6.py
│   │   ├── run_main_window.py
│   │   └── widgets/
│   └── debug_main_window.py
│
├── application/
│   ├── runtime_controller.py
│   ├── algorithm_controller.py
│   ├── product_session.py
│   └── inspection_executor.py
│
├── domain/
│   ├── inspection_items.py
│   ├── inspection_models.py
│   ├── result_aggregator.py
│   └── recipe_manager.py
│
├── devices/
│   ├── io_manager.py
│   ├── di_monitor.py
│   ├── light_controller.py
│   └── tower_light_controller.py
│
├── services/
│   └── ...
│
├── config/
│   ├── io_config.py
│   └── defaults/
│
└── qr_gui_pyside6.py   # 过渡入口，后面再精简/替换


EmbeddingTest/
├── ui/
├── runtime/
├── config/
├── devices/
├── services/
├── models/
└── qr_gui_pyside6.py