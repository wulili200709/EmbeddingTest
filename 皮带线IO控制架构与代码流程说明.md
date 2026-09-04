# 皮带线 IO 控制架构与代码流程说明

> 文档用途：说明当前代码中 DI 采集、状态机、FIFO、视觉检测、NG 吹气、自动清线、堵料监控和 DO 输出之间的完整调用关系。  
> 适用代码：`EmbeddingTest-conveyor` 当前版本。  
> 配套设计文档：`皮带线IO信号与控制逻辑.md`。  
> 更新日期：2026-08-29。

## 1. 核心结论

当前 IO 控制已经从“一个文件同时负责采集、判断和输出”调整为分层结构：

```text
IO板卡只负责物理读写
DI轮询器负责整字采集、去抖和边沿事件
产线状态机负责决定允许什么动作
独立领域组件负责FIFO、出口确认、吹气、清线和堵料计时
输出仲裁器负责执行最终DO请求
```

主要原则：

1. 状态机使用业务名称，不直接使用板卡通道号。
2. 所有 DI 在同一扫描周期从一个完整 DI 字解析，避免输入时间不一致。
3. DI5、DI9、DI10和 DI 读取失败属于高优先级事件，优先关闭 DO4、DO3。
4. 连续物料使用逐件 FIFO，不使用全局 `last_result`。
5. 算法通过 `sequence_id + epoch` 回写结果，算法乱序完成不会改变物料顺序。
6. 自动清线时 DO3 持续吹气，预吹气结束后 DO4 必须运行，否则无法把物料清出设备。
7. DI8 当前有料不阻止启动清线；持续有效超过堵料阈值才构成堵料。
8. 安全恢复、门恢复、报警确认后均不得自动启动皮带。
9. DI1只完成第一段跟踪：GOOD转入等待DI7，NG转入等待DI8；正确出口有效沿到达后产品才真正完成。
10. DI7、DI8同时承担“逐件出口确认”和“持续有效堵料检测”，两种判断使用不同计时起点和参数。

## 2. 整体软件架构

```mermaid
flowchart TD
    A["NK IO板卡<br/>DI0～DI10 / DO0～DO9"] --> B["IoController<br/>物理IO读写与有效电平转换"]
    B --> C["DiPoller<br/>整字读取、去抖、边沿检测"]
    C --> D{"输入类型"}

    D -->|"DI5、DI9、DI10<br/>安全输入"| E["直接调用线程安全控制入口<br/>优先处理"]
    D -->|"其他DI事件"| F["Qt事件队列<br/>顺序处理"]

    E --> G["ConveyorLineController<br/>状态机和最终动作决策"]
    F --> G

    G --> H["WorkpieceTracker<br/>FIFO和epoch"]
    G --> O["OutletConfirmationTracker<br/>DI7/DI8逐件出口确认"]
    G --> I["RejectBlowController<br/>NG吹气窗口"]
    G --> J["AutoPurgeController<br/>一键清线"]
    G --> K["JamMonitor<br/>DI6/DI7/DI8堵料"]

    H --> G
    O --> G
    I --> G
    J --> G
    K --> G

    G --> L["OutputArbiter<br/>输出优先级与蜂鸣器仲裁"]
    L --> B
    B --> A

    M["相机采图任务"] --> N["后台检测任务"]
    G -->|"sequence_id + epoch"| M
    N -->|"按编号回写GOOD/NG"| G
```

## 3. 代码文件职责

| 层级 | 文件 | 主要职责 |
|---|---|---|
| IO映射 | `config/defaults/io_mapping.json` | DI/DO业务名称、板卡通道和有效电平 |
| 控制参数 | `config/defaults/conveyor_control.json` | 去抖、吹气、FIFO、出口确认、清线和堵料时间参数；`_comments`提供中文备注 |
| 板卡业务接口 | `devices/io_controller.py` | 读取完整DI字、写DO、物理电平与业务状态转换 |
| DI扫描 | `devices/di_poller.py` | 定时扫描、去抖、生成上升沿/下降沿事件 |
| 输出仲裁 | `devices/output_arbiter.py` | 执行产线输出、批量关闭DO3/DO4、仲裁共享蜂鸣器 |
| 产线状态机 | `domain/conveyor_line.py` | 安全联锁、状态转换、动作决策、故障恢复分类 |
| 领域组件 | `domain/conveyor_components.py` | FIFO、DI7/DI8出口等待、NG吹气窗口、清线上下文、堵料运动计时 |
| 产线运行接入 | `application/runtime/conveyor.py` | 连接状态机、相机采图、检测任务和运行界面 |
| 硬件运行接入 | `application/runtime/hardware.py` | 打开IO、建立安全初始输出、启动DI轮询 |
| 操作界面 | `ui/runtime/runtime_mode_pyside6.py` | 显示产线/FIFO/故障状态并提供清线、继续和确认操作 |

## 4. IO 业务信号

### 4.1 主要 DI

| DI | 业务名称 | 有效方式 | 状态机用途 |
|---|---|---|---|
| DI0 | `camera_trigger_sensor` | 低有效 | 创建物料、加入FIFO并触发采图 |
| DI1 | `reject_position_sensor` | 低有效 | 消费FIFO队头，决定GOOD通过或NG吹气 |
| DI2 | `start_button` | 高有效 | 请求启动正常生产 |
| DI3 | `stop_button` | 低有效 | 请求受控停止；清线中则暂停清线 |
| DI5 | `safety_ok` | 高有效 | 安全继电器运行许可 |
| DI6 | `end_test_sensor` | 高有效 | 专用堵料监控，可配置关闭 |
| DI7 | `good_outlet_sensor` | 高有效 | GOOD逐件出料确认，同时监控GOOD出口持续堵料 |
| DI8 | `waste_outlet_sensor` | 高有效 | NG逐件剔除确认，同时监控废料出口持续堵料和清线物料活动 |
| DI9 | `door_closed` | 高有效 | 下部门关闭许可 |
| DI10 | `door_upper_closed` | 高有效 | 上部门关闭许可，可配置启用 |

### 4.2 主要 DO

| DO | 业务名称 | 用途 |
|---|---|---|
| DO0 | `tower_red` | 检测NG红灯 |
| DO1 | `tower_green` | 检测GOOD绿灯 |
| DO2 | `tower_blue` | 检测待机蓝灯 |
| DO3 | `waste_removal` | 吹气电磁阀，业务ON时持续吹气 |
| DO4 | `conveyor_run` | 皮带运行使能 |
| DO5 | `button_green` | 按钮盒绿色运行灯 |
| DO7 | `button_blue` | 安全复位提示灯 |
| DO8 | `buzzer` | 检测短鸣和产线故障蜂鸣器 |
| DO9 | `button_red` | 按钮盒红色停止灯 |

所有现用 DO 均沿用现场验证的 `active_high:false`。状态机只使用：

```text
业务True  = 要求设备动作
业务False = 要求设备停止
```

实际板卡高低电平由 `IoController` 根据 `active_high` 转换。

## 5. 程序启动与 IO 初始化

```mermaid
flowchart TD
    A["RuntimeController启动"] --> B["读取io_mapping.json和板卡配置"]
    B --> C["创建IoController并打开板卡"]
    C --> D["创建OutputArbiter"]
    D --> E["写入安全初始状态<br/>DO4 OFF、DO3 OFF"]
    E --> F["创建DiPoller"]
    F --> G["读取一次完整DI初始快照"]
    G --> H["创建ConveyorLineController"]
    H --> I["initialize_inputs"]
    I --> J{"安全和门许可"}
    J -->|"满足"| K["READY_STOPPED"]
    J -->|"DI5无效"| L["SAFETY_LOCKED"]
    J -->|"门打开"| M["DOOR_OPEN_STOPPED"]
```

IO板卡打开后，必须在相机连接和模型预热之前先建立安全输出：

```python
safe_outputs = {
    "conveyor_run": False,
    "waste_removal": False,
    "button_green": False,
    "button_red": True,
    "button_blue": True,
    "buzzer": False,
}
```

## 6. DI 采集和事件流程

### 6.1 一个周期只读取一次完整 DI 字

`IoController.snapshot_inputs()` 的核心逻辑：

```python
raw_word = board.read_di_word()

for name in input_names:
    raw_level = get_bit(raw_word, mapping[name].channel)
    business_state = level_to_business(raw_level, mapping[name].active_high)
```

这样 DI0～DI10 来自同一个板卡快照，不会出现“先读DI0，几毫秒后再读DI5”而拼成不一致状态。

### 6.2 去抖

每个输入保存三组状态：

```text
stable_state    已确认的稳定状态
candidate_state 当前候选状态
candidate_since 候选状态开始时间
```

候选状态持续达到 `debounce_ms` 后才变成稳定状态并产生边沿事件。输入持续有效不会重复触发，必须先恢复无效后才能重新布防。

### 6.3 安全事件优先级

DI5、DI9、DI10事件不在界面事件队列中等待，而是从 DI 轮询线程直接调用线程安全的状态机入口：

```python
if name in {"safety_ok", "door_closed", "door_upper_closed"}:
    conveyor.handle_input_change(name, state)
else:
    emit_qt_di_event(name, state)
```

状态机内部使用可重入锁串行化所有公开操作。

安全事件到达时的顺序是：

```text
累计此前已经发生的皮带运动时间
→ 写入新的安全输入状态
→ 立即处理安全联锁
→ 批量关闭DO4、DO3
→ 不执行旧安全状态下到期的普通吹气或清线动作
```

## 7. 正常生产流程

```mermaid
flowchart TD
    A["READY_STOPPED"] -->|"DI2有效沿"| B["检查IO、安全、门、故障和相机"]
    B -->|"允许"| C["RUNNING<br/>DO4 ON"]
    B -->|"拒绝"| A

    C -->|"DI0有效沿"| D["创建sequence_id和epoch"]
    D --> E["加入FIFO队尾"]
    E --> F["提交相机采图"]
    F --> G["后台算法检测"]
    G -->|"按编号回写"| H["PENDING变为GOOD或NG"]

    C -->|"DI1有效沿"| I["读取FIFO队头"]
    I --> J{"队头状态"}
    J -->|"FIFO为空"| K["FIFO_UNDERFLOW"]
    J -->|"PENDING"| L["立即关闭DO4，进入WAITING_INSPECTION"]
    L -->|"结果在等待时间内完成"| J
    L -->|"超过inspection_result_wait_timeout_ms"| Z["RESULT_NOT_READY"]
    J -->|"GOOD且DO3关闭"| M["移除队头，建立等待DI7记录"]
    J -->|"GOOD且DO3开启"| N["BLOW_WINDOW_CONFLICT"]
    J -->|"NG"| O["移除队头，提交吹气并建立DI7防错记录"]
    O --> P["等待reject_blow_delay_ms"]
    P --> Q["DO3 ON"]
    Q --> R["保持reject_blow_duration_ms"]
    R --> S["DO3 OFF"]
    M --> T{"规定运行时间内收到DI7？"}
    S --> U{"防错窗口内是否收到DI7？"}
    T -->|"是"| V["GOOD完整离开设备"]
    U -->|"否"| W["NG剔除成功"]
    T -->|"否"| X["GOOD_OUTLET_TIMEOUT"]
    U -->|"是"| Y["REJECT_FAILED_WRONG_OUTLET"]
```

### 7.1 启动条件

```text
run_permitted =
    io_ready
    AND safety_ok
    AND door_lower_closed
    AND door_upper_closed（启用时）
    AND no_latched_fault
```

状态机允许从以下状态启动正常生产：

```text
READY_STOPPED
READY_TO_RESUME
```

恢复许可不等于启动。安全或门恢复后仍需要再次按 DI2。

## 8. FIFO 与异步检测

### 8.1 FIFO 的数据结构

`WorkpieceTracker` 同时维护：

```text
fifo               按物理先后顺序保存物料
records_by_id      按sequence_id快速查找物料
sequence           持续递增的物料流水号
epoch              当前生产/清线批次
```

每件物料保存：

```text
sequence_id
epoch
created_at
created_motion_s
inspection_status
inspection_detail
result_at
```

状态值包括：

```text
PENDING / GOOD / NG / ERROR / PURGED
```

当前检测规则中，模板失败、采图异常和其他非 OK 最终结果统一按 NG 处理。

### 8.2 DI0 创建物料的等价代码

```python
if state not in {RUNNING, CONTROLLED_STOPPING}:
    return

if len(fifo) >= max_inflight_items:
    trip_fault("FIFO_OVERFLOW")
    return

record = tracker.create(
    now=now,
    motion_s=motion_elapsed_s,
)

active_captures.add((record.sequence_id, record.epoch))
request_inspection(record.sequence_id, record.epoch)
```

### 8.3 算法乱序返回

假设 FIFO 顺序：

```text
队头 → #1001 → #1002 → #1003 → 队尾
```

算法可能按以下顺序返回：

```text
#1002 NG
#1003 GOOD
#1001 GOOD
```

程序根据编号分别更新记录，FIFO 顺序仍然是：

```text
#1001 GOOD → #1002 NG → #1003 GOOD
```

DI1第一次只能处理 #1001，不能处理最先完成算法的 #1002。

### 8.4 epoch 防止旧结果控制输出

一键清线会递增 epoch。例如：

```text
清线前：epoch=3，FIFO=[#21, #22]
清线后：epoch=4，FIFO=[]
```

清线后如果旧算法返回：

```text
sequence_id=22, epoch=3, result=NG
```

状态机发现结果 epoch 与当前 epoch 不一致，直接忽略，旧任务不能重新启动 DO3。

### 8.5 FIFO 的物理前提

FIFO只知道顺序，不能测量每件产品的精确位置，必须保证：

1. 产品在 DI0 到 DI1 之间不能超车。
2. 每件产品必须各触发一次 DI0 和 DI1。
3. 产品间隙必须足以让传感器恢复无效。
4. 正常生产期间不能人工插入、取走或移动内部产品。
5. 程序重启后若设备内仍有物料，必须清线后再生产。

### 8.6 两段在途跟踪

“在途产品”不是只有 `WorkpieceTracker.fifo`。当前代码分成两段：

```text
第一段：DI0 → DI1
  WorkpieceTracker.fifo
  保存检测结果并按物理顺序等待DI1

第二段：DI1 → DI7/DI8
  OutletConfirmationTracker.pending
  GOOD等待DI7，NG等待DI8
```

`max_inflight_items`约束两段数量之和。界面分别显示“DI1前、等待DI7、等待DI8”，收到正确出口有效沿后才从总在途中移除。

## 9. DI1 与 NG 吹气

DI1只读取 FIFO 队头，等价逻辑如下：

```python
record = tracker.head()

if record is None:
    trip_fault("FIFO_UNDERFLOW", recovery="PURGE_REQUIRED")

elif record.inspection_status == PENDING:
    conveyor_run = False
    state = WAITING_INSPECTION
    # 结果完成后自动按GOOD/NG分流并恢复皮带；只有等待超时才报警
    if wait_ms >= inspection_result_wait_timeout_ms:
        trip_fault("RESULT_NOT_READY", recovery="PURGE_REQUIRED")

elif record.inspection_status == GOOD:
    if waste_removal_output_is_on:
        trip_fault("BLOW_WINDOW_CONFLICT", recovery="PURGE_REQUIRED")
    else:
        tracker.pop_head()
        outlet.expect(record, expected_input="good_outlet_sensor", ...)

elif record.inspection_status == NG:
    tracker.pop_head()
    reject.schedule(
        sequence_id=record.sequence_id,
        motion_s=motion_elapsed_s,
        delay_s=reject_blow_delay_ms / 1000,
        duration_s=reject_blow_duration_ms / 1000,
    )
    outlet.expect(record, expected_input="waste_outlet_sensor", ...)
```

NG从第一段FIFO移除后，已经提交的吹气动作由 `RejectBlowController` 保存，逐件剔除确认由 `OutletConfirmationTracker` 保存。连续NG的吹气窗口可以重叠，DO3在任一窗口有效时保持ON。

如果 GOOD 到达 DI1 时 DO3 仍为ON，程序不继续消费这件 GOOD，而是锁存 `BLOW_WINDOW_CONFLICT`，关闭DO4、DO3并要求清线。

DI7、DI8有效沿按各自合法到达窗口匹配：GOOD收到DI7、NG收到DI8才算正确完成；超过最大运行时间未收到、走错出口或没有待匹配产品却出现出口信号，都会停机并要求清线。

## 10. 周期 tick

除 DI 边沿之外，状态机通过周期 `tick()` 推动计时动作：

```mermaid
flowchart LR
    A["周期tick"] --> B["累计皮带实际运行时间"]
    B --> C["更新NG吹气窗口"]
    C --> D["检查DI6/DI7/DI8堵料"]
    D --> E["检查DI7/DI8出口确认超时"]
    E --> F["检查DI0贴料/间距和FIFO队头到达超时"]
    F --> G["处理受控停止"]
    G --> H["处理自动清线"]
    H --> I["刷新按钮灯和蜂鸣器"]
```

等价代码：

```python
advance_motion_clock()
update_reject_output()
monitor_jams()
monitor_inspection_wait()
monitor_outlet_timeouts()
monitor_front_sensor_spacing()
monitor_fifo_timeout()

if state == CONTROLLED_STOPPING:
    evaluate_controlled_stop()
elif state in {PURGE_PREPARING, PURGE_RUNNING}:
    update_purge()

apply_indicator_outputs()
```

只有 `conveyor_run=True` 时才累计 `motion_elapsed_s`。急停、开门和停止期间不累计：

- DI0到DI1物料到达超时；
- DI1到吹气口延时；
- 正常NG吹气持续时间；
- DI1到DI7/DI8出口确认时间；
- DI6、DI7、DI8堵料时间。

## 11. DI3 受控停止

```mermaid
flowchart TD
    A["RUNNING"] -->|"DI3有效沿"| B["CONTROLLED_STOPPING"]
    B --> C["要求停止继续上料<br/>仍跟踪停止前已进入并触发DI0的物料"]
    C --> D{"是否有已提交采图<br/>或吹气窗口？"}
    D -->|"是"| E["等待有限动作完成"]
    E --> D
    D -->|"否"| F["关闭DO4、DO3"]
    F --> G["READY_STOPPED<br/>两段在途队列保留"]
    B -->|"超过controlled_stop_timeout_ms"| H["CONTROLLED_STOP_TIMEOUT"]
```

DI3与急停不同：DI3允许已经提交的采图和正常吹气窗口完成；DI5失效或开门则立即关闭DO4、DO3，不等待普通动作。

## 12. 安全联锁

```mermaid
flowchart TD
    A["正常生产或清线"] --> B{"DI5失效<br/>DI9/DI10开门<br/>DI读取失败"}
    B --> C["高优先级安全处理"]
    C --> D["一次批量输出<br/>DO4 OFF、DO3 OFF"]
    D --> E["停止累计运动时间"]
    E --> F{"中断前的流程"}

    F -->|"普通生产"| G["SAFETY_PAUSED或DOOR_PAUSED"]
    F -->|"NG正在吹气"| H["锁存BLOW_INTERRUPTED"]
    F -->|"自动清线"| I["PURGE_PAUSED"]
    F -->|"IO失败"| J["IO_NOT_READY"]

    G --> K["安全恢复后READY_TO_RESUME"]
    H --> L["安全恢复后FAULT_STOPPED<br/>只能清线"]
    I --> M["安全恢复后等待人工继续清线"]
    J --> N["必须重连IO"]
```

NG吹气中断后，物料是否已经成功剔除无法可靠判断，因此处理策略固定为：

```text
BLOW_INTERRUPTED
→ PURGE_REQUIRED
→ 不允许普通报警复位
→ 不允许直接恢复生产
→ 必须一键清线
```

安全恢复只代表重新具备动作许可，不代表自动恢复动作。

### 12.1 急停时第二段出口队列是否还能对应

可以保留对应关系，但前提是现场已确认的条件成立：急停后皮带立即停止，期间无人移动、取走或插入产品，程序进程也没有重启。实现方式如下：

```text
急停前：#101 GOOD等待DI7，#102 NG等待DI8
急停发生：DO4/DO3立即关闭，OutletConfirmationTracker.pending原样保留
急停期间：motion_elapsed_s停止累计，两个产品的出口到达窗口同时冻结
安全复位：只进入READY_TO_RESUME，不自动开皮带
人工重新启动：运动时间继续累计，后续DI7/DI8仍按原sequence_id和顺序确认
```

有两个例外不能继续原队列：

1. 急停打断了正在执行的NG吹气，程序锁存 `BLOW_INTERRUPTED` 并要求一键清线，因为该NG是否被剔除已经不可确定；
2. 急停期间有人移动物料、程序重启或传感器状态无法确认，软件没有位置编码器可重建一一对应，必须清线后重新生产。

> DI5急停由硬件安全回路切断执行机构电源。DI9、DI10目前属于软件门联锁，不能替代安全门接入安全继电器的硬件安全设计。

## 13. 一键自动清线

### 13.1 启动条件

```text
IO通信正常
AND DI5有效
AND DI9下部门关闭
AND DI10上部门关闭（启用时）
AND 皮带当前停止
AND 状态允许清线
```

允许清线的状态：

```text
READY_STOPPED
READY_TO_RESUME
FAULT_STOPPED且fault_recovery=PURGE_REQUIRED
```

DI8当前有料不否决清线。若已经锁存 `JAM_DETECTED`，则必须先清除对应传感器并确认堵料报警。

### 13.2 执行流程

```mermaid
flowchart TD
    A["点击一键清线"] --> B["检查IO、安全、门和停止状态"]
    B -->|"不满足"| C["拒绝清线"]
    B -->|"满足"| D["epoch递增"]
    D --> E["epoch递增，第一段FIFO标记PURGED并清空"]
    E --> F["清除第二段出口等待、旧采图提交和NG吹气窗口"]
    F --> G["进入PURGE_PREPARING"]
    G --> H["DO4 OFF、DO3 ON"]
    H --> I["等待purge_air_lead_ms"]
    I --> J["进入PURGE_RUNNING<br/>DO4 ON、DO3持续ON"]
    J --> K["皮带把全部物料带过吹气位置"]
    K --> L{"达到清线完成条件？"}
    L -->|"否"| J
    L -->|"是"| M["先关闭DO4"]
    M --> N["再关闭DO3"]
    N --> O["READY_STOPPED"]
```

清线运行过程中 DO4 必须开启。只开 DO3、不运行皮带，设备内部产品无法移动到吹气位置，不能完成清线。

### 13.3 完成条件

需要同时满足：

```text
清线运行达到purge_min_run_s
AND DI0、DI1、DI6、DI7、DI8当前全部无料
AND 最后一次物料传感器活动后达到purge_tail_run_s
AND DI8连续无料达到purge_quiet_s
```

超过 `purge_max_run_s` 仍不能完成，则关闭 DO4、DO3并报告 `PURGE_TIMEOUT`。

### 13.4 中断和继续

清线过程中急停、开门或操作员停止：

```text
DO4 OFF
DO3 OFF
PURGE_PAUSED
```

安全恢复后不自动继续。操作员点击“继续清线”后重新执行：

```text
DO3 ON
→ 等待purge_air_lead_ms
→ DO4 ON
→ 继续清线
```

## 14. DI6堵料与DI7、DI8出口确认/堵料复用

现场当前默认关闭DI8的NG逐件确认（`waste_outlet_confirmation_enabled=false`）：
NG吹气后在配置时间窗口内监控DI7，DI7按顺序匹配到NG时报告剔除失败；窗口
结束且DI7无信号则完成NG跟踪。DI6只保留专用堵料检测。此开关不关闭DI8堵料
检测，也不改变一键清线对DI8废料活动和连续无料时间的使用。

堵料按皮带实际运行时间判断：

```text
传感器业务状态有效
AND DO4业务状态为ON
AND 连续运动时间超过配置阈值
→ JAM_DETECTED
```

| 输入 | 阈值 |
|---|---|
| DI6 `end_test_sensor` | `end_test_blocked_timeout_s` |
| DI7 `good_outlet_sensor` | `good_outlet_blocked_timeout_s` |
| DI8 `waste_outlet_sensor` | `waste_outlet_blocked_timeout_s` |

皮带启动时，如果堵料输入已经有效，计时从皮带启动的运动时刻开始。皮带停止时清除本次运动计时。

DI7、DI8还有一套独立的逐件确认计时，不能与堵料阈值混用：

```text
GOOD离开DI1 → 在good_outlet_arrival_min/max_run_ms窗口等待DI7上升沿
NG离开DI1   → 在waste_outlet_arrival_min/max_run_ms窗口等待DI8上升沿

未收到正确有效沿 → GOOD_OUTLET_TIMEOUT / WASTE_OUTLET_TIMEOUT
NG却到DI7       → REJECT_FAILED_WRONG_OUTLET
GOOD却到DI8     → GOOD_WRONG_OUTLET
无待确认产品     → UNEXPECTED_GOOD_OUTLET / UNEXPECTED_WASTE_OUTLET
```

因此，DI7/DI8短脉冲用于“确认一件产品”，持续高电平超过堵料阈值用于“出口堵料”。两套计时都只累计皮带实际运行时间。

DI8在清线中的判断：

```text
清线前DI8当前有料       → 允许启动清线
清线中DI8短暂有效       → 正常废料通过
清线中DI8持续超过阈值   → 废料口堵料，立即停止
DI8尚未恢复无料         → 不允许确认堵料报警
```

## 15. 故障和恢复方式

状态机通过 `fault_recovery` 明确每种故障允许的恢复操作。“报警复位”按钮始终可以关闭蜂鸣器，但只有 `ACKNOWLEDGE` 类型在恢复条件满足时会同时解除故障：

| 恢复类型 | 典型故障 | 恢复操作 |
|---|---|---|
| `ACKNOWLEDGE` | 堵料已清除、受控停止超时、清线超时 | 消音并确认报警，回到停止状态 |
| `PURGE_REQUIRED` | FIFO溢出/下溢、结果未就绪、物料到达或出口确认超时、走错出口、非预期出口产品、多料同视野、产品间距过小、吹气中断、吹气窗口冲突 | 可消音，但仍必须一键清线 |
| `RECONNECT_IO` | IO未就绪、DO写入失败 | 可尝试消音，但仍必须重新连接IO |

所有恢复方式完成后皮带均保持停止，不自动启动。

## 16. DO 输出和仲裁

状态机输出业务动作：

```python
set_output("conveyor_run", True)
set_output("waste_removal", False)
```

调用关系：

```text
ConveyorLineController
        ↓ 业务输出
OutputArbiter
        ↓
IoController
        ↓ active_high转换
NK IO板卡
```

### 16.1 DO3、DO4安全关闭

安全和故障停止使用一次批量板卡写入：

```python
set_outputs(
    {
        "conveyor_run": False,
        "waste_removal": False,
    },
    force=True,
)
```

这样可避免先关闭一个输出、再关闭另一个输出之间的时间窗口。

### 16.2 正常清线完成顺序

正常清线完成不是紧急停止，按工艺顺序执行：

```text
先关闭DO4皮带
再关闭DO3吹气
```

### 16.3 蜂鸣器仲裁

DO8同时服务于：

```text
检测NG短鸣
产线故障持续鸣响
```

最终状态为：

```text
DO8 = result_buzzer OR line_fault_buzzer
```

NG短鸣定时结束不会关闭仍处于产线故障状态的蜂鸣器。

## 17. 产线状态机

```mermaid
stateDiagram-v2
    [*] --> BOOTING
    BOOTING --> SAFETY_LOCKED: DI5无效或IO未准备
    BOOTING --> DOOR_OPEN_STOPPED: 门打开
    BOOTING --> READY_STOPPED: 安全、门和IO正常

    READY_STOPPED --> RUNNING: DI2启动
    RUNNING --> CONTROLLED_STOPPING: DI3停止
    CONTROLLED_STOPPING --> READY_STOPPED: 已提交动作完成

    RUNNING --> SAFETY_PAUSED: DI5失效
    RUNNING --> DOOR_PAUSED: DI9或DI10失效
    SAFETY_PAUSED --> READY_TO_RESUME: 安全恢复
    DOOR_PAUSED --> READY_TO_RESUME: 门恢复
    READY_TO_RESUME --> RUNNING: 人工重新启动

    READY_STOPPED --> PURGE_PREPARING: 一键清线
    READY_TO_RESUME --> PURGE_PREPARING: 一键清线
    PURGE_PREPARING --> PURGE_RUNNING: 预吹气完成
    PURGE_RUNNING --> READY_STOPPED: 清线完成
    PURGE_RUNNING --> PURGE_PAUSED: 停止或安全中断
    PURGE_PAUSED --> PURGE_PREPARING: 人工继续清线

    RUNNING --> FAULT_STOPPED: 堵料或流程异常
    PURGE_RUNNING --> FAULT_STOPPED: 堵料或清线超时
    FAULT_STOPPED --> READY_STOPPED: 报警确认
    FAULT_STOPPED --> PURGE_PREPARING: 必须清线
```

状态含义：

| 状态 | 含义 |
|---|---|
| `BOOTING` | 程序与IO初始化 |
| `SAFETY_LOCKED` | 安全继电器或IO未准备 |
| `DOOR_OPEN_STOPPED` | 门打开，禁止动作 |
| `READY_STOPPED` | 皮带停止，等待启动或清线 |
| `RUNNING` | 正常连续生产 |
| `CONTROLLED_STOPPING` | 等待已经提交的有限动作完成 |
| `SAFETY_PAUSED` | 运行中安全失效，FIFO冻结 |
| `DOOR_PAUSED` | 运行中开门，FIFO冻结 |
| `READY_TO_RESUME` | 安全恢复，等待人工启动 |
| `PURGE_PREPARING` | DO3预吹气，DO4尚未启动 |
| `PURGE_RUNNING` | DO3持续吹气且DO4运行 |
| `PURGE_PAUSED` | 清线被安全或停止事件中断 |
| `FAULT_STOPPED` | 堵料、队列、吹气、清线或IO故障 |

## 18. 操作互锁

以下情况禁止手动拍照、相机重连和相机参数应用：

- 正常生产运行；
- 受控停止中；
- 清线准备、运行或暂停；
- 第一段FIFO或第二段出口等待队列中仍有在途物料；
- 仍有采图任务未完成；
- DO4或DO3仍处于动作状态。

只有产线停止、两段在途队列均为空、没有采图任务、没有清线上下文时才允许调试和相机配置操作。

## 19. 控制参数

当前规范参数名称：

```json
{
  "poll_interval_ms": 10,
  "debounce_ms": 20,
  "capture_commit_guard_ms": 250,
  "inspection_result_wait_timeout_ms": 3000,
  "controlled_stop_timeout_ms": 1500,
  "reject_blow_delay_ms": 0,
  "reject_blow_duration_ms": 300,
  "max_inflight_items": 20,
  "front_to_reject_max_run_ms": 5000,
  "front_sensor_max_active_ms": 0,
  "front_sensor_min_clear_ms": 0,
  "good_outlet_arrival_min_run_ms": 500,
  "good_outlet_arrival_max_run_ms": 3000,
  "waste_outlet_arrival_min_run_ms": 500,
  "waste_outlet_arrival_max_run_ms": 3000,
  "end_test_sensor_enabled": true,
  "waste_outlet_confirmation_enabled": false,
  "upper_door_sensor_enabled": false,
  "end_test_blocked_timeout_s": 3.0,
  "good_outlet_blocked_timeout_s": 3.0,
  "waste_outlet_blocked_timeout_s": 3.0,
  "purge_air_lead_ms": 200,
  "purge_min_run_s": 10.0,
  "purge_tail_run_s": 5.0,
  "purge_quiet_s": 2.0,
  "purge_max_run_s": 30.0
}
```

标准JSON不允许在每行后直接写 `// 中文备注`。默认配置使用合法的 `_comments` 对象逐项保存中文解释，`ConveyorConfig.from_mapping()`只读取数据字段并自动忽略 `_comments`。程序仍兼容旧参数名，但默认配置、代码和文档统一使用以上规范名称。

`front_sensor_max_active_ms`和`front_sensor_min_clear_ms`当前为0，即保护逻辑已实现但暂未启用。必须先现场测出单件最长遮挡和最短正常无料间隙，再设置非零阈值，避免误报警。

### 19.1 DI0产品间距保护代码示例

`front_sensor_max_active_ms`检查DI0单次遮挡是不是太长：

```text
正常单件：DI0有效 ───── 120ms ───── DI0恢复
两件贴料：产品1遮挡 ── 产品2继续遮挡 ── 总共400ms
```

例如设置 `"front_sensor_max_active_ms": 200`，DI0连续有效超过200ms时，`_monitor_front_sensor_spacing()`报告 `PRODUCT_SPACING_TOO_SMALL`。该判断也能发现产品卡在DI0或传感器粘连。

`front_sensor_min_clear_ms`检查两件产品之间DI0恢复无料的时间是不是太短：

```text
产品1离开 → DI0恢复无料100ms → 产品2到达
```

例如设置 `"front_sensor_min_clear_ms": 50`，第二次DI0有效沿到达时，如果之前无料时间不足50ms，`_on_camera_sensor()`报告 `PRODUCT_SPACING_TOO_SMALL`，并且不会再为这次异常触发创建正常产品记录。

| 参数 | 代码检查内容 |
|---|---|
| `front_sensor_max_active_ms` | DI0一次持续有效时间上限 |
| `front_sensor_min_clear_ms` | 两次DI0有效沿之间的最短无料时间 |

两个阈值和出口确认时间一样，都按皮带实际运行时间计算，停止和急停期间不累计。当前值为0表示禁用。以上120ms、400ms、200ms、100ms和50ms只用于解释代码，必须在最快皮带速度下测量单件最长遮挡、正常最小无料间隙后再留余量设置。

### 19.2 视觉多产品结果接口

`application/runtime/conveyor.py / _reported_product_count()`从算法响应中读取：

```text
product_count
detected_product_count
instance_count
object_count
```

例如算法返回：

```json
{
  "product_count": 2
}
```

运行接入层把最终结果改为 `MULTIPLE_PRODUCTS_IN_FOV`，领域状态机随后关闭DO4、DO3并要求一键清线。

当前代码只完成了数量字段接入和故障处理，没有新增通用视觉计数算法。如果画面实际有两件，但现有算法只找一个最佳匹配并返回1或不返回数量字段，程序不会在拍照阶段立即知道画面中有两件。后续可能表现为：

- 两件分别触发DI1但FIFO只有一件，产生 `FIFO_UNDERFLOW`；
- DI7/DI8只产生一个有效沿，第二件出口确认超时；
- DI0形成长遮挡或无料间隔过短，产生 `PRODUCT_SPACING_TOO_SMALL`；
- 所有传感器都把两件当成一个长脉冲且视觉也只返回一件时，纯IO无法保证发现多料。

完整保护链为：

```text
机械分料保证间距
→ DI0长遮挡/最短无料间隔
→ 视觉算法输出产品数量
→ DI7/DI8逐件出口确认
```

机械和DI0是第一道保护，视觉产品计数是第二道确认，DI7/DI8是最终出料校验。

## 20. 从输入到输出的完整调用示例

以“DI0检测到产品，最终NG吹气”为例：

```text
1. NK板卡DI0原始电平由1变0
2. IoController读取完整DI字
3. active_high=false把DI0转换为业务True
4. DiPoller确认去抖时间满足
5. DiPoller生成camera_trigger_sensor=True事件
6. ConveyorLineController处理DI0事件
7. WorkpieceTracker创建物料#1001并加入FIFO
8. application/runtime/conveyor.py提交#1001采图任务
9. 后台算法返回#1001=NG
10. 状态机按sequence_id和epoch把#1001改为NG
11. 产品到达DI1，DI1产生reject_position_sensor=True事件
12. 状态机读取FIFO队头#1001
13. 从第一段FIFO移除#1001
14. RejectBlowController建立延时和吹气窗口
15. OutletConfirmationTracker建立#1001等待DI8记录
16. 周期tick发现吹气窗口已经开始
17. 状态机请求waste_removal=True
18. OutputArbiter接收DO3业务请求
19. IoController根据active_high=false转换物理电平
20. NK板卡DO3动作，电磁阀持续吹气
21. 吹气窗口结束后DO3业务状态恢复False
22. 产品在合法到达窗口内触发DI8有效沿
23. OutletConfirmationTracker移除#1001，确认NG剔除成功
```

## 21. 现场验证重点

软件自动化测试不能替代现场联机验收。上线前至少确认：

1. DI0、DI1低有效极性和完整边沿是否正确。
2. DI2高有效、DI3低有效是否与实体按钮一致。
3. DI5、DI9、DI10任一路失效时，DO4和DO3是否立即关闭。
4. DO3业务ON是否持续吹气，OFF是否停止吹气。
5. DO4业务ON/OFF是否对应皮带实际启停。
6. `reject_blow_delay_ms`和`reject_blow_duration_ms`是否覆盖NG且不会误吹后续GOOD。
7. DI6、DI7、DI8堵料阈值是否大于正常产品遮挡时间。
8. 清线时是否按“DO3预吹气→DO4运行→DO4停止→DO3停止”执行。
9. DI8有料能否启动清线，持续堵料能否在阈值后停机。
10. 急停、开门和清线中断恢复后是否始终保持停止并等待人工操作。
11. GOOD是否逐件触发DI7、NG是否逐件触发DI8，未到达和走错出口是否正确停机。
12. 两件产品最小间距下DI0、DI1、DI7、DI8是否都能恢复无效并形成两个独立有效沿。
13. 视觉算法是否输出 `product_count`、`detected_product_count`、`instance_count` 或 `object_count`；若没有，当前框架无法凭空识别同一视野多件产品。

## 22. 相关源代码入口

- `domain/conveyor_line.py`
  - `ConveyorLineController.handle_input_change()`：所有DI事件入口；
  - `request_start()`：启动；
  - `request_controlled_stop()`：受控停止；
  - `request_purge()`：一键清线；
  - `continue_purge()`：继续清线；
  - `inspection_completed()`：检测结果回写；
  - `tick()`：周期定时动作；
  - `_on_camera_sensor()`：DI0物料创建；
  - `_on_reject_sensor()`：DI1 FIFO消费；
  - `_on_outlet_sensor()`：DI7/DI8逐件确认、走错出口和非预期出口处理；
  - `_handle_interlock_change()`：DI5、DI9、DI10联锁；
  - `_monitor_jams()`：堵料监控；
  - `_monitor_outlet_timeouts()`：DI7/DI8未到达超时；
  - `_monitor_front_sensor_spacing()`：DI0长遮挡和最小无料间隙保护；
  - `_trip_fault()`：故障锁存和安全停止。
- `domain/conveyor_components.py`
  - `WorkpieceTracker`：FIFO；
  - `OutletConfirmationTracker`：GOOD→DI7、NG→DI8的合法到达窗口和逐件确认；
  - `RejectBlowController`：NG吹气窗口；
  - `AutoPurgeController`：清线上下文；
  - `JamMonitor`：堵料运动时间。
- `devices/io_controller.py`
  - `snapshot_inputs()`：单次整字DI快照；
  - `set_output()`：单个业务输出；
  - `set_outputs()`：多个输出批量写入。
- `devices/output_arbiter.py`
  - `set_line_output()`：产线单输出；
  - `set_line_outputs()`：产线批量输出；
  - `set_result_buzzer()`：检测结果蜂鸣请求。

## 23. 软件工程架构

### 23.1 架构类型

当前系统的软件架构定义为：

```text
模块化单体应用
+ 分层架构
+ 事件驱动
+ 中央状态机
+ 独立领域组件
+ 异步视觉任务
+ 统一IO输出仲裁
```

系统运行在同一个应用进程中，不是微服务。不同职责通过 Python 模块和对象边界拆分，避免 UI、算法和硬件控制相互直接修改状态。

### 23.2 分层与依赖方向

```mermaid
flowchart TD
    UI["表现层<br/>RuntimeModePage"]
    APP["应用编排层<br/>RuntimeController、runtime/conveyor.py、runtime/hardware.py"]
    DOMAIN["领域控制层<br/>ConveyorLineController"]
    COMPONENTS["领域组件层<br/>FIFO、吹气、清线、堵料"]
    DEVICES["设备抽象层<br/>IoController、DiPoller、OutputArbiter"]
    DRIVER["硬件驱动层<br/>NkioBoard、相机SDK"]
    CONFIG["配置层<br/>IO映射、控制参数、相机参数"]

    UI --> APP
    APP --> DOMAIN
    DOMAIN --> COMPONENTS
    APP --> DEVICES
    DEVICES --> DRIVER
    CONFIG --> APP
    CONFIG --> DEVICES

    DOMAIN -. "通过回调表达输出、检测和状态发布意图" .-> APP
```

依赖规则：

1. UI 只发送操作意图并显示状态快照，不决定 DO 动作。
2. 应用层负责装配对象和协调相机、算法、IO、UI。
3. 领域层负责产线业务规则，不依赖 PySide6、NK板卡SDK或具体界面。
4. 领域组件只能管理各自的数据和计时，不直接写物理IO。
5. 设备层负责硬件转换和读写，不决定生产工艺。
6. 算法和采图任务只能返回结果，不能直接操作DO。

### 23.3 主要对象关系

```mermaid
classDiagram
    class RuntimeController {
        +连接相机()
        +初始化IO()
        +提交采图和检测任务()
        +发布界面状态()
    }

    class ConveyorLineController {
        +handle_input_change()
        +request_start()
        +request_controlled_stop()
        +request_purge()
        +inspection_completed()
        +tick()
        +snapshot()
    }

    class WorkpieceTracker {
        +fifo
        +epoch
        +create()
        +head()
        +pop_head()
        +invalidate_for_purge()
    }

    class RejectBlowController {
        +schedule()
        +is_active()
        +has_pending
    }

    class AutoPurgeController {
        +begin()
        +restart_lead()
        +record_activity()
        +clear()
    }

    class JamMonitor {
        +observe_input()
        +first_timeout()
        +reset()
    }

    class DiPoller {
        +start()
        +snapshot()
        +add_change_callback()
    }

    class IoController {
        +snapshot_inputs()
        +set_output()
        +set_outputs()
    }

    class OutputArbiter {
        +set_line_output()
        +set_line_outputs()
        +set_result_buzzer()
    }

    RuntimeController --> ConveyorLineController
    RuntimeController --> DiPoller
    RuntimeController --> IoController
    ConveyorLineController --> WorkpieceTracker
    ConveyorLineController --> RejectBlowController
    ConveyorLineController --> AutoPurgeController
    ConveyorLineController --> JamMonitor
    ConveyorLineController --> OutputArbiter : 注入输出回调
    DiPoller --> IoController
    OutputArbiter --> IoController
```

`ConveyorLineController` 是业务决策中心，其他领域组件不能互相直接启动动作。例如 `JamMonitor` 只报告哪个输入超时，由状态机决定进入 `FAULT_STOPPED` 并关闭输出。

### 23.4 依赖注入和硬件隔离

领域状态机创建时由应用层注入功能回调：

```python
ConveyorLineController(
    output_writer=write_output,
    output_batch_writer=write_outputs,
    inspection_requester=enqueue_inspection,
    state_listener=publish_state,
    log_writer=write_log,
    start_authorizer=prepare_start,
)
```

状态机只表达业务动作：

```python
set_output("conveyor_run", True)
set_output("waste_removal", False)
```

状态机不需要知道：

- DO4在板卡上的实际端子；
- 输出是高有效还是低有效；
- 当前使用真实NK板卡还是测试用假IO；
- 状态最终显示在哪个Qt控件上；
- 检测任务由哪个线程执行。

因此领域逻辑可以在没有相机和真实IO板卡的情况下执行自动化测试。

### 23.5 事件驱动架构

中央状态机接收以下事件：

```text
DI稳定边沿事件
周期tick事件
采图完成事件
算法结果事件
启动/停止/清线界面命令
IO读取或写入错误
```

```mermaid
flowchart LR
    DI["DI事件"] --> STATE["ConveyorLineController"]
    TICK["周期tick"] --> STATE
    RESULT["算法结果"] --> STATE
    UI["界面命令"] --> STATE
    ERROR["IO错误"] --> STATE

    STATE --> SNAPSHOT["状态快照"]
    SNAPSHOT --> VIEW["UI显示"]
    STATE --> REQUEST["业务输出请求"]
    REQUEST --> ARBITER["OutputArbiter"]
```

每次事件处理后，状态机发布完整状态快照。UI根据快照显示，不在界面代码中重新推导产线规则。

典型快照：

```python
{
    "state": "RUNNING",
    "fifo_count": 3,
    "fifo": [...],
    "fault_code": "",
    "fault_recovery": "ACKNOWLEDGE",
    "manual_operations_permitted": False,
    "configuration_operations_permitted": False,
    "motion_elapsed_s": 12.5,
    "inputs": {...},
    "outputs": {...},
}
```

### 23.6 线程模型

```mermaid
flowchart TD
    QT["Qt主线程<br/>UI、普通DI事件、周期tick和操作命令"]
    DI["DI轮询线程<br/>整字读取、去抖"]
    CAPTURE["相机采图工作线程"]
    ALGORITHM["算法检测工作线程"]
    STATE["带可重入锁的中央状态机"]
    OUTPUT["OutputArbiter和IO板卡锁"]

    DI -->|"普通DI通过Qt信号"| QT
    DI -->|"DI5/DI9/DI10直接处理"| STATE
    QT --> STATE
    CAPTURE --> STATE
    ALGORITHM -->|"按编号回写结果"| STATE
    STATE --> OUTPUT
```

当前模型不是严格的单线程 Actor，而是：

```text
多个事件来源
→ 同一个带锁的中央状态机
→ 统一输出通道
```

线程规则：

1. 普通DI、UI命令和周期tick主要在Qt主线程处理。
2. DI轮询在线程中持续运行，避免阻塞界面。
3. DI5、DI9、DI10安全边沿直接进入线程安全状态机，避免等待UI事件队列。
4. 相机和算法工作在线程中执行，完成后只回写对应物料结果。
5. 状态机所有公开入口使用同一把可重入锁，避免FIFO和状态并发修改。
6. IO板卡底层也通过锁串行化硬件调用。

### 23.7 控制流与数据流分离

控制流决定设备是否动作：

```text
DI/定时/UI/错误事件
→ 中央状态机
→ 输出仲裁
→ DO3、DO4、按钮灯和蜂鸣器
```

数据流决定每件产品的检测结果：

```text
DI0
→ sequence_id + epoch
→ 相机图像
→ 后台算法
→ GOOD/NG
→ 按编号写回FIFO
```

两条流在 DI1 汇合：

```text
DI1确认实物到达
        +
FIFO队头的检测结果
        ↓
状态机决定通过、吹气或故障停机
```

DI1之后再进入出口确认数据流：

```text
GOOD → OutletConfirmationTracker等待DI7 ─┐
NG   → 吹气 + OutletConfirmationTracker等待DI8 ─┤
                                              ↓
正确出口有效沿 → 完成
未到达/走错出口/非预期出口信号 → 停机并要求清线
```

### 23.8 架构边界

当前架构明确禁止：

- UI直接写DO；
- 算法线程直接启动吹气；
- DI轮询器直接修改FIFO；
- `WorkpieceTracker`自行启动相机；
- `JamMonitor`自行写皮带输出；
- 检测结果绕过 `sequence_id + epoch` 更新全局结果；
- 三色灯短鸣定时器直接覆盖产线故障蜂鸣状态。

架构保留的安全边界：

```text
软件状态机负责工艺控制和软件联锁
硬件安全继电器负责人员安全和执行机构安全断电
```

软件快速关闭DO3、DO4是必要的防自动恢复措施，但不能替代硬件安全回路。

## 24. 代码重构后的结构约束

本轮重构不改变皮带动作时序、FIFO规则、DI7/DI8出口确认规则和清线规则，主要调整软件依赖关系及错误诊断能力。

### 24.1 运行控制器不再动态注入方法

旧结构在模块导入时通过 `setattr()` 把硬件、检测、皮带和状态方法写入 `RuntimeController`，方法来源和依赖不够明确。当前已经取消动态绑定：

```text
RuntimeController
├── RuntimeStatusService          状态计算和界面状态发布
├── RuntimeConveyorService        皮带运行接入
├── RuntimeHardwareService        IO和相机硬件接入
└── RuntimeExecutionService       手动检测执行
```

Mixin只保留原方法名称的兼容委托，实际职责由组合服务对象承担；程序启动时不会再修改控制器类。

相关代码：

- `application/runtime/controller.py`
- `application/runtime/status_bus.py`
- `application/runtime/operation_mixins.py`

### 24.2 调试工具页不再动态注入方法

`ToolPage` 已取消导入时 `setattr()`，原有操作由 `ToolPageOperationsMixin` 显式声明。现有的ROI、产品会话、样本列表、训练和测试控制器继续作为实际职责组件使用。

```text
ToolPage
├── RoiAnnotationController
├── ProductSessionController
├── SampleListController
├── TrainingController
├── TestExecutionController
└── ToolPageOperationsMixin       尚未迁移功能的兼容过渡层
```

相关代码：`ui/debug/tool_page/operation_mixins.py`。

### 24.3 调试检测和正式运行共用检测分流

`ToolPageRuntimeContext` 与 `ProductRuntimeContext` 现在共用以下处理：

```text
检测项列表
→ 过滤禁用项
→ 学习算法 / 传统算法 / 测量算法统一分组
→ 调用对应预测入口
→ 按原检测项顺序返回结果
```

这样可以避免调试检测和皮带自动检测分别维护算法分流规则。定位方式和产品数据来源仍由各自运行上下文提供。

相关代码：`application/runtime_context.py`中的 `InspectionItemGroups`、`_group_inspection_items()` 和 `_predict_grouped_items_from_path()`。

### 24.4 皮带参数在启动前严格校验

`ConveyorConfig` 会在对象创建和JSON加载时检查：

- 布尔参数必须是JSON布尔值，字符串 `"false"` 不再被接受；
- 时间、消抖和延迟参数不得为负数或非有限数；
- DI轮询周期和最大在途数量必须大于0；
- GOOD/废料出口最早到达时间不得大于最晚到达时间；
- 清线最短、尾部和安静时间不得超过清线最大时间。

配置不合法时，程序在建立产线控制器之前报告错误，不允许带错误参数进入生产状态。

### 24.5 强类型状态快照与兼容接口

状态机新增 `ConveyorSnapshot`，内部状态字段具有明确类型。已有界面和运行层仍可继续调用 `snapshot()` 获取字典；新代码可以调用 `snapshot_model()` 获取强类型对象。

```text
snapshot_model() → ConveyorSnapshot
                         ↓ to_dict()
snapshot()       → 原有dict兼容接口
```

### 24.6 安全输出失败诊断

当批量DO写入失败时，状态机会继续尝试分别关闭皮带和废料吹气。如果安全关闭写入也失败：

```text
OUTPUT_WRITE_FAILED
safe-off failed, physical output state unknown
```

故障详情会明确指出物理输出状态未知，并记录每个关闭失败的输出。IO、塔灯、光源和相机管理器关闭失败也会写入运行日志，不再静默忽略。

### 24.7 回归测试

本轮新增测试覆盖：

- 皮带配置类型、范围和关联参数校验；
- 安全输出关闭失败诊断；
- 强类型快照与原字典快照兼容性；
- 调试/正式运行共用检测分流和结果顺序；
- RuntimeController、ToolPage不再依赖动态方法绑定；
- 运行状态文本中的重复耗时字段清理。

### 24.8 共享采图流水线和自动ROI规格

手动检测和皮带检测共用 `application/runtime/capture_pipeline.py`：

```text
应用相机参数
→ 准备灯光
→ 等待灯光稳定
→ 通知采图状态
→ 相机采图
→ finally关闭灯光
→ RuntimeCapturedFrame
```

无论相机采图成功还是抛出异常，已经打开的灯光都会在 `finally` 中关闭。两种运行模式不再分别维护这段安全清理逻辑。

自动ROI的Shape/NCC引用检查、标签和工作线程参数由 `application/auto_roi_service.py` 中的 `AutoRoiExecutionSpec` 统一生成，工具页批量生成与样本预览窗口共用同一套规格。

### 24.9 检测执行计划

`InspectionExecutor` 新增 `InspectionExecutionPlan`，在执行前明确分出：

- 普通预测项；
- 后处理距离项；
- 禁用项。

这一步保持现有结果计算不变，为后续把不同算法拆成独立Handler提供稳定边界和特征测试。

## 25. 第二阶段架构收敛

### 25.1 RuntimeController使用完整对象组合

皮带、硬件、检测执行和状态发布现在分别由独立服务对象持有：

```text
RuntimeController
├── RuntimeConveyorService
├── RuntimeHardwareService
├── RuntimeExecutionService
└── RuntimeStatusService
```

`RuntimeController`保留原方法名称作为兼容入口，入口只把参数委托给服务对象。业务函数不再直接作为控制器类方法挂载。对应文件：

- `application/runtime/conveyor_service.py`
- `application/runtime/hardware_service.py`
- `application/runtime/execution_service.py`
- `application/runtime/status_bus.py`
- `application/runtime/operation_mixins.py`

这样可以单独测试服务边界，并逐步把旧业务函数内部的 `runtime._xxx` 依赖替换为明确依赖参数。

### 25.2 多语言资源按功能域拆分

`ui/i18n.py`只保留语言选择、持久化、翻译查找和状态文本转换。实际中英文资源位于：

```text
ui/i18n_resources/
├── zh_cn/
│   ├── shell.py
│   ├── runtime.py
│   ├── debug.py
│   ├── template.py
│   ├── ncc.py
│   └── access.py
└── en_us/
    └── 同样的功能域结构
```

资源加载时检查重复键；测试继续检查中英文键集合完全一致以及格式化占位符一致。

### 25.3 InspectionExecutor执行阶段拆分

相机检测执行现在分为三个明确阶段：

```text
InspectionExecutionPlan
        ↓
PrimaryInspectionBatch      普通学习/传统/测量工具预测
        ↓
DistanceInspectionBatch     中心距离和直线距离后处理
        ↓
决策策略、相机结果和耗时汇总
```

禁用项、普通预测项和距离项不再全部混在 `execute()` 的同一个大循环中。

### 25.4 大型UI构建函数分段

运行界面和调试工具页的控件名称、布局层级和信号连接保持不变，但构建函数按稳定区域拆开：

```text
RuntimeModePage
├── _build_header_ui
├── _build_runtime_body_ui
├── _build_footer_ui
└── _build_compatibility_controls

ToolPage
├── _build_header_ui
├── _build_workspace_ui
├── _build_footer_ui
└── _build_compatibility_ui
```

以后修改顶栏、相机主体区、底栏或兼容控件时，不再需要进入一个数百行的单体构建函数。


NG吹掉之后界面上的FIFO的红灯过很久才灭和消失吗
