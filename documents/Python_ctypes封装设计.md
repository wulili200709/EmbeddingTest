# Python ctypes 封装设计

## 1. 文档目的

本文档用于明确第一版 Python 侧对以下 DLL 的最小封装方案：

- `NKIOLIBx64.dll`
- `NKLCLIBx64.dll`

目标不是一次性把全部 SDK 功能封满，而是先为后续运行界面和调试界面提供稳定、可测试、可扩展的 Python 接口。

## 2. 当前结论

基于现有需求和现场确认信息，第一版硬件结论如下：

- 板卡型号：`NP-6133-16I16O`
- 脚踏：走 `DI`
- 三色灯：走普通 `DO`
- 光源：走普通 `DO`
- 三色灯：高电平点亮
- 光源有效电平：待现场确认

因此第一版真正必须打通的是：

- `NKIOLIBx64.dll`

`NKLCLIBx64.dll` 当前不作为第一版正式依赖，但建议保留设计入口，方便后续扩展。

换句话说：

- `V1` 正式路径只支持 `DO` 开关光源
- `NKLC` 仅作为未来扩展预研，不进入 `V1` 主流程

## 3. 封装范围

### 3.1 第一版必须支持

来自 `NKIOLIBx64.dll`：

- `NKDIO_LibraryInit`
- `NKDIO_LibraryDeinit`
- `NKDIO_PollingReadDiByte`
- `NKDIO_PollingReadDiWord`
- `NKDIO_PollingWriteDoByte`
- `NKDIO_PollingWriteDoWord`
- `NKDIO_PollingReadDoByte`
- `NKDIO_PollingReadDoWord`

业务上需要的能力：

- 初始化 SDK
- 读取全部 16 路 DI
- 读取某一路 DI
- 写全部 16 路 DO
- 写某一路 DO
- 回读全部 16 路 DO
- 根据 `active_high` 自动做极性转换

### 3.2 第一版暂不作为正式依赖

来自 `NKLCLIBx64.dll`：

- `NKLC_LibraryInit`
- `NKLC_LibraryDeinit`
- `NKLC_OpenDevice_Async`
- `NKLC_CloseDevice_Async`
- `NKLC_Process_Async`
- `NKLC_SetPwmParams_Async`
- `NKLC_GetPwmParams_Async`

理由：

- 当前光源控制已确认走普通 `DO`
- 第一版业务流程不依赖串口 PWM 光源控制
- 如果一开始把异步回调也一起封装，会增加调试复杂度
- 第一版运行链路、联调脚本和异常处理都应统一围绕 `DO` 光源开关实现

但建议保留预研接口和模块位置，后面如果改光源方案可直接接入。

## 4. SDK 原始接口

根据头文件 `NKIOLIB.h`，建议 Python 侧对应以下签名：

```text
int NKDIO_LibraryInit(const char *configFile)
void NKDIO_LibraryDeinit(void)
int NKDIO_PollingReadDiByte(unsigned char diByteIndex, unsigned char *pByteValue)
int NKDIO_PollingReadDiWord(unsigned char diWordIndex, unsigned short *pWordValue)
int NKDIO_PollingWriteDoByte(unsigned char doByteIndex, unsigned char doByteValue)
int NKDIO_PollingWriteDoWord(unsigned char doWordIndex, unsigned short doWordValue)
int NKDIO_PollingReadDoByte(unsigned char doByteIndex, unsigned char *pByteValue)
int NKDIO_PollingReadDoWord(unsigned char doWordIndex, unsigned short *pWordValue)
```

错误码：

- `0` = `NKIO_ENOERR`
- 非 `0` 视为失败

已知错误码：

- `1` `NKIO_EBUSY`
- `2` `NKIO_EINUSED`
- `3` `NKIO_EIO`
- `4` `NKIO_ETIMEOUT`
- `5` `NKIO_EDEVERR`
- `6` `NKIO_EINVAL`
- `7` `NKIO_EFILE`

## 5. Python 封装分层建议

建议分三层封装，不要直接在业务代码中操作 `ctypes`。

### 5.1 原始 DLL 层

文件建议：

- `EmbeddingTest/devices/nkio_raw.py`

职责：

- 负责 `ctypes.WinDLL` 或 `ctypes.CDLL` 加载
- 绑定 `argtypes` / `restype`
- 提供最原始函数调用

这一层不做业务语义，只负责“正确调用 DLL”。

### 5.2 板卡语义层

文件建议：

- `EmbeddingTest/devices/nkio_board.py`

职责：

- 封装初始化 / 反初始化
- 提供 `read_di_word()`、`write_do_word()` 等更好用的接口
- 提供 bit 级读写接口
- 管理当前 DO 缓存状态
- 屏蔽 byte / word 拼装细节

### 5.3 业务适配层

文件建议：

- `EmbeddingTest/devices/io_mapping.py`
- `EmbeddingTest/devices/io_controller.py`

职责：

- 把 `DI0`、`DO3` 这种底层点位映射成业务名称
- 例如：
  - `foot_switch`
  - `tower_red`
  - `tower_green`
  - `tower_blue`
  - `light_cam1`
  - `light_cam2`
- 同时处理 `active_high` / `active_low`

## 6. 推荐模块结构

建议目录：

```text
EmbeddingTest/
  devices/
    nkio_raw.py
    nkio_errors.py
    nkio_board.py
    io_mapping.py
    io_controller.py
    di_monitor.py     # DiMonitor 类，DI 轮询监听、上升沿检测、去抖、触发回调
```

如果后续要保留光源串口控制扩展，可以再补：

```text
EmbeddingTest/
  devices/
    nklc_raw.py
    nklc_light_controller.py
```

## 7. 第一版最小封装接口

### 7.1 `nkio_raw.py`

建议暴露：

```python
class NkioRawLib:
    def __init__(self, dll_path: str | None = None) -> None: ...
    def library_init(self, config_file: str) -> int: ...
    def library_deinit(self) -> None: ...
    def read_di_byte(self, index: int) -> tuple[int, int]: ...
    def read_di_word(self, index: int = 0) -> tuple[int, int]: ...
    def write_do_byte(self, index: int, value: int) -> int: ...
    def write_do_word(self, index: int, value: int) -> int: ...
    def read_do_byte(self, index: int) -> tuple[int, int]: ...
    def read_do_word(self, index: int = 0) -> tuple[int, int]: ...
```

说明：

- 返回值建议保留 `ret_code`
- 由上一层决定是否抛异常

### 7.2 `nkio_errors.py`

建议提供：

```python
class NkioError(RuntimeError): ...
class NkioBusyError(NkioError): ...
class NkioTimeoutError(NkioError): ...
...
```

以及：

```python
def raise_for_code(code: int, operation: str) -> None: ...
```

### 7.3 `nkio_board.py`

建议提供：

```python
class NkioBoard:
    def __init__(self, config_file: str, dll_path: str | None = None) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def read_di_word(self) -> int: ...
    def read_do_word(self) -> int: ...
    def write_do_word(self, value: int) -> None: ...
    def read_di_channel(self, channel: int) -> bool: ...
    def read_do_channel(self, channel: int) -> bool: ...
    def write_do_channel(self, channel: int, on: bool) -> None: ...
    def set_do_channels(self, updates: dict[int, bool]) -> None: ...
```

核心建议：

- 维护一个当前 `do_word_cache`
- 修改单路 `DO` 时，不要直接只写 byte，而是先更新缓存再整体写回
- 避免多模块同时写 DO 导致覆盖别的位

## 8. 位操作约定

建议统一约定：

- `DI0` 对应 bit0
- `DI1` 对应 bit1
- ...
- `DO15` 对应 bit15

位处理示例：

```text
word_value & (1 << channel)
```

建议工具函数：

```python
def get_bit(value: int, bit: int) -> bool: ...
def set_bit(value: int, bit: int, on: bool) -> int: ...
```

## 9. 极性处理设计

这部分很关键，不能把“点亮”和“写 1”直接绑定。

建议定义：

```python
@dataclass
class IoChannelConfig:
    name: str
    channel: int
    active_high: bool
```

统一转换规则：

```text
业务上 on/off -> 逻辑电平 -> 写入 DO
```

转换函数建议：

```python
def business_to_level(on: bool, active_high: bool) -> bool:
    if active_high:
        return on
    return not on
```

这样程序里永远使用：

- `set_output("tower_red", True)`
- 而不是到处手写 `1` 或 `0`

## 10. 建议配置结构

建议单独维护 IO 配置文件，例如：

```yaml
board_model: NP-6133-16I16O
nkio_config_file: C:/path/to/nkio_config.ini

di:
  foot_switch:
    channel: 0

do:
  tower_red:
    channel: 0
    active_high: true
  tower_green:
    channel: 1
    active_high: true
  tower_blue:
    channel: 2
    active_high: true
  light_cam1:
    channel: 3
    active_high: false
  light_cam2:
    channel: 4
    active_high: false
```

说明：

- 三色灯已确认低电平亮，因此示例中三色灯统一使用 `active_high: false`
- 光源极性仍需现场确认，因此示例里先写成 `false` 只是占位，不代表最终结论

## 11. DI 轮询设计

因为 DIO 是轮询模式，建议封装一个独立轮询器：

文件建议：

- `EmbeddingTest/devices/di_monitor.py`

> **说明**：类名统一为 `DiMonitor`，与 `系统模块设计.md` 中对应。`di_monitor.py` 内部采用轮询实现（poll loop 运行在独立线程）。

建议职责：

- 周期读取 `read_di_word()`
- 比较前后值
- 检测上升沿
- 做去抖
- 发出事件给调度层

建议接口：

```python
class DiMonitor:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_callback(self, callback) -> None: ...
```

建议参数：

- `poll_interval_ms = 10 ~ 30`
- `debounce_ms = 30 ~ 100`

## 12. 第一版验证顺序

在正式接入业务前，建议先写最小验证脚本：

### 12.1 DIO 初始化验证

验证内容：

- DLL 是否能加载
- `nkio_config.ini` 是否能正确初始化
- `LibraryInit` 是否返回 `0`

### 12.2 DI 读取验证

验证内容：

- 能否读到 16 路 `DI`
- 脚踏按下时是否能看到对应位变化
- 是否能稳定检测 `0 -> 1`

### 12.3 DO 输出验证

验证内容：

- 能否点亮红 / 绿 / 蓝灯
- 能否控制相机1 / 相机2 光源
- 光源极性是否与预期一致

### 12.4 并发安全验证

验证内容：

- 多线程下是否只允许一个地方写 DO
- 是否需要 `threading.Lock` 保护 `do_word_cache`

建议结论：

- `NkioBoard.write_do_word()` 和 `write_do_channel()` 必须加锁

## 13. 第一版暂不建议做的事

- 不要一开始就把 `NKLCLIB` 异步回调完整封装进去
- 不要一开始就把 `DIO` 和海康相机调度写在同一个模块里
- 不要直接在 UI 按钮事件里操作 DLL
- 不要把点位编号写死在运行逻辑中

## 14. 推荐的最小落地顺序

建议按下面顺序实施：

1. 先写 `nkio_errors.py`
2. 再写 `nkio_raw.py`
3. 再写 `nkio_board.py`
4. 再写一个最小验证脚本
5. 验证脚踏、三色灯、光源
6. 再接入 `DiMonitor`
7. 最后才挂到 `InspectionScheduler`

## 15. 与后续代码的关系

后续真正业务代码建议只依赖：

- `NkioBoard`
- `IoController`
- `DiMonitor`

不建议业务代码直接依赖：

- `ctypes`
- 原始 DLL 函数
- byte / word 位运算细节

## 16. 下一步建议

这份设计文档完成后，下一步最合理的动作就是：

- 直接开始写 `nkio_errors.py`
- `nkio_raw.py`
- `nkio_board.py`

也就是先把 `NKIOLIB` 的最小可用 Python 封装做出来，再开始接运行逻辑。
