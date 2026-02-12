# MiniMemory / MiniCache Server

一个轻量级、类 Redis 的内存 KV 服务端实现，使用 RESP 协议收发命令，支持基础 KV、数据库选择、事务、过期时间、持久化（MCDB 快照 + AOF 追加日志）等。

## 目录结构

- 配置模板：`src/conf/mcs.conf`
- 测试用例参考：`docs/test.md`
- 构建产物（默认）：`build/bin/`
  - 服务端：`mini_cache_server(.exe)`
  - 客户端：`mini_cache_cli(.exe)`
  - 默认配置目录：`build/bin/conf/`

## 构建

### Windows（推荐）

```powershell
cmake -S . -B build
cmake --build build --config Release
```

构建完成后，可执行文件通常位于：

- `build/bin/mini_cache_server.exe`
- `build/bin/mini_cache_cli.exe`

### Linux / macOS

```bash
cmake -S . -B build
cmake --build build -j
```

## 配置说明

服务端默认读取 `conf/mcs.conf`，也可以通过命令行指定：

```bash
mini_cache_server --config conf/mcs.conf
```

### 常用配置项

以 `build/bin/conf/mcs.conf` 为例：

```conf
bind 0.0.0.0
port 6379
#requirepass password123

appendonly yes
appendfilename ./data/appendonly.aof
appendfsync everysec

save 1 1
save 15 10
save 60 1000

maxmemory 2gb
maxmemory-policy allkeys-lru
```

- `bind` / `port`：监听地址与端口
- `requirepass`：密码（当前工程对认证状态的校验逻辑较简化）
- `appendonly`：是否启用 AOF
- `appendfilename`：AOF 文件路径
  - 建议写相对路径（如 `./data/appendonly.aof`）
  - 程序会将相对路径解析为“相对配置文件所在目录”，避免工作目录变化导致找不到文件
- `save <seconds> <changes>`：MCDB 快照保存条件（类似 RDB）
- `maxmemory` / `maxmemory-policy`：内存限制与淘汰策略（策略实现依赖 DataStore 的具体逻辑）

### 持久化模式（快照 / AOF）

当前支持两类持久化：

- **MCDB 快照（dump.mcdb）**：由 `save <seconds> <changes>` 规则触发，或手动执行 `SAVE` 触发一次落盘。
- **AOF 追加日志（appendonly.aof）**：由 `appendonly yes` 开启，后续写命令会追加到 AOF；`appendfsync` 控制刷盘频率。

两者可以同时开启。重启恢复时请以服务端启动日志为准（例如会打印 `Resolved data paths` 以及实际加载/恢复过程）。

### 持久化文件位置

当 `appendfilename` 使用相对路径时，AOF 最终会写到“配置文件所在目录”下。例如：

- 配置：`build/bin/conf/mcs.conf`
- `appendfilename ./data/appendonly.aof`
- 实际文件：`build/bin/conf/data/appendonly.aof`

同理，MCDB 快照文件默认在配置目录下：

- `build/bin/conf/dump.mcdb`

如果你看到 `build/bin/data/appendonly.aof` 或 `build/bin/dump.mcdb`，通常是旧版本/其他路径产生的文件，请以服务端启动时打印的 `Resolved data paths` 为准。

## 启动服务器

在构建目录下启动（推荐从 `build/bin` 目录运行）：

```powershell
cd build/bin
.\mini_cache_server.exe --config conf/mcs.conf
```

启动成功后会看到类似输出：

- `Server is listening on 0.0.0.0:6379`
- `MiniCache server started on 0.0.0.0:6379`

## 使用客户端连接与执行命令

```powershell
cd build/bin
.\mini_cache_cli.exe -h 127.0.0.1 -p 6379
```

进入交互界面后，命令示例（更多用例见 `docs/test.md`）：

```text
MC> PING
MC> SET mykey "Hello World"
MC> GET mykey
MC> EXISTS mykey
MC> DEL mykey
MC> GET mykey
```

### 持久化测试（参考）

```text
MC> SET persist_key "This will be saved"
MC> SAVE
# 重启服务器后
MC> GET persist_key
```

如果你只想验证 AOF 追加模式，可以在 `appendonly yes` 情况下：

- 不执行 `SAVE`，直接重启服务器，再 `GET` 验证是否能恢复
- 如果关闭 `appendonly`，则需要依赖 `SAVE` 或 `save` 规则触发快照后才能在重启后恢复

## 在其他 C++ 项目中复用

本仓库不仅提供服务端/客户端可执行文件，也暴露了若干可复用的库目标（CMake）：

- `data_store`：核心存储引擎
- `resp_parser`：RESP 解析
- `command_handler`：命令处理（基于 DataStore）
- `config_parser`：配置解析
- `tcp_server`：TCP 服务器实现（依赖上述模块）

### 方式 A：作为子工程（推荐）

将本仓库作为子目录（或 git submodule）引入你的项目，例如目录结构：

```text
your_project/
  CMakeLists.txt
  third_party/
    MiniMemory/
```

在你的 `CMakeLists.txt` 中：

```cmake
add_subdirectory(third_party/MiniMemory)

add_executable(your_app main.cpp)
target_link_libraries(your_app PRIVATE data_store command_handler resp_parser)
target_include_directories(your_app PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/third_party/MiniMemory/src/server
)
```

然后在代码里直接包含头文件，例如：

```cpp
#include "DataStore.hpp"
#include "CommandHandler.hpp"
```

### 方式 B：链接已构建产物

先在 MiniMemory 中完成构建（产物在 `build/bin`），然后在你的项目里：

- 头文件：添加 `MiniMemory/src/server` 到 include 路径
- 库文件：链接 `data_store` / `command_handler` / `resp_parser` 等对应的 `.lib/.dll`（Windows）或 `.so/.dylib`（Linux/macOS）
- 运行时：确保依赖的动态库与可执行文件在同一目录，或在系统库搜索路径中可见

## 协议兼容性

客户端与服务端通过 RESP 协议交互，命令格式与示例可参考 `docs/test.md`。
