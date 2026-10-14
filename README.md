# eino-memory

基于 [CloudWeGo eino](https://github.com/cloudwego/eino) 实现 9 种 AI Agent 记忆优化策略。

## 目录结构

```
eino-memory/
├── llm/client.go            # eino ChatModel + Embedder 封装（OpenAI 兼容协议）
├── agent/agent.go           # 带记忆的 Agent，用 compose.Chain 编排
├── memory/
│   ├── memory.go            # Strategy 接口与公共工具
│   ├── basic.go             # 1 顺序记忆 / 2 滑动窗口
│   ├── summarization.go     # 3 摘要记忆
│   ├── vectorstore.go       # 内存向量库（实现 eino Indexer + Retriever）
│   ├── retrieval.go         # 4 检索式记忆 (RAG)
│   ├── augmented.go         # 5 记忆增强 / 6 层次化记忆
│   ├── graph.go             # 7 图记忆
│   └── compression_os.go    # 8 压缩记忆 / 9 类 OS 记忆
└── cmd/demo/main.go         # 演示各策略的对话效果
```

## 快速开始

```bash
cd eino-memory

# 必填：API Key（不要写死在代码里）
export OPENAI_API_KEY=sk-xxx          # Windows PowerShell: $env:OPENAI_API_KEY="sk-xxx"
# 可选：默认是 SiliconFlow + DeepSeek-V4-Flash + Qwen3-Embedding-8B
export OPENAI_BASE_URL=https://api.siliconflow.cn/v1
export CHAT_MODEL=deepseek-ai/DeepSeek-V4-Flash
export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
# 可选：向量服务单独配置（DeepSeek 官方没有 embedding 接口时用）
export EMBEDDING_API_KEY=sk-yyy
export EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
# 可选：开启思考模式（默认关闭，记忆策略用不到推理，关掉更快更省）
export THINKING=on
# 可选：本机开了 Clash 等代理时要显式设置，Go 不会读 Windows 系统代理
export HTTPS_PROXY=http://127.0.0.1:7897

go run ./cmd/demo -s sliding     # 运行单个策略
go run ./cmd/demo -s all         # 依次运行全部 9 种
go test ./...                    # 单元测试（mock LLM，不需要 API Key）
```

### 可用模型（2026-10 通过两家 `/models` 接口核实）

| 服务商 | BaseURL | 对话模型 | 向量模型 |
| --- | --- | --- | --- |
| SiliconFlow | `https://api.siliconflow.cn/v1` | `deepseek-ai/DeepSeek-V4-Flash`（默认）、`deepseek-ai/DeepSeek-V4-Pro`、`deepseek-ai/DeepSeek-V3.2` | `Qwen/Qwen3-Embedding-8B`（4096 维，默认）、`Qwen/Qwen3-Embedding-4B`、`BAAI/bge-m3` |
| DeepSeek 官方 | `https://api.deepseek.com` | `deepseek-flash`（V4.1-Flash）、`deepseek-v4-pro` | 无，需配合 `EMBEDDING_*` 走 SiliconFlow |

DeepSeek 官方已宣布停用旧名 `deepseek-chat` / `deepseek-reasoner`（目前调用仍能返回，但不建议再用）。

可选的 `-s` 值：`sequential` `sliding` `summary` `retrieval` `augmented` `hierarchical` `graph` `compression` `os` `all`

## 用到的 eino 组件

| 用途 | 组件 |
| --- | --- |
| 对话模型 | `eino-ext/components/model/openai` → `model.BaseChatModel` |
| 向量化 | `eino-ext/components/embedding/openai` → `embedding.Embedder` |
| 编排 | `compose.Chain`：`Lambda(读记忆) → ChatTemplate → ChatModel` |
| 向量库 | `InMemoryVectorStore`，实现了 `indexer.Indexer` 和 `retriever.Retriever` |
| 知识图谱 | 邻接表 + set 去重 |
| Token 统计 | 读取 `msg.ResponseMeta.Usage` 中模型返回的真实用量 |

Agent 的编排链路：

```
query ─▶ [Lambda load_memory] ─▶ [ChatTemplate prompt] ─▶ [ChatModel] ─▶ reply
              │                                                         │
              └──── Strategy.GetContext                Strategy.AddMessage ◀┘
```

因为走的是 eino 编排，可以直接挂 `callbacks`（日志、链路追踪、Langfuse 等），也可以在链上继续追加 Tools、Retriever 等节点。

## 扩展

- 换模型：`llm.Client` 只依赖 `model.BaseChatModel` / `embedding.Embedder`，把 `Chat` 或 `Embedder` 换成 `eino-ext` 里的 ark、ollama、qwen、deepseek 等实现即可。
- 换向量库：`RetrievalMemory` 内部用的是 eino 标准的 Indexer/Retriever 接口，生产环境可以换成 `eino-ext` 的 Milvus、Redis、ES8。
- 自定义策略：实现 `memory.Strategy` 接口（`Name / AddMessage / GetContext / Clear`），然后传给 `agent.New`。

## 实现要点

- 图记忆：要求 LLM 输出 JSON 数组，解析失败时用元组正则兜底；实体链接用子串匹配，适配中文没有空格分词的情况。
- 类 OS 记忆：关键词匹配之外加了中文双字（bigram）匹配，否则纯中文问题难以触发缺页。
- 向量检索：用余弦相似度衡量相似度。
- 并发：每个策略都带锁，`Agent.Chat` 串行执行，保证"读记忆 → 生成 → 写记忆"不会交错。

## 策略对比

| 策略 | 逻辑描述 | Go 实现（类型 / 文件） | demo 参数 | 复杂度 | 成本 | 召回质量 | 适用场景 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sequential | 保存全部历史，每轮把完整对话塞进上下文 | `SequentialMemory` / [memory/basic.go](memory/basic.go) | `-s sequential` | 很低 | 很高 | 完美（直到超限） | Demo、短对话 |
| Sliding Window | 只保留最近 N 轮，超出窗口丢弃最旧一轮 | `SlidingWindowMemory` / [memory/basic.go](memory/basic.go) | `-s sliding` | 低 | 低 | 差（旧信息） | 通用聊天 |
| Summarization | 消息先进缓冲区，达到阈值后由 LLM 把“旧摘要 + 缓冲区”合并成新摘要并清空缓冲 | `SummarizationMemory` / [memory/summarization.go](memory/summarization.go) | `-s summary` | 中 | 中 | 好（可能丢细节） | 长对话、创作 |
| Retrieval (RAG) | 用户输入和 AI 回复分别向量化入库，查询时按余弦相似度取 TopK | `RetrievalMemory` / [memory/retrieval.go](memory/retrieval.go)（向量库：[memory/vectorstore.go](memory/vectorstore.go)） | `-s retrieval` | 高 | 低 | 优（具体细节） | 问答、知识型 Agent |
| Memory-Augmented | 滑动窗口 + 长期便签；每轮让 LLM 判断是否有关键事实，有则提炼成一句话永久保存 | `MemoryAugmentedMemory` / [memory/augmented.go](memory/augmented.go) | `-s augmented` | 高 | 高 | 优（关键事实） | 个人助理 |
| Hierarchical | 所有对话进工作记忆（滑动窗口）；含“记住/偏好/过敏”等关键词的再提升到长期检索记忆，查询时两层合并 | `HierarchicalMemory` / [memory/augmented.go](memory/augmented.go) | `-s hierarchical` | 高 | 中高 | 很好（分层） | 多任务 Agent |
| Graph | LLM 抽取 (主语, 关系, 宾语) 三元组写入有向图；查询时找出提及的实体，返回其入边和出边 | `GraphMemory` / [memory/graph.go](memory/graph.go) | `-s graph` | 很高 | 高 | 优（关系推理） | 知识库 |
| Compression | 每轮由 LLM 压缩成一条精简事实，上下文只放事实列表 | `CompressionMemory` / [memory/compression_os.go](memory/compression_os.go) | `-s compression` | 中 | 低 | 好（只保留事实） | 会议纪要、任务型 |
| OS-Like | RAM 存最近 N 轮，满了把最旧一轮换出到 Disk；查询命中 Disk 内容（关键词/中文双字）时缺页换入 | `OSMemory` / [memory/compression_os.go](memory/compression_os.go) | `-s os` | 很高 | 低中 | 很好（换页准确时） | 长期运行 Agent |

路径都相对于 `eino-memory/` 目录。各策略的演示对话在 [cmd/demo/main.go](cmd/demo/main.go) 的 `scenarios` 里，离线测试在 [memory/memory_test.go](memory/memory_test.go)。

## 关于作者

- **GitHub**：[T-DWAG](https://github.com/T-DWAG)
- **邮箱**：wuonchannel@gmail.com
- **地点**：成都（Chengdu）
- **状态**：在职，寻找合适的机会（open to suitable opportunities）
