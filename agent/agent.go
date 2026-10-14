// Package agent 实现可插拔记忆策略的对话 Agent。
//
// 用 eino 的 compose.Chain 把一次对话编排成：
//
//	用户输入(string)
//	   │
//	   ▼
//	[Lambda] loadMemory   —— 调用 memory.Strategy.GetContext 取出记忆上下文
//	   │  map[string]any{"system","context","query"}
//	   ▼
//	[ChatTemplate]        —— 用 FString 模板拼出 system + user 消息
//	   │  []*schema.Message
//	   ▼
//	[ChatModel]           —— 调用大模型
//	   │  *schema.Message
//	   ▼
//	回复 → memory.Strategy.AddMessage 写回记忆
//
// 好处：编排后的 Runnable 天然支持 eino 的 callbacks（链路追踪、日志、
// Langfuse/APMPlus 等），也方便后续在链上插入 Retriever、Tools 等节点。
package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	"github.com/T-DWAG/eino-memory/memory"
)

// DefaultSystemPrompt 默认系统提示词。
const DefaultSystemPrompt = "You are a helpful AI assistant."

// userTemplate 用户消息模板。
// 注意：FString 模板中 {xxx} 是变量占位符；变量的"值"里出现花括号不会有问题。
const userTemplate = "### MEMORY CONTEXT\n{context}\n\n### CURRENT REQUEST\n{query}"

// Agent 带记忆的对话 Agent。
type Agent struct {
	mu           sync.Mutex // 串行化 Chat，保证"读记忆→生成→写记忆"的原子性
	memory       memory.Strategy
	systemPrompt string
	runnable     compose.Runnable[string, *schema.Message]
}

// Option Agent 的可选配置。
type Option func(*Agent)

// WithSystemPrompt 自定义系统提示词。
func WithSystemPrompt(p string) Option { return func(a *Agent) { a.systemPrompt = p } }

// New 创建 Agent，并编译 eino Chain。
func New(ctx context.Context, chatModel model.BaseChatModel, mem memory.Strategy, opts ...Option) (*Agent, error) {
	a := &Agent{memory: mem, systemPrompt: DefaultSystemPrompt}
	for _, o := range opts {
		o(a)
	}

	// 节点 1：读取记忆，输出模板变量
	loadMemory := compose.InvokableLambda(func(ctx context.Context, query string) (map[string]any, error) {
		start := time.Now()
		memCtx, err := a.memory.GetContext(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("读取记忆失败: %w", err)
		}
		fmt.Printf("\n--- Agent 调试信息 ---\n记忆检索时间: %.4f 秒\n", time.Since(start).Seconds())
		return map[string]any{
			"system":  a.systemPrompt,
			"context": memCtx,
			"query":   query,
		}, nil
	})

	// 节点 2：Prompt 模板。系统提示词也作为变量传入，避免其中的花括号被当成占位符。
	tpl := prompt.FromMessages(schema.FString,
		schema.SystemMessage("{system}"),
		schema.UserMessage(userTemplate),
	)

	chain := compose.NewChain[string, *schema.Message]()
	chain.
		AppendLambda(loadMemory, compose.WithNodeName("load_memory")).
		AppendChatTemplate(tpl, compose.WithNodeName("prompt")).
		AppendChatModel(chatModel, compose.WithNodeName("chat_model"))

	r, err := chain.Compile(ctx, compose.WithGraphName("memory_agent"))
	if err != nil {
		return nil, fmt.Errorf("编译 eino chain 失败: %w", err)
	}
	a.runnable = r
	fmt.Printf("Agent 已初始化，使用 %s\n", mem.Name())
	return a, nil
}

// Memory 返回当前使用的记忆策略（便于外部 Clear 等操作）。
func (a *Agent) Memory() memory.Strategy { return a.memory }

// Chat 处理单轮对话：读记忆 → 生成 → 写记忆，并返回回复内容。
func (a *Agent) Chat(ctx context.Context, userInput string) (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	bar := strings.Repeat("=", 25)
	fmt.Printf("\n%s 新对话 %s\n用户 > %s\n", bar, bar, userInput)

	start := time.Now()
	msg, err := a.runnable.Invoke(ctx, userInput)
	if err != nil {
		return "", err
	}
	genTime := time.Since(start)

	// 使用模型返回的真实 token 用量
	if msg.ResponseMeta != nil && msg.ResponseMeta.Usage != nil {
		u := msg.ResponseMeta.Usage
		fmt.Printf("Prompt Tokens: %d, Completion Tokens: %d\n", u.PromptTokens, u.CompletionTokens)
	}

	// 写回记忆。失败不影响本轮回复，只打印警告。
	if err := a.memory.AddMessage(ctx, userInput, msg.Content); err != nil {
		fmt.Printf("[警告] 写入记忆失败: %v\n", err)
	}

	fmt.Printf("\nAgent > %s\n(总耗时: %.4f 秒)\n", msg.Content, genTime.Seconds())
	return msg.Content, nil
}
