// Package llm 封装 eino 的 ChatModel 与 Embedder。
//
// 这里使用 eino-ext 的 OpenAI 兼容实现，因此可以对接 OpenAI、SiliconFlow、
// DeepSeek、通义等任意 OpenAI 协议兼容的服务，只需修改 BaseURL / Model。
package llm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	embopenai "github.com/cloudwego/eino-ext/components/embedding/openai"
	chatopenai "github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

// 默认配置：SiliconFlow + DeepSeek-V4-Flash + Qwen3-Embedding-8B。
// 模型名已于 2026-10 通过两家的 /models 接口核实：
//   - SiliconFlow 对话：deepseek-ai/DeepSeek-V4-Flash（更强：deepseek-ai/DeepSeek-V4-Pro；
//     deepseek-ai/DeepSeek-V3.2 仍可用）
//   - DeepSeek 官方：BaseURL https://api.deepseek.com，模型 deepseek-flash / deepseek-v4-pro
//     （旧名 deepseek-chat / deepseek-reasoner 官方已宣布停用）
//   - 向量：Qwen/Qwen3-Embedding-8B（4096 维），DeepSeek 官方没有 embedding 接口
const (
	DefaultBaseURL        = "https://api.siliconflow.cn/v1"
	DefaultChatModel      = "deepseek-ai/DeepSeek-V4-Flash"
	DefaultEmbeddingModel = "Qwen/Qwen3-Embedding-8B"
)

// Config 描述连接大模型服务所需的参数。
type Config struct {
	APIKey         string // API Key（必填，建议通过环境变量传入，切勿硬编码进代码）
	BaseURL        string // OpenAI 兼容服务地址
	ChatModel      string // 对话模型名称
	EmbeddingModel string // 向量模型名称
	// 向量服务可以单独配置（例如对话用 DeepSeek 官方，向量用 SiliconFlow）。
	// 为空时沿用 APIKey / BaseURL。
	EmbeddingAPIKey  string
	EmbeddingBaseURL string
	// Thinking 是否开启思考模式。DeepSeek V4 系列默认开启，
	// 但记忆策略里的摘要/抽取/压缩不需要推理，关掉更快更省。默认 false。
	Thinking bool
	Timeout  time.Duration // 单次请求超时
}

// ConfigFromEnv 从环境变量读取配置，未设置的项使用默认值。
//
//	OPENAI_API_KEY     必填
//	OPENAI_BASE_URL    可选，默认 SiliconFlow
//	CHAT_MODEL         可选
//	EMBEDDING_MODEL    可选
//	EMBEDDING_API_KEY  可选，向量服务单独的 Key，默认同 OPENAI_API_KEY
//	EMBEDDING_BASE_URL 可选，向量服务单独的地址，默认同 OPENAI_BASE_URL
//	THINKING           可选，设为 on / true / 1 开启思考模式，默认关闭
func ConfigFromEnv() Config {
	apiKey := os.Getenv("OPENAI_API_KEY")
	baseURL := getenv("OPENAI_BASE_URL", DefaultBaseURL)
	return Config{
		APIKey:           apiKey,
		BaseURL:          baseURL,
		ChatModel:        getenv("CHAT_MODEL", DefaultChatModel),
		EmbeddingModel:   getenv("EMBEDDING_MODEL", DefaultEmbeddingModel),
		EmbeddingAPIKey:  getenv("EMBEDDING_API_KEY", apiKey),
		EmbeddingBaseURL: getenv("EMBEDDING_BASE_URL", baseURL),
		Thinking:         isTruthy(os.Getenv("THINKING")),
		Timeout:          120 * time.Second,
	}
}

func isTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "on", "yes":
		return true
	}
	return false
}

// thinkingFields 按服务商生成开关思考模式的请求字段（两家字段名不同），
// 通过 eino-ext openai 的 ExtraFields 合并进请求体。其他服务商返回 nil，不添加字段。
//   - DeepSeek 官方：{"thinking": {"type": "enabled" | "disabled"}}
//   - SiliconFlow：  {"enable_thinking": true | false}
func thinkingFields(baseURL string, on bool) map[string]any {
	switch u := strings.ToLower(baseURL); {
	case strings.Contains(u, "api.deepseek.com"):
		t := "disabled"
		if on {
			t = "enabled"
		}
		return map[string]any{"thinking": map[string]any{"type": t}}
	case strings.Contains(u, "siliconflow"):
		return map[string]any{"enable_thinking": on}
	}
	return nil
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// Client 同时持有 eino 的对话模型和向量模型。
// 记忆策略只依赖这个结构体，方便替换为 Ark、Ollama 等其他 eino 组件。
type Client struct {
	Chat     model.BaseChatModel // eino 标准对话模型接口
	Embedder embedding.Embedder  // eino 标准向量化接口
}

// NewClient 基于 eino-ext 的 OpenAI 组件创建 Client。
func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("缺少 API Key，请设置环境变量 OPENAI_API_KEY")
	}

	chat, err := chatopenai.NewChatModel(ctx, &chatopenai.ChatModelConfig{
		APIKey:  cfg.APIKey,
		BaseURL: cfg.BaseURL,
		Model:   cfg.ChatModel,
		Timeout: cfg.Timeout,
		// 按服务商附加思考模式开关字段
		ExtraFields: thinkingFields(cfg.BaseURL, cfg.Thinking),
	})
	if err != nil {
		return nil, fmt.Errorf("创建 ChatModel 失败: %w", err)
	}

	// 向量服务未单独配置时沿用对话服务的配置
	embKey, embURL := cfg.EmbeddingAPIKey, cfg.EmbeddingBaseURL
	if embKey == "" {
		embKey = cfg.APIKey
	}
	if embURL == "" {
		embURL = cfg.BaseURL
	}
	emb, err := embopenai.NewEmbedder(ctx, &embopenai.EmbeddingConfig{
		APIKey:  embKey,
		BaseURL: embURL,
		Model:   cfg.EmbeddingModel,
		Timeout: cfg.Timeout,
	})
	if err != nil {
		return nil, fmt.Errorf("创建 Embedder 失败: %w", err)
	}

	return &Client{Chat: chat, Embedder: emb}, nil
}

// GenerateText 发送 system + user 两条消息并返回模型回复文本。
// 记忆策略内部（摘要、事实抽取、压缩等）会用到。
func (c *Client) GenerateText(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	msg, err := c.Chat.Generate(ctx, []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	})
	if err != nil {
		return "", fmt.Errorf("文本生成失败: %w", err)
	}
	return msg.Content, nil
}

// Embed 将单条文本转换为向量。
func (c *Client) Embed(ctx context.Context, text string) ([]float64, error) {
	vecs, err := c.Embedder.EmbedStrings(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("embedding 生成失败: %w", err)
	}
	if len(vecs) == 0 {
		return nil, errors.New("embedding 返回为空")
	}
	return vecs[0], nil
}
