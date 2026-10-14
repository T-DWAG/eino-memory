// Package memory 实现 9 种 AI Agent 记忆优化策略。
//
// 所有策略都实现同一个 Strategy 接口，Agent 只依赖接口，
// 因此可以"即插即用"地切换记忆策略。
package memory

import (
	"context"
	"fmt"
	"strings"
)

// Strategy 是所有记忆策略的统一接口。
type Strategy interface {
	// Name 返回策略名称，用于日志输出。
	Name() string
	// AddMessage 把一轮完整对话（用户输入 + AI 回复）写入记忆。
	AddMessage(ctx context.Context, userInput, aiResponse string) error
	// GetContext 根据当前用户查询生成要注入 Prompt 的上下文。
	// 部分策略（检索、图、OS 换页）会利用 query 做相关性查找。
	GetContext(ctx context.Context, query string) (string, error)
	// Clear 清空记忆。
	Clear()
}

// Message 表示一条对话消息。
type Message struct {
	Role    string // "user" 或 "assistant"
	Content string
}

// Turn 表示一轮对话（用户 + 助手各一条）。
type Turn struct {
	User      string
	Assistant string
}

// formatMessages 把消息列表格式化为 "User: xxx\nAssistant: yyy" 形式。
func formatMessages(msgs []Message) string {
	var sb strings.Builder
	for i, m := range msgs {
		if i > 0 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "%s: %s", capitalize(m.Role), m.Content)
	}
	return sb.String()
}

// formatTurn 把一轮对话格式化为统一文本，供压缩、图抽取、OS 换页等策略使用。
func formatTurn(userInput, aiResponse string) string {
	return fmt.Sprintf("用户: %s\nAI: %s", userInput, aiResponse)
}

// capitalize 首字母大写。
func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// logf 统一的调试输出。
func logf(format string, args ...any) {
	fmt.Printf(format+"\n", args...)
}
