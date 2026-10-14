package memory

import (
	"context"
	"fmt"
	"sync"
)

// TextGenerator 是记忆策略调用 LLM 的最小依赖。
// llm.Client 已实现该接口；单元测试中可以用 mock 替换。
type TextGenerator interface {
	GenerateText(ctx context.Context, systemPrompt, userPrompt string) (string, error)
}

// ============================================================================
// 策略 3：摘要记忆（Summarization Memory）
// ----------------------------------------------------------------------------
// 思路：
//  1. 最近消息先放进缓冲区（buffer）；
//  2. 缓冲区达到阈值（threshold）后，调用 LLM 把"旧摘要 + 缓冲区"合并成新摘要；
//  3. 新摘要替换旧摘要，并清空缓冲区。
//
// 优点：上下文大小可控，同时保留对话要点。
// 缺点：摘要可能丢失细节；每次整合多一次 LLM 调用。
// 适用：长对话、创作类场景。
// ============================================================================

// SummarizationMemory 维护一个滚动摘要 + 最近消息缓冲区。
type SummarizationMemory struct {
	mu             sync.Mutex
	llm            TextGenerator
	threshold      int       // 触发摘要的消息条数（4 条 = 2 轮）
	runningSummary string    // 持续更新的摘要
	buffer         []Message // 尚未被摘要的最近消息
}

// NewSummarizationMemory 创建摘要记忆。threshold 为触发整合的消息条数。
func NewSummarizationMemory(llm TextGenerator, threshold int) *SummarizationMemory {
	if threshold <= 0 {
		threshold = 4
	}
	return &SummarizationMemory{llm: llm, threshold: threshold}
}

func (m *SummarizationMemory) Name() string { return "SummarizationMemory" }

// AddMessage 将消息写入缓冲区，必要时触发摘要整合。
func (m *SummarizationMemory) AddMessage(ctx context.Context, userInput, aiResponse string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.buffer = append(m.buffer,
		Message{Role: "user", Content: userInput},
		Message{Role: "assistant", Content: aiResponse},
	)
	if len(m.buffer) >= m.threshold {
		return m.consolidate(ctx)
	}
	return nil
}

// consolidate 调用 LLM 把旧摘要和缓冲区合并成新摘要（调用方需持有锁）。
func (m *SummarizationMemory) consolidate(ctx context.Context) error {
	logf("\n--- [触发记忆整合] ---")
	prompt := fmt.Sprintf(
		"你是一个摘要专家。请将'之前的摘要'与'新对话'合并成一个简洁的摘要。"+
			"捕获所有关键事实、名称和决定。\n\n"+
			"### 之前的摘要:\n%s\n\n### 新对话:\n%s\n\n### 更新后的摘要:",
		m.runningSummary, formatMessages(m.buffer),
	)
	summary, err := m.llm.GenerateText(ctx, "你是专业的摘要引擎。", prompt)
	if err != nil {
		// 失败时保留缓冲区，下次再尝试整合，避免丢数据
		return fmt.Errorf("摘要整合失败: %w", err)
	}
	m.runningSummary = summary
	m.buffer = nil
	logf("--- [新摘要: '%s'] ---", summary)
	return nil
}

// GetContext 返回 "摘要 + 最近缓冲区"。
func (m *SummarizationMemory) GetContext(_ context.Context, _ string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return fmt.Sprintf("### 过去对话的摘要:\n%s\n\n### 最近消息:\n%s",
		m.runningSummary, formatMessages(m.buffer)), nil
}

func (m *SummarizationMemory) Clear() {
	m.mu.Lock()
	m.runningSummary, m.buffer = "", nil
	m.mu.Unlock()
	logf("摘要式记忆已清除")
}
