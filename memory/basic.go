package memory

import (
	"context"
	"sync"
)

// ============================================================================
// 策略 1：顺序记忆（Sequential Memory）
// ----------------------------------------------------------------------------
// 思路：保存全部历史，每次全部塞进上下文。
// 优点：实现最简单，回忆完美。
// 缺点：Token 随轮次线性增长，成本高，最终会超出上下文窗口。
// 适用：Demo、短对话。
// ============================================================================

// SequentialMemory 保存完整对话历史。
type SequentialMemory struct {
	mu      sync.RWMutex
	history []Message
}

// NewSequentialMemory 创建顺序记忆。
func NewSequentialMemory() *SequentialMemory { return &SequentialMemory{} }

func (m *SequentialMemory) Name() string { return "SequentialMemory" }

// AddMessage 把用户和助手消息依次追加到历史末尾。
func (m *SequentialMemory) AddMessage(_ context.Context, userInput, aiResponse string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.history = append(m.history,
		Message{Role: "user", Content: userInput},
		Message{Role: "assistant", Content: aiResponse},
	)
	return nil
}

// GetContext 返回全部历史（忽略 query）。
func (m *SequentialMemory) GetContext(_ context.Context, _ string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return formatMessages(m.history), nil
}

func (m *SequentialMemory) Clear() {
	m.mu.Lock()
	m.history = nil
	m.mu.Unlock()
	logf("顺序记忆已清除")
}

// ============================================================================
// 策略 2：滑动窗口记忆（Sliding Window Memory）
// ----------------------------------------------------------------------------
// 思路：只保留最近 N 轮对话，旧的自动丢弃。
// 优点：Token 数量恒定可控。
// 缺点：窗口外的信息会被彻底遗忘。
// 适用：一般聊天机器人。
// ============================================================================

// SlidingWindowMemory 固定大小的对话窗口。
type SlidingWindowMemory struct {
	mu         sync.RWMutex
	windowSize int
	turns      []Turn
}

// NewSlidingWindowMemory 创建滑动窗口记忆，windowSize 为保留的轮次数。
func NewSlidingWindowMemory(windowSize int) *SlidingWindowMemory {
	if windowSize <= 0 {
		windowSize = 4
	}
	return &SlidingWindowMemory{windowSize: windowSize}
}

func (m *SlidingWindowMemory) Name() string { return "SlidingWindowMemory" }

// AddMessage 追加一轮对话；超出窗口时丢弃最旧的一轮。
func (m *SlidingWindowMemory) AddMessage(_ context.Context, userInput, aiResponse string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.turns = append(m.turns, Turn{User: userInput, Assistant: aiResponse})
	if len(m.turns) > m.windowSize {
		// 只保留最后 windowSize 轮
		m.turns = append([]Turn(nil), m.turns[len(m.turns)-m.windowSize:]...)
	}
	return nil
}

// GetContext 返回窗口内的对话。
func (m *SlidingWindowMemory) GetContext(_ context.Context, _ string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	msgs := make([]Message, 0, len(m.turns)*2)
	for _, t := range m.turns {
		msgs = append(msgs,
			Message{Role: "user", Content: t.User},
			Message{Role: "assistant", Content: t.Assistant},
		)
	}
	return formatMessages(msgs), nil
}

func (m *SlidingWindowMemory) Clear() {
	m.mu.Lock()
	m.turns = nil
	m.mu.Unlock()
	logf("滑动窗口记忆已清除")
}
