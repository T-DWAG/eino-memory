package memory

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/cloudwego/eino/components/embedding"
)

// ============================================================================
// 策略 5：记忆增强（Memory-Augmented Memory）
// ----------------------------------------------------------------------------
// 思路：短期滑动窗口 + 长期"便签"(memory tokens)。
// 每轮对话后让 LLM 判断是否包含值得长期记住的核心事实（偏好、决定、ID 等），
// 有则提炼成一句话存为记忆 token，永久注入上下文。
//
// 优点：关键事实不会因窗口滑动而丢失。
// 缺点：每轮多一次 LLM 调用，成本较高。
// 适用：需要可靠记住用户偏好的个人助理。
// ============================================================================

// noFactMarker 是 LLM 表示"无需记忆"时的约定回复。
const noFactMarker = "无重要事实"

// MemoryAugmentedMemory 短期窗口 + 长期事实便签。
type MemoryAugmentedMemory struct {
	mu           sync.RWMutex
	llm          TextGenerator
	recent       *SlidingWindowMemory
	memoryTokens []string
}

// NewMemoryAugmentedMemory 创建记忆增强策略，windowSize 为短期窗口轮次数。
func NewMemoryAugmentedMemory(llm TextGenerator, windowSize int) *MemoryAugmentedMemory {
	return &MemoryAugmentedMemory{llm: llm, recent: NewSlidingWindowMemory(windowSize)}
}

func (m *MemoryAugmentedMemory) Name() string { return "MemoryAugmentedMemory" }

// AddMessage 写入短期窗口，并让 LLM 抽取需要长期记住的事实。
func (m *MemoryAugmentedMemory) AddMessage(ctx context.Context, userInput, aiResponse string) error {
	_ = m.recent.AddMessage(ctx, userInput, aiResponse)

	prompt := fmt.Sprintf(
		"分析以下对话轮次。它是否包含应该长期记忆的核心事实、偏好或决定? "+
			"示例包括用户偏好('我讨厌坐飞机')、关键决定('预算是$1000')或重要事实('我的用户ID是12345')。\n\n"+
			"对话轮次:\n用户: %s\nAI: %s\n\n"+
			"如果包含这样的事实,请用一句话简洁地陈述该事实。否则,回复'%s'。",
		userInput, aiResponse, noFactMarker,
	)
	fact, err := m.llm.GenerateText(ctx, "你是事实提取专家。", prompt)
	if err != nil {
		return fmt.Errorf("事实抽取失败: %w", err)
	}
	fact = strings.TrimSpace(fact)
	if fact == "" || strings.Contains(fact, noFactMarker) {
		return nil
	}
	logf("--- [记忆增强: 创建新记忆token: '%s'] ---", fact)
	m.mu.Lock()
	m.memoryTokens = append(m.memoryTokens, fact)
	m.mu.Unlock()
	return nil
}

// GetContext 组合长期事实与最近对话。
func (m *MemoryAugmentedMemory) GetContext(ctx context.Context, query string) (string, error) {
	recent, _ := m.recent.GetContext(ctx, query)
	m.mu.RLock()
	lines := make([]string, len(m.memoryTokens))
	for i, t := range m.memoryTokens {
		lines[i] = "- " + t
	}
	m.mu.RUnlock()
	return fmt.Sprintf("### 关键记忆Token(长期事实):\n%s\n\n### 最近对话:\n%s",
		strings.Join(lines, "\n"), recent), nil
}

func (m *MemoryAugmentedMemory) Clear() {
	m.recent.Clear()
	m.mu.Lock()
	m.memoryTokens = nil
	m.mu.Unlock()
	logf("记忆增强记忆已清除")
}

// ============================================================================
// 策略 6：层次化记忆（Hierarchical Memory）
// ----------------------------------------------------------------------------
// 思路：两层存储
//   - 工作记忆：滑动窗口，保存所有最近对话（快）；
//   - 长期记忆：向量检索，仅保存命中"提升关键词"的重要对话（持久）。
//
// 查询时同时拼接两层内容。
//
// 优点：兼顾近期连贯性与长期关键信息。
// 缺点：提升规则是启发式的，可能漏掉未带关键词的重要信息
//
//	（可改为用 LLM 判断，参考 MemoryAugmentedMemory）。
//
// 适用：复杂多任务 Agent。
// ============================================================================

// DefaultPromotionKeywords 默认的长期记忆提升关键词。
var DefaultPromotionKeywords = []string{"记住", "规则", "偏好", "总是", "从不", "过敏"}

// HierarchicalMemory 工作记忆 + 长期检索记忆。
type HierarchicalMemory struct {
	working           *SlidingWindowMemory
	longTerm          *RetrievalMemory
	promotionKeywords []string
}

// NewHierarchicalMemory 创建层次化记忆。
// windowSize：工作记忆轮次数；k：长期记忆检索条数；keywords 为空时使用默认关键词。
func NewHierarchicalMemory(embedder embedding.Embedder, windowSize, k int, keywords []string) *HierarchicalMemory {
	if len(keywords) == 0 {
		keywords = DefaultPromotionKeywords
	}
	return &HierarchicalMemory{
		working:           NewSlidingWindowMemory(windowSize),
		longTerm:          NewRetrievalMemory(embedder, k),
		promotionKeywords: keywords,
	}
}

func (m *HierarchicalMemory) Name() string { return "HierarchicalMemory" }

// AddMessage 所有对话进入工作记忆；含关键词的额外提升到长期记忆。
func (m *HierarchicalMemory) AddMessage(ctx context.Context, userInput, aiResponse string) error {
	_ = m.working.AddMessage(ctx, userInput, aiResponse)
	lower := strings.ToLower(userInput)
	for _, kw := range m.promotionKeywords {
		if strings.Contains(lower, strings.ToLower(kw)) {
			logf("--- [层次化记忆: 将消息提升到长期存储。] ---")
			return m.longTerm.AddMessage(ctx, userInput, aiResponse)
		}
	}
	return nil
}

// GetContext 拼接长期检索结果与最近工作记忆。
func (m *HierarchicalMemory) GetContext(ctx context.Context, query string) (string, error) {
	working, _ := m.working.GetContext(ctx, query)
	longTerm, err := m.longTerm.GetContext(ctx, query)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("### 检索到的长期记忆:\n%s\n\n### 最近对话(工作记忆):\n%s", longTerm, working), nil
}

func (m *HierarchicalMemory) Clear() {
	m.working.Clear()
	m.longTerm.Clear()
	logf("层次化记忆已清除")
}
