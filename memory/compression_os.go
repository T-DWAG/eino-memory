package memory

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

// ============================================================================
// 策略 8：压缩记忆（Compression & Consolidation Memory）
// ----------------------------------------------------------------------------
// 思路：每轮对话后让 LLM 充当"数据压缩器"，把整轮对话提炼成一条精简事实。
// 优点：极其省 Token，事实召回好。
// 缺点：丢失语气、风格等对话细节。
// 适用：会议纪要、任务规划等只关心事实的场景。
// ============================================================================

// CompressionMemory 只保存压缩后的事实列表。
type CompressionMemory struct {
	mu    sync.RWMutex
	llm   TextGenerator
	facts []string
}

// NewCompressionMemory 创建压缩记忆。
func NewCompressionMemory(llm TextGenerator) *CompressionMemory {
	return &CompressionMemory{llm: llm}
}

func (m *CompressionMemory) Name() string { return "CompressionMemory" }

// AddMessage 调用 LLM 压缩本轮对话并保存。
func (m *CompressionMemory) AddMessage(ctx context.Context, userInput, aiResponse string) error {
	prompt := fmt.Sprintf(
		"你是数据压缩引擎。你的任务是将以下文本提炼成其最基本的事实陈述。"+
			"尽可能简洁,删除所有对话性的冗余内容。例如,'用户询问我的名字,我(AI)回复说我的名字是AI助手'应该变成'用户询问AI的名字'。\n\n"+
			"要压缩的文本:\n\"\"\"%s\"\"\"",
		formatTurn(userInput, aiResponse),
	)
	fact, err := m.llm.GenerateText(ctx, "你是专业的数据压缩器。", prompt)
	if err != nil {
		return fmt.Errorf("压缩失败: %w", err)
	}
	fact = strings.TrimSpace(fact)
	logf("--- [压缩记忆: 存储新事实: '%s'] ---", fact)
	m.mu.Lock()
	m.facts = append(m.facts, fact)
	m.mu.Unlock()
	return nil
}

// GetContext 返回所有压缩事实。
func (m *CompressionMemory) GetContext(_ context.Context, _ string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.facts) == 0 {
		return "记忆中没有压缩事实。", nil
	}
	return "### 压缩事实记忆:\n- " + strings.Join(m.facts, "\n- "), nil
}

func (m *CompressionMemory) Clear() {
	m.mu.Lock()
	m.facts = nil
	m.mu.Unlock()
	logf("压缩记忆已清除")
}

// ============================================================================
// 策略 9：类 OS 记忆管理（OS-Like Memory）
// ----------------------------------------------------------------------------
// 思路：模拟操作系统内存分层
//   - 活动记忆（RAM）：最近 ramSize 轮，始终放入上下文；
//   - 被动记忆（Disk）：RAM 满时把最旧的一轮"换出"到这里；
//   - 缺页（Page Fault）：query 与被动记忆中的内容相关时，把它"换入"上下文。
//
// 这里用关键词匹配模拟相关性判断。由于中文没有空格分词，
// 额外加入了中文双字（bigram）匹配；生产环境建议换成向量检索
// （可直接复用 InMemoryVectorStore 或 eino-ext 的 Retriever）。
//
// 优点：活动上下文很小，理论上记忆无限。
// 缺点：两层管理复杂，换入策略不准会漏召回。
// 适用：跨天/跨周的长期运行 Agent。
// ============================================================================

type pagedTurn struct {
	id   int
	data string
}

// OSMemory 两级（RAM / Disk）记忆。
type OSMemory struct {
	mu        sync.Mutex
	ramSize   int
	active    []pagedTurn    // RAM，按时间顺序
	passive   map[int]string // Disk，turnID -> 文本
	turnCount int
}

// NewOSMemory 创建类 OS 记忆，ramSize 为 RAM 中保留的轮次数。
func NewOSMemory(ramSize int) *OSMemory {
	if ramSize <= 0 {
		ramSize = 2
	}
	return &OSMemory{ramSize: ramSize, passive: map[int]string{}}
}

func (m *OSMemory) Name() string { return "OSMemory" }

// AddMessage 写入 RAM；RAM 已满则先将最旧一轮换出到 Disk（LRU）。
func (m *OSMemory) AddMessage(_ context.Context, userInput, aiResponse string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.active) >= m.ramSize {
		lru := m.active[0]
		m.active = m.active[1:]
		m.passive[lru.id] = lru.data
		logf("--- [OS记忆: 将轮次%d换出到被动存储。] ---", lru.id)
	}
	m.active = append(m.active, pagedTurn{id: m.turnCount, data: formatTurn(userInput, aiResponse)})
	m.turnCount++
	return nil
}

// GetContext 返回 RAM 内容，并对命中的被动记忆执行"换入"。
func (m *OSMemory) GetContext(_ context.Context, query string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	activeParts := make([]string, len(m.active))
	for i, t := range m.active {
		activeParts[i] = t.data
	}

	// 按轮次 ID 排序，保证换入顺序稳定（Go map 遍历无序）
	ids := make([]int, 0, len(m.passive))
	for id := range m.passive {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	terms := queryTerms(query)
	var paged strings.Builder
	for _, id := range ids {
		data := strings.ToLower(m.passive[id])
		for _, t := range terms {
			if strings.Contains(data, t) {
				fmt.Fprintf(&paged, "\n(从轮次%d换入): %s", id, m.passive[id])
				logf("--- [OS记忆: 缺页!从被动存储换入轮次%d。] ---", id)
				break
			}
		}
	}
	return fmt.Sprintf("### 活动记忆(RAM):\n%s\n\n### 从被动记忆换入(磁盘):\n%s",
		strings.Join(activeParts, "\n"), paged.String()), nil
}

func (m *OSMemory) Clear() {
	m.mu.Lock()
	m.active, m.passive, m.turnCount = nil, map[int]string{}, 0
	m.mu.Unlock()
	logf("类OS记忆已清除")
}

// queryTerms 从 query 中提取用于匹配的关键词：
//   - 按空白切分后长度 > 3 的英文词；
//   - 中文字符的相邻双字组合（bigram），弥补中文无空格分词的问题。
func queryTerms(query string) []string {
	q := strings.ToLower(query)
	var terms []string
	for _, w := range strings.Fields(q) {
		if utf8.RuneCountInString(w) > 3 && isASCII(w) {
			terms = append(terms, w)
		}
	}
	var han []rune
	flush := func() {
		for i := 0; i+1 < len(han); i++ {
			terms = append(terms, string(han[i:i+2]))
		}
		han = han[:0]
	}
	for _, r := range q {
		if r >= 0x4E00 && r <= 0x9FFF { // CJK 统一表意文字
			han = append(han, r)
		} else {
			flush()
		}
	}
	flush()
	return terms
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] > 0x7F {
			return false
		}
	}
	return true
}
