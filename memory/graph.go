package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// ============================================================================
// 策略 7：图记忆（Graph-Based Memory）
// ----------------------------------------------------------------------------
// 思路：
//  1. 每轮对话后让 LLM 抽取 (主语, 关系, 宾语) 三元组；
//  2. 写入有向知识图谱（这里用邻接表实现）；
//  3. 查询时找出 query 中出现的实体，返回其所有入边/出边作为上下文。
//
// 实现说明：
//   - 要求 LLM 输出 JSON 数组，解析失败时用元组正则兜底；
//   - 实体链接采用"节点名是否出现在 query 中"的子串匹配，
//     对中文（无空格分词）更友好。
//
// 优点：擅长关系推理、多跳问题。
// 缺点：实现复杂，抽取质量依赖 LLM。
// 适用：知识库、专家系统、推荐。
// ============================================================================

// Triple 知识三元组。
type Triple struct {
	Subject  string `json:"subject"`
	Relation string `json:"relation"`
	Object   string `json:"object"`
}

// edge 图中的一条有向边。
type edge struct {
	from, to, relation string
}

// GraphMemory 简单的内存有向图。
type GraphMemory struct {
	mu    sync.RWMutex
	llm   TextGenerator
	nodes map[string]struct{}
	edges map[edge]struct{} // 用 set 去重
}

// NewGraphMemory 创建图记忆。
func NewGraphMemory(llm TextGenerator) *GraphMemory {
	return &GraphMemory{llm: llm, nodes: map[string]struct{}{}, edges: map[edge]struct{}{}}
}

func (m *GraphMemory) Name() string { return "GraphMemory" }

var (
	// jsonArrayRe 从 LLM 回复中截取第一个 JSON 数组（兼容 ```json 代码块包裹）。
	jsonArrayRe = regexp.MustCompile(`(?s)\[.*\]`)
	// tupleRe 兜底：匹配 ('a', 'b', 'c') 形式的元组。
	tupleRe = regexp.MustCompile(`\(['"](.*?)['"],\s*['"](.*?)['"],\s*['"](.*?)['"]\)`)
)

// extractTriples 调用 LLM 抽取三元组并解析。
func (m *GraphMemory) extractTriples(ctx context.Context, text string) ([]Triple, error) {
	logf("--- [图记忆: 尝试从文本中提取三元组。] ---")
	prompt := fmt.Sprintf(
		"你是知识提取引擎。你的任务是从给定文本中提取主语-关系-宾语三元组。"+
			"严格只输出 JSON 数组,不要输出其他内容。格式示例:"+
			`[{"subject":"Sam","relation":"works_for","object":"Innovatech"}]。`+
			"实体名请保持原文写法。如果没有找到三元组,返回空数组[]。\n\n要分析的文本:\n\"\"\"%s\"\"\"",
		text,
	)
	resp, err := m.llm.GenerateText(ctx, "你是专业的知识图谱提取器。", prompt)
	if err != nil {
		return nil, err
	}

	var triples []Triple
	if raw := jsonArrayRe.FindString(resp); raw != "" {
		if err := json.Unmarshal([]byte(raw), &triples); err == nil {
			logf("--- [图记忆: 提取的三元组: %v] ---", triples)
			return triples, nil
		}
	}
	// JSON 解析失败时尝试元组格式
	for _, g := range tupleRe.FindAllStringSubmatch(resp, -1) {
		triples = append(triples, Triple{Subject: g[1], Relation: g[2], Object: g[3]})
	}
	logf("--- [图记忆: 提取的三元组: %v] ---", triples)
	return triples, nil
}

// AddMessage 抽取三元组并写入图。
func (m *GraphMemory) AddMessage(ctx context.Context, userInput, aiResponse string) error {
	triples, err := m.extractTriples(ctx, formatTurn(userInput, aiResponse))
	if err != nil {
		return fmt.Errorf("三元组抽取失败: %w", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, t := range triples {
		s, r, o := strings.TrimSpace(t.Subject), strings.TrimSpace(t.Relation), strings.TrimSpace(t.Object)
		if s == "" || r == "" || o == "" {
			continue
		}
		m.nodes[s] = struct{}{}
		m.nodes[o] = struct{}{}
		m.edges[edge{from: s, to: o, relation: r}] = struct{}{}
	}
	return nil
}

// GetContext 找到 query 中提及的实体，返回与之相连的所有关系。
func (m *GraphMemory) GetContext(_ context.Context, query string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.nodes) == 0 {
		return "知识图谱为空。", nil
	}

	// 实体链接：节点名（忽略大小写）出现在 query 中即视为命中
	lq := strings.ToLower(query)
	hit := map[string]bool{}
	for n := range m.nodes {
		if strings.Contains(lq, strings.ToLower(n)) {
			hit[n] = true
		}
	}
	if len(hit) == 0 {
		return "在知识图谱中未找到查询中的相关实体。", nil
	}

	// 收集命中实体的出边与入边
	set := map[string]struct{}{}
	for e := range m.edges {
		if hit[e.from] || hit[e.to] {
			set[fmt.Sprintf("%s --[%s]--> %s", e.from, e.relation, e.to)] = struct{}{}
		}
	}
	facts := make([]string, 0, len(set))
	for f := range set {
		facts = append(facts, f)
	}
	sort.Strings(facts) // 排序保证输出稳定
	return "### 从知识图谱检索到的事实:\n" + strings.Join(facts, "\n"), nil
}

func (m *GraphMemory) Clear() {
	m.mu.Lock()
	m.nodes, m.edges = map[string]struct{}{}, map[edge]struct{}{}
	m.mu.Unlock()
	logf("图记忆已清除")
}
