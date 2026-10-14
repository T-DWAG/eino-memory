package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/schema"
)

// ============================================================================
// 策略 4：检索式记忆（Retrieval / RAG Memory）
// ----------------------------------------------------------------------------
// 思路：
//  1. 每轮对话的"用户输入"和"AI 回复"分别向量化存入向量库（细粒度检索）；
//  2. 新问题到来时，向量化 query，检索语义最相近的 TopK 条记录作为上下文。
//
// eino 结合点：向量库实现了 eino 的 Indexer / Retriever 接口，
// 换成 eino-ext 的 Milvus、Redis、ES8 时本策略代码无需修改。
//
// 优点：只注入相关内容，Token 少，对具体细节召回好。
// 缺点：依赖 embedding 质量；需要额外的向量化调用。
// 适用：问答、知识型 Agent。
// ============================================================================

// RetrievalMemory 基于向量检索的记忆。
type RetrievalMemory struct {
	store *InMemoryVectorStore
}

// NewRetrievalMemory 创建检索式记忆，k 为每次检索的文档数。
func NewRetrievalMemory(embedder embedding.Embedder, k int) *RetrievalMemory {
	return &RetrievalMemory{store: NewInMemoryVectorStore(embedder, k)}
}

func (m *RetrievalMemory) Name() string { return "RetrievalMemory" }

// AddMessage 将用户输入和 AI 回复作为两个独立文档写入向量库。
func (m *RetrievalMemory) AddMessage(ctx context.Context, userInput, aiResponse string) error {
	docs := []*schema.Document{
		{Content: "用户说: " + userInput},
		{Content: "AI回复: " + aiResponse},
	}
	if _, err := m.store.Store(ctx, docs); err != nil {
		return fmt.Errorf("检索记忆写入失败: %w", err)
	}
	return nil
}

// GetContext 按语义相似度检索 TopK 文档。
func (m *RetrievalMemory) GetContext(ctx context.Context, query string) (string, error) {
	if m.store.Len() == 0 {
		return "记忆中还没有信息。", nil
	}
	docs, err := m.store.Retrieve(ctx, query)
	if err != nil {
		return "无法处理检索查询。", err
	}
	if len(docs) == 0 {
		return "在记忆中找不到任何相关信息。", nil
	}
	parts := make([]string, len(docs))
	for i, d := range docs {
		parts[i] = d.Content
	}
	return "### 从记忆中检索到的相关信息:\n" + strings.Join(parts, "\n---\n"), nil
}

func (m *RetrievalMemory) Clear() {
	m.store.Reset()
	logf("检索式记忆(文档和向量索引)已清除")
}
